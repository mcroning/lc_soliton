"""Clean extracted LC soliton core code from the latest working notebook.

This module intentionally keeps the numerics close to the trusted notebook while
removing notebook UI, ad-hoc execution cells, and Streamlit/package wrappers.
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import numba as nb
    _HAS_NUMBA = True
except Exception:
    nb = None
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @nb.njit(parallel=True, cache=True)
    def _thomas_batched_numba(a, bvec, c, rhs):
        B, n = rhs.shape
        x = np.empty_like(rhs)

        for j in nb.prange(B):
            bj = bvec[j]

            cpv = np.empty(n, dtype=np.complex64)
            dpv = np.empty(n, dtype=np.complex64)

            denom = np.complex64(bj)
            cpv[0] = np.complex64(c) / denom
            dpv[0] = rhs[j, 0] / denom

            for i in range(1, n):
                denom = np.complex64(bj) - np.complex64(a) * cpv[i - 1]
                cpv[i] = np.complex64(c) / denom if i < n - 1 else np.complex64(0.0)
                dpv[i] = (rhs[j, i] - np.complex64(a) * dpv[i - 1]) / denom

            x[j, n - 1] = dpv[n - 1]
            for i in range(n - 2, -1, -1):
                x[j, i] = dpv[i] - cpv[i] * x[j, i + 1]

        return x
else:
    _thomas_batched_numba = None
import numpy as np
from lc_soliton.core.backend import xp_default as cp

try:
    import cupyx.scipy.fft as spfft
except ImportError:
    import scipy.fft as spfft

try:
    from cupyx.scipy.ndimage import gaussian_filter
except ImportError:
    from scipy.ndimage import gaussian_filter
try:
    from lc_soliton.legacy_validated.lc_offload_tools import _thomas_kernel
except Exception:
    _thomas_kernel = None
import scipy.special as spspec
from scipy.optimize import root_scalar
from scipy.signal.windows import tukey
from scipy.ndimage import zoom as sp_zoom

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

j = 1j

from .launch_core import build_theta_bias_IC, build_theta_bias_IC_dirichlet_value, compute_n_bg_from_bias, genrot, build_amp_pair


def _is_numpy_backend_array(a):
    return a.__class__.__module__.split(".")[0] == "numpy"

def _thomas_cpu_solve_banded(lower, diag, upper, rhs):
    import numpy as np
    from scipy.linalg import solve_banded

    lower = np.asarray(lower)
    diag = np.asarray(diag)
    upper = np.asarray(upper)
    rhs = np.asarray(rhs)

    n = diag.size
    ab = np.zeros((3, n), dtype=rhs.dtype)
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]

    return solve_banded((1, 1), ab, rhs)

def prepare_ie_ky_operator(*, dt, mobility, du, dv, Ny):
    a_ie = float(dt) / float(mobility)
    lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=cp).astype(cp.float32, copy=False)
    off = cp.float32((-a_ie) * (1.0 / (du * du)))
    diag = (1.0 + (2.0 * a_ie) * (1.0 / (du * du)) - a_ie * lam_y).astype(cp.float32, copy=False)
    return cp.float32(a_ie), off, diag, lam_y

def intens(amp, coh, out=None):
    xp = cp.get_array_module(amp)
    a, b = amp[0], amp[1]
    if out is None:
        out = xp.empty(a.shape, a.real.dtype)
    if coh:
        s = a + b
        out[...] = (s.real * s.real + s.imag * s.imag)
    else:
        out[...] = (a.real * a.real + a.imag * a.imag + b.real * b.real + b.imag * b.imag)
    return out

def intens_into(out_I32, amp, *, coh: bool):
    if coh:
        s = amp[0] + amp[1]
        out_I32[...] = (s.real * s.real + s.imag * s.imag).astype(cp.float32, copy=False)
    else:
        a0 = amp[0]
        a1 = amp[1]
        out_I32[...] = (
            a0.real * a0.real + a0.imag * a0.imag +
            a1.real * a1.real + a1.imag * a1.imag
        ).astype(cp.float32, copy=False)
    return out_I32

def normalize_Ishape(Ixy, dx, dy):
    norm = cp.sum(Ixy) * dx * dy
    norm = cp.where(norm == 0.0, 1.0, norm)
    return Ixy / norm

def normalize_Ishape_inplace(Ixy, dx, dy):
    norm = cp.sum(Ixy) * cp.float32(dx * dy)
    norm = cp.where(norm == 0.0, 1.0, norm)
    Ixy /= norm
    return Ixy

def hop_linear(amp, hker, windowxy=None):
    A = spfft.fft2(amp, axes=(-2, -1))
    A *= hker
    amp = spfft.ifft2(A, axes=(-2, -1))
    if windowxy is not None:
        amp = amp * windowxy
    return amp

def hop_linear_inplace(amp, hker, windowxy, Ahat, *, plan_f, plan_i):
    if plan_f is None:
        A = spfft.fft2(amp, axes=(-2, -1))
    else:
        with plan_f:
            A = spfft.fft2(amp, axes=(-2, -1))

    A *= hker

    if plan_i is None:
        out = spfft.ifft2(A, axes=(-2, -1))
    else:
        with plan_i:
            out = spfft.ifft2(A, axes=(-2, -1))

    amp[...] = out.astype(cp.complex64, copy=False)

    if windowxy is not None:
        amp *= windowxy

    return amp

def lc_dn_from_theta(theta, ne, no, refin):
    th64 = theta.astype(cp.float64, copy=False)
    ct, st = cp.cos(th64), cp.sin(th64)
    n_eff = (ne * no) / cp.sqrt((ne * ct)**2 + (no * st)**2)
    return (n_eff - refin).astype(cp.float32, copy=False)

def apply_nonlinear_phase_inplace(amp, theta_xy, *, dz_step_um=None, dz_step=None, kout, ne, no, refin):
    if dz_step_um is None:
        dz_step_um = dz_step
    dn = lc_dn_from_theta(theta_xy, ne, no, refin=refin)
    phase = cp.exp((1j * cp.float32(kout * dz_step_um)) * dn).astype(cp.complex64, copy=False)
    amp *= phase
    return amp

def apply_nonlinear_phase(amp, theta_xy, *, dz_step_um=None, dz_step=None, kout, ne, no, refin):
    if dz_step_um is None:
        dz_step_um = dz_step
    out = cp.array(amp, copy=True)
    apply_nonlinear_phase_inplace(
        out, theta_xy,
        dz_step_um=dz_step_um,
        kout=kout, ne=ne, no=no, refin=refin
    )
    return out

def _laplacian_dirichletx_periody(theta, du, dv):
    th = theta.astype(cp.float32, copy=False)

    th_yplus = cp.roll(th, -1, axis=1)
    th_yminus = cp.roll(th, +1, axis=1)

    du2 = cp.float32(du * du)
    dv2 = cp.float32(dv * dv)

    lap = cp.zeros_like(th)

    # interior ONLY
    lap[1:-1, :] = (
        (th[2:, :] - 2.0 * th[1:-1, :] + th[:-2, :]) / du2
        + (th_yplus[1:-1, :] - 2.0 * th[1:-1, :] + th_yminus[1:-1, :]) / dv2
    )

    # boundary rows remain zero
    return lap

def _lam_y_periodic_second_diff(Ny, dv, xp=cp):
    k = xp.arange(Ny, dtype=xp.float32)
    return (-4.0 * xp.sin(xp.pi * k / Ny)**2) / (dv * dv)




def thomas_batched_const_tridiag(a, bvec, c, d_hatB):
    B, n = d_hatB.shape

    # CPU fallback: d_hatB shape is (B, n), each row is one RHS.
    # scipy solve_banded expects RHS columns, so solve transposed.
    if _thomas_kernel is None:
        import numpy as np
        from scipy.linalg import solve_banded

        rhs = np.asarray(d_hatB, dtype=np.complex64)
        B, n = rhs.shape
        b = np.asarray(bvec, dtype=np.float32)

        if b.ndim > 0 and b.size == B and _thomas_batched_numba is not None:
            x = _thomas_batched_numba(
                np.float32(a),
                b.astype(np.float32, copy=False),
                np.float32(c),
                rhs,
            )
            return cp.asarray(x, dtype=cp.complex64)

        # universal SciPy fallback
        x = np.empty_like(rhs)
        for j in range(B):
            bj = float(b[j]) if b.ndim > 0 and b.size == B else float(b)

            ab = np.zeros((3, n), dtype=np.complex64)
            ab[0, 1:] = np.complex64(c)
            ab[1, :] = np.complex64(bj)
            ab[2, :-1] = np.complex64(a)

            x[j, :] = solve_banded((1, 1), ab, rhs[j, :])

        return cp.asarray(x, dtype=cp.complex64)

    d_hatB = cp.ascontiguousarray(d_hatB.astype(cp.complex64, copy=False))
    x = cp.empty_like(d_hatB)
    cprime = cp.empty_like(d_hatB)
    dprime = cp.empty_like(d_hatB)

    threads = 128
    blocks = (B + threads - 1) // threads

    _thomas_kernel(
        (blocks,), (threads,),
        (
            cp.float32(a),
            bvec.astype(cp.float32),
            cp.float32(c),
            d_hatB.view(cp.float32).reshape(B, n, 2),
            x.view(cp.float32).reshape(B, n, 2),
            cprime.view(cp.float32).reshape(B, n, 2),
            dprime.view(cp.float32).reshape(B, n, 2),
            cp.int32(B),
            cp.int32(n),
        ),
    )
    return x

def strict_static_relax_slice_selfconsistent(
    amp_in,
    theta_seed,
    tp,
    tn,
    ctx,
    *,
    dz_sub,
    Nsub,
    h_sub,
    Ahat,
    plan_f,
    plan_i,
    sS,
    offS,
    diagS,
    lamS,
    use_linear_seed=True,
    early_accept_linear_seed=True,
    residual_tol_max=5e-2,
    residual_tol_rms=1e-2,
    max_outer_passes=8,
    max_selfcons_passes=3,
    selfcons_tol_theta=1e-4,
    selfcons_tol_I=1e-4,
    verbose=False,
):
    """
    Self-consistent strict static slice solve for gamma_z = 0 case.

    Returns
    -------
    theta_out : cp.ndarray
    I_mid_out : cp.ndarray
    amp_out   : cp.ndarray
    info      : dict
    """
    # incoming intensity
    I_b = cp.empty((ctx.Nx, ctx.Ny), dtype=cp.float32)
    intens_into(I_b, amp_in, coh=ctx.coh)

    # work buffers
    amp_work = cp.empty_like(amp_in)
    I_a = cp.empty_like(I_b)
    I_mid = cp.empty_like(I_b)

    # initial theta guess
    theta = theta_seed.astype(cp.float32, copy=False)
    theta = enforce_theta_constraints(theta, ctx.theta_clamp)

    last_I_mid = None
    last_info = None
    converged = False

    for sc_pass in range(max_selfcons_passes):
        # optics with current theta
        amp_work[...] = amp_in
        for _ in range(Nsub):
            apply_nonlinear_phase_inplace(
                amp_work, theta,
                dz_step_um=dz_sub,
                kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
            )
            hop_linear_inplace(
                amp_work, h_sub, ctx.windowxy, Ahat,
                plan_f=plan_f, plan_i=plan_i
            )

        intens_into(I_a, amp_work, coh=ctx.coh)
        fill_mid_intensity(I_mid, I_b, I_a)

        # strict theta solve at this midpoint forcing
        theta_new, info = strict_static_relax_slice(
            theta,
            I_mid,
            tp,
            tn,
            ctx,
            sS=sS,
            offS=offS,
            diagS=diagS,
            lamS=lamS,
            use_linear_seed=(use_linear_seed and sc_pass == 0),
            early_accept_linear_seed=(early_accept_linear_seed and sc_pass == 0),
            residual_tol_max=residual_tol_max,
            residual_tol_rms=residual_tol_rms,
            max_outer_passes=max_outer_passes,
            verbose=verbose,
        )

        theta = enforce_theta_constraints(theta, ctx.theta_clamp)

        dtheta = float(cp.max(cp.abs(theta_new - theta)))

        if last_I_mid is None:
            dI = cp.inf
        else:
            denom = max(float(cp.max(cp.abs(last_I_mid))), 1e-12)
            dI = float(cp.max(cp.abs(I_mid - last_I_mid)) / denom)

        if verbose:
            print(
                f"[strict selfcons] pass={sc_pass+1:2d} "
                f"dtheta={dtheta:.3e} dI={dI:.3e} "
                f"max={info['max_interior']:.3e} rms={info['rms_interior']:.3e}"
            )

        theta = theta_new
        last_info = info

        if last_I_mid is None:
            last_I_mid = I_mid.copy()
        else:
            last_I_mid[...] = I_mid

        if (
            dtheta <= selfcons_tol_theta
            and dI <= selfcons_tol_I
            and info["max_interior"] <= residual_tol_max
            and info["rms_interior"] <= residual_tol_rms
        ):
            converged = True
            break

    return theta, I_mid, amp_work, {
        "converged": converged,
        "n_selfcons_passes": sc_pass + 1,
        "max_interior": last_info["max_interior"],
        "rms_interior": last_info["rms_interior"],
    }

def fill_mid_intensity(I_mid, I_b, I_a):
    I_mid[...] = 0.5 * (I_b + I_a)
    return I_mid

def td_get_slice_mid_intensity(
    ctx,
    amp,
    theta_use,
    I_b,
    I_a,
    I_mid,
    *,
    Nsub,
    dz_sub,
    h_sub,
    Ahat,
    plan_f,
    plan_i,
    frozen_I_mid_k=None,
):
    """
    Fill I_mid for one TD slice.

    If frozen_I_mid_k is provided, use it directly and leave amp unchanged.
    Otherwise march optics through the slice using theta_use.
    Returns
    -------
    amp_after : cp.ndarray
        The optical field after this slice (same object as amp if marched in place).
    """
    if frozen_I_mid_k is not None:
        I_mid[...] = frozen_I_mid_k
        return amp

    for _ in range(Nsub):
        apply_nonlinear_phase_inplace(
            amp, theta_use,
            dz_step_um=dz_sub,
            kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
        )
        hop_linear_inplace(
            amp, h_sub, ctx.windowxy, Ahat,
            plan_f=plan_f, plan_i=plan_i
        )

    intens_into(I_a, amp, coh=ctx.coh)
    fill_mid_intensity(I_mid, I_b, I_a)
    return amp

def td_update_theta_from_midintensity(
    ctx,
    theta_ref_k,
    tp,
    tn,
    I_mid,
    *,
    dt_j,
    true_td,
    a_ie,
    s,
    off,
    diag,
    lam_y,
    Nsub,
    dz_sub,
    h_sub,
    amp_for_pred=None,
    I_b=None,
    I_a=None,
    Ahat=None,
    plan_f=None,
    plan_i=None,
):
    """
    Update theta for one TD slice given I_mid.

    Returns
    -------
    theta_new : cp.ndarray
    """
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))
    predictor_only = bool(getattr(ctx, "true_td_predictor_only", False))
    do_full_pred = bool(getattr(ctx, "true_td_full_pred_optics", False))

    if not true_td:
        return relax_theta_to_current_I_td_zcoupled(
            theta_ref_k,
            I_mid,
            tp,
            tn,
            ctx,
            dt=dt_j,
            s=s,
            off=off,
            diag=diag,
            lam_y=lam_y,
        )

    # true physical-time branch
    if gamma_z != 0.0:
        theta_new = advance_theta_physical_timestep_semiimplicit_zcoupled(
            theta_ref_k,
            dt=dt_j,
            b=ctx.b,
            bi=ctx.bi,
            Ixy=I_mid,
            theta_prev_n=tp,
            theta_next_n=tn,
            gamma_z=gamma_z,
            mobility=ctx.mobility,
            du=ctx.du,
            dv=ctx.dv,
            a_ie=a_ie,
            off=off,
            diag=diag,
            lam_y=lam_y,
            inv_dz2=_get_inv_dz2(ctx),
            clamp=ctx.theta_clamp,
        )
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        theta_new[0, :] = theta_bc
        theta_new[-1, :] = theta_bc
        return theta_new

    # gamma_z == 0: predictor/corrector route
    theta_pred_k = advance_theta_physical_timestep_semiimplicit_zcoupled(
        theta_ref_k,
        dt=dt_j,
        b=ctx.b,
        bi=ctx.bi,
        Ixy=I_mid,
        theta_prev_n=tp,
        theta_next_n=tn,
        gamma_z=0.0,
        mobility=ctx.mobility,
        du=ctx.du,
        dv=ctx.dv,
        a_ie=a_ie,
        off=off,
        diag=diag,
        lam_y=lam_y,
        inv_dz2=_get_inv_dz2(ctx),
        clamp=ctx.theta_clamp,
    )

    if predictor_only:
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        theta_pred_k[0, :] = theta_bc
        theta_pred_k[-1, :] = theta_bc
        return theta_pred_k

    if do_full_pred:
        amp_pred_buf = ctx._amp_pred_buf
        amp_pred_buf[...] = amp_for_pred

        for _ in range(Nsub):
            apply_nonlinear_phase_inplace(
                amp_pred_buf, theta_pred_k,
                dz_step_um=dz_sub,
                kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
            )
            hop_linear_inplace(
                amp_pred_buf, h_sub, ctx.windowxy, Ahat,
                plan_f=plan_f, plan_i=plan_i
            )

        I_mid_pred = ctx._I_mid_pred_buf
        intens_into(I_a, amp_pred_buf, coh=ctx.coh)
        fill_mid_intensity(I_mid_pred, I_b, I_a)
        I_pic_use = I_mid_pred
    else:
        I_pic_use = I_mid

    s_cn, off_cn, diag_cn, lam_y_cn = _get_cn_ky_operator_cached(ctx, dt_j)


    theta_new = advance_theta_timestep_cn_trap_picard_prepared(
        theta_ref_k,
        dt=dt_j,
        b=ctx.b,
        bi=ctx.bi,
        I_n=I_mid,
        I_pic=I_pic_use,
        mobility=ctx.mobility,
        du=ctx.du,
        dv=ctx.dv,
        s=s_cn,
        off=off_cn,
        diag=diag_cn,
        lam_y=lam_y_cn,
        max_iter=getattr(ctx, "picard_iters", 4),
        tol_update=getattr(ctx, "picard_tol_up", 1e-6),
        clamp=ctx.theta_clamp,
    )

    theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
    theta_new[0, :] = theta_bc
    theta_new[-1, :] = theta_bc

    return theta_new

def make_coarse_theta_ctx(ctx_f, factor=2):
    import copy

    ctx_t = copy.copy(ctx_f)   # shallow, not deepcopy

    ctx_t.Nx = int(ctx_f.Nx // factor)
    ctx_t.Ny = int(ctx_f.Ny // factor)

    ctx_t.dx = float(ctx_f.dx) * factor
    ctx_t.dy = float(ctx_f.dy) * factor
    ctx_t.du = float(ctx_f.du) * factor
    ctx_t.dv = float(ctx_f.dv) * factor

    ctx_t._h_cache = {}
    ctx_t._cn_ky_cache = {}

    ctx_t.fxy2 = None
    ctx_t.windowxy = None

    ctx_t.theta_bias_2d = cp.empty((ctx_t.Nx, ctx_t.Ny), dtype=cp.float32)
    ctx_t.theta_full = None
    ctx_t.theta_prev_time = None

    return ctx_t

# @title YZ movie streamer

def enforce_theta_constraints(theta, theta_clamp):
    """
    Enforce optional clamp and Dirichlet-x boundary conditions.

    Parameters
    ----------
    theta : cp.ndarray
        Array of shape (Nx, Ny).
    theta_clamp : tuple[float, float] or None
        Optional (theta_min, theta_max).

    Returns
    -------
    cp.ndarray
        Constrained theta field.
    """
    if theta_clamp is not None:
        theta = cp.clip(
            theta,
            cp.float32(theta_clamp[0]),
            cp.float32(theta_clamp[1]),
        )
    return theta

def _get_inv_dz2(ctx):
    dz = float(ctx.dz)
    return cp.float32(1.0 / (dz * dz))

def choose_optics_substeps(dz_um, *, kout, dn_max_est, dz_opt_max_phi, max_substeps):
    phi_est = abs(float(kout)) * abs(float(dz_um)) * abs(float(dn_max_est))
    nsub = int(np.ceil(phi_est / float(dz_opt_max_phi))) if phi_est > 0 else 1
    nsub = max(1, min(int(max_substeps), nsub))
    dz_sub = float(dz_um) / nsub
    return nsub, dz_sub, phi_est

def get_h_for_dz(ctx, dz_step_um):
    key = float(np.round(float(dz_step_um), 12))
    if key in ctx._h_cache:
        return ctx._h_cache[key]

    lm = ctx.lm
    refin = ctx.refin
    fxy2 = ctx.fxy2

    h = cp.where(
        (lm / refin)**2 * fxy2 < 1.0,
        cp.exp(
            2.0j * cp.pi * refin * cp.float32(dz_step_um) / lm
            * cp.sqrt(1.0 - (lm / refin)**2 * fxy2)
        ),
        0.0
    ).astype(cp.complex64, copy=False)

    ctx._h_cache[key] = h
    return h

def prepare_cn_ky_operator(*, dt, mobility, du, dv, Ny):
    mob = float(mobility)
    s = float(dt) / (2.0 * mob)
    lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=cp).astype(cp.float32, copy=False)
    off = cp.float32((-s) * (1.0 / (du * du)))
    diag = (1.0 + (2.0 * s) * (1.0 / (du * du)) - s * lam_y).astype(cp.float32, copy=False)
    return cp.float32(s), off, diag, lam_y

def prepare_ie_ky_operator(*, dt, mobility, du, dv, Ny):
    a_ie = float(dt) / float(mobility)
    lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=cp).astype(cp.float32, copy=False)
    off = cp.float32((-a_ie) * (1.0 / (du * du)))
    diag = (1.0 + (2.0 * a_ie) * (1.0 / (du * du)) - a_ie * lam_y).astype(cp.float32, copy=False)
    return cp.float32(a_ie), off, diag, lam_y

def _get_cn_ky_operator_cached(ctx, dt):
    """
    Cache CN operator by (dt, mobility, du, dv, Ny).
    """
    if not hasattr(ctx, "_cn_ky_cache"):
        ctx._cn_ky_cache = {}

    key = (
        float(dt),
        float(ctx.mobility),
        float(ctx.du),
        float(ctx.dv),
        int(ctx.Ny),
    )

    out = ctx._cn_ky_cache.get(key, None)
    if out is None:
        out = prepare_cn_ky_operator(
            dt=dt,
            mobility=ctx.mobility,
            du=ctx.du,
            dv=ctx.dv,
            Ny=ctx.Ny,
        )
        ctx._cn_ky_cache[key] = out

    return out

def _cn_solve_dirichletx_periody(rhs, *, off, diag, theta_bc=0.0):
    theta_bc = cp.asarray(theta_bc, dtype=cp.float32)

    rhs2 = rhs.astype(cp.float32, copy=True)
    rhs2[0, :] = theta_bc
    rhs2[-1, :] = theta_bc

    rhs_hat = cp.fft.fft(rhs2, axis=1).astype(cp.complex64, copy=False)
    d_hatB = rhs_hat[1:-1, :].T.copy(order="C")  # (Ny, Nx-2)

    Ny_local = rhs.shape[1]
    bc_hat = cp.fft.fft(
        cp.full((Ny_local,), theta_bc, dtype=cp.float32)
    ).astype(cp.complex64)

    d_hatB[:, 0]  -= off * bc_hat
    d_hatB[:, -1] -= off * bc_hat

    x_hatB = thomas_batched_const_tridiag(off, diag, off, d_hatB)

    theta_hat = cp.zeros_like(rhs_hat)
    theta_hat[1:-1, :] = x_hatB.T

    th = cp.fft.ifft(theta_hat, axis=1).real.astype(cp.float32, copy=False)
    th[0, :] = theta_bc
    th[-1, :] = theta_bc
    return th

def advance_theta_timestep_cn_fft_thomas_prepared(
    theta_n, *, dt, b, bi, Ixy,
    mobility, du, dv,
    s, off, diag, lam_y
):
    theta_n = theta_n.astype(cp.float32, copy=False)
    Ixy = Ixy.astype(cp.float32, copy=False)

    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv)).astype(cp.float32, copy=False)
    alpha = cp.float32(b) + cp.float32(bi) * Ixy
    drive_n = alpha * cp.sin(2.0 * theta_n)

    rhs = theta_n + s * lap_n + cp.float32(float(dt) / float(mobility)) * drive_n
    theta_bc = theta_n[0, 0].astype(cp.float32)
    rhs[0, :] = theta_bc
    rhs[-1, :] = theta_bc
    return _cn_solve_dirichletx_periody(rhs, off=off, diag=diag, theta_bc=theta_bc)    
#    rhs[0, :] = 0.0
#    rhs[-1, :] = 0.0
#    return _cn_solve_dirichletx_periody(rhs, off=off, diag=diag)

def advance_theta_timestep_cn_trap_picard_prepared(
    theta_n, *, dt, b, bi, I_n, I_pic,
    mobility, du, dv,
    s, off, diag, lam_y,
    max_iter, tol_update, clamp
):


    theta_n = theta_n.astype(cp.float32, copy=False)
    I_n = I_n.astype(cp.float32, copy=False)
    I_pic = I_pic.astype(cp.float32, copy=False)

    alpha_old = cp.float32(b) + cp.float32(bi) * I_n
    alpha_pic = cp.float32(b) + cp.float32(bi) * I_pic

    lap_n = _laplacian_dirichletx_periody(
        theta_n, float(du), float(dv)
    ).astype(cp.float32, copy=False)

    half_dt_over_m = cp.float32(0.5 * float(dt) / float(mobility))
    N_n = alpha_old * cp.sin(2.0 * theta_n)

    rhs_base = theta_n + s * lap_n + half_dt_over_m * N_n
    theta_bc = theta_n[0, 0].astype(cp.float32)
    rhs_base[0, :] = theta_bc
    rhs_base[-1, :] = theta_bc

    theta_g = advance_theta_timestep_cn_fft_thomas_prepared(
        theta_n,
        dt=dt, b=b, bi=bi, Ixy=I_n,
        mobility=mobility, du=du, dv=dv,
        s=s, off=off, diag=diag, lam_y=lam_y
    )

    for _ in range(int(max_iter)):
        N_g = alpha_pic * cp.sin(2.0 * theta_g)

        rhs = rhs_base + half_dt_over_m * N_g
        rhs[0, :] = theta_bc
        rhs[-1, :] = theta_bc

#        theta_new = _cn_solve_dirichletx_periody(rhs, off=off, diag=diag)
        theta_new = _cn_solve_dirichletx_periody(
            rhs,
            off=off,
            diag=diag,
            theta_bc=theta_bc,
        )
        theta_new = enforce_theta_constraints(theta_new, clamp)

        dth = theta_new - theta_g
        rms_up = float(cp.sqrt(cp.mean(dth * dth)))
        theta_g = theta_new

        if rms_up < float(tol_update):
            break

    return theta_g

def advance_theta_physical_timestep_semiimplicit_zcoupled(
    theta_n, *, dt, b, bi, Ixy,
    theta_prev_n, theta_next_n, gamma_z,
    mobility, du, dv,
    a_ie, off, diag, lam_y,
    inv_dz2,
    clamp=None
):
    theta_n = theta_n.astype(cp.float32, copy=False)
    Ixy = Ixy.astype(cp.float32, copy=False)
    tp = theta_prev_n.astype(cp.float32, copy=False)
    tn = theta_next_n.astype(cp.float32, copy=False)

    alpha = cp.float32(b) + cp.float32(bi) * Ixy
    drive_n = alpha * cp.sin(2.0 * theta_n)

    gam = cp.float32(gamma_z) * inv_dz2
    dt_over_m = cp.float32(float(dt) / float(mobility))

#    rhs = theta_n + dt_over_m * (drive_n + gam * (tp + tn))
#    
#    rhs[0, :] = 0.0
#    rhs[-1, :] = 0.0

#    diag_eff = diag + cp.float32(2.0) * dt_over_m * gam#
#    theta_np1 = _cn_solve_dirichletx_periody(rhs, off=off, diag=diag_eff)

    rhs = theta_n + dt_over_m * (drive_n + gam * (tp + tn))

    theta_bc = theta_n[0, 0].astype(cp.float32)
    
    rhs[0, :] = theta_bc
    rhs[-1, :] = theta_bc
    
    diag_eff = diag + cp.float32(2.0) * dt_over_m * gam
    
    theta_np1 = _cn_solve_dirichletx_periody(
        rhs,
        off=off,
        diag=diag_eff,
        theta_bc=theta_bc,
    )
    
    return enforce_theta_constraints(theta_np1, clamp)

def advance_theta_timestep_cn_fft_thomas_prepared_zcoupled(
    theta_n, *, dt, b, bi, Ixy,
    theta_prev, theta_next, gamma_z,
    mobility, du, dv,
    s, off, diag, lam_y,
    inv_dz2
):
    theta_n = theta_n.astype(cp.float32, copy=False)
    Ixy = Ixy.astype(cp.float32, copy=False)
    tp = theta_prev.astype(cp.float32, copy=False)
    tn = theta_next.astype(cp.float32, copy=False)

    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv)).astype(cp.float32, copy=False)
    alpha = cp.float32(b) + cp.float32(bi) * Ixy
    drive_n = alpha * cp.sin(2.0 * theta_n)

    gam = cp.float32(gamma_z) * inv_dz2
    dt_over_m = cp.float32(float(dt) / float(mobility))

    rhs = theta_n + s * lap_n + dt_over_m * (drive_n + gam * (tp + tn))

    theta_bc = theta_n[0, 0].astype(cp.float32)
    rhs[0, :] = theta_bc
    rhs[-1, :] = theta_bc

    diag_eff = diag + cp.float32(2.0) * dt_over_m * gam

    theta_np1 = _cn_solve_dirichletx_periody(
        rhs,
        off=off,
        diag=diag_eff,
        theta_bc=theta_bc,
    )

    return theta_np1

def advance_theta_timestep_cn_trap_picard_prepared_zcoupled(
    theta_n, *, dt, b, bi, Ixy,
    theta_prev, theta_next, gamma_z,
    mobility, du, dv,
    s, off, diag, lam_y,
    max_iter, tol_update, clamp,
    inv_dz2
):
    theta_n = theta_n.astype(cp.float32, copy=False)
    Ixy = Ixy.astype(cp.float32, copy=False)
    tp = theta_prev.astype(cp.float32, copy=False)
    tn = theta_next.astype(cp.float32, copy=False)

    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv)).astype(cp.float32, copy=False)
    alpha = cp.float32(b) + cp.float32(bi) * Ixy
    N_n = alpha * cp.sin(2.0 * theta_n)

    gam = cp.float32(gamma_z) * inv_dz2
    dt_over_m = cp.float32(float(dt) / float(mobility))

    rhs_base = (
        theta_n
        + s * lap_n
        + cp.float32(0.5) * dt_over_m * N_n
        + dt_over_m * gam * (tp + tn)
    )
    #rhs_base[0, :] = 0.0
    #rhs_base[-1, :] = 0.0
    theta_bc = theta_n[0, 0].astype(cp.float32)
    rhs_base[0, :] = theta_bc
    rhs_base[-1, :] = theta_bc

    diag_eff = diag + cp.float32(2.0) * dt_over_m * gam

    theta_g = advance_theta_timestep_cn_fft_thomas_prepared_zcoupled(
        theta_n,
        dt=dt, b=b, bi=bi, Ixy=Ixy,
        theta_prev=tp, theta_next=tn, gamma_z=gamma_z,
        mobility=mobility, du=du, dv=dv,
        s=s, off=off, diag=diag, lam_y=lam_y,
        inv_dz2=inv_dz2
    )

    for _ in range(int(max_iter)):
        N_g = alpha * cp.sin(2.0 * theta_g)
        rhs = rhs_base + cp.float32(0.5) * dt_over_m * N_g
        rhs[0, :] = theta_bc
        rhs[-1, :] = theta_bc

        #theta_new = _cn_solve_dirichletx_periody(rhs, off=off, diag=diag_eff)
        theta_new = _cn_solve_dirichletx_periody(
            rhs,
            off=off,
            diag=diag_eff,
            theta_bc=theta_bc,
        )
        theta_new = enforce_theta_constraints(theta_new, clamp)

        dth = theta_new - theta_g
        rms_up = float(cp.sqrt(cp.mean(dth * dth)))
        theta_g = theta_new

        if rms_up < float(tol_update):
            break

    return theta_g

def relax_theta_to_current_I_td_zcoupled(
    theta_init, Ixy, theta_prev, theta_next, ctx, *,
    dt, s, off, diag, lam_y
):
    th = theta_init.astype(cp.float32, copy=True)
    I32 = Ixy.astype(cp.float32, copy=False)

    nrelax = max(1, int(ctx.td_theta_relax_steps))
    omega = float(ctx.td_theta_relax_omega)
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))
    inv_dz2 = _get_inv_dz2(ctx)

    for _ in range(nrelax):
        th_new = advance_theta_timestep_cn_trap_picard_prepared_zcoupled(
            th,
            dt=float(dt),
            b=ctx.b,
            bi=ctx.bi,
            Ixy=I32,
            theta_prev=theta_prev,
            theta_next=theta_next,
            gamma_z=gamma_z,
            mobility=ctx.mobility,
            du=ctx.du,
            dv=ctx.dv,
            s=s,
            off=off,
            diag=diag,
            lam_y=lam_y,
            max_iter=ctx.picard_iters,
            tol_update=ctx.picard_tol_up,
            clamp=ctx.theta_clamp,
            inv_dz2=inv_dz2,
        )

        th = (1.0 - omega) * th + omega * th_new
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        th[0, :] = theta_bc
        th[-1, :] = theta_bc

        if ctx.theta_clamp is not None:
            th = cp.clip(th, cp.float32(ctx.theta_clamp[0]), cp.float32(ctx.theta_clamp[1]))

    return th

def lc_residual64(theta, Ixy, *, b, bi, du, dv, mobility=1.0):
    th = theta.astype(cp.float64, copy=False)
    I64 = Ixy.astype(cp.float64, copy=False)

    du2 = cp.float64(du) * cp.float64(du)
    dv2 = cp.float64(dv) * cp.float64(dv)

    th_yplus = cp.roll(th, -1, axis=1)
    th_yminus = cp.roll(th, +1, axis=1)

    lap = cp.zeros_like(th)

    lap[1:-1, :] = (
        (th[2:, :] - 2.0 * th[1:-1, :] + th[:-2, :]) / du2
        +
        (th_yplus[1:-1, :] - 2.0 * th[1:-1, :] + th_yminus[1:-1, :]) / dv2
    )

    drive = (cp.float64(b) + cp.float64(bi) * I64) * cp.sin(2.0 * th)

    R = (lap + drive) / cp.float64(mobility)

    # prescribed Dirichlet rows are not residual-tested
    R[0, :] = 0.0
    R[-1, :] = 0.0

    return R

def lc_residual2d_dirichletx_periody(theta, Ixy, *, b, bi, du, dv, mobility=1.0):
    return lc_residual64(theta, Ixy, b=b, bi=bi, du=du, dv=dv, mobility=mobility)

def residual_stats_2d(R):
    R = R.astype(cp.float64, copy=False)
    Rint = R[1:-1, :]
    return {
        "max_full": float(cp.max(cp.abs(R))),
        "rms_full": float(cp.sqrt(cp.mean(R * R))),
        "max_interior": float(cp.max(cp.abs(Rint))),
        "rms_interior": float(cp.sqrt(cp.mean(Rint * Rint))),
    }

def _slice_residual_rms_local3d_z(theta_k, I_k, theta_prev, theta_next, ctx, gamma_z):
    th = theta_k.astype(cp.float64, copy=False)
    I64 = I_k.astype(cp.float64, copy=False)
    tp = theta_prev.astype(cp.float64, copy=False)
    tn = theta_next.astype(cp.float64, copy=False)

    theta_bc = cp.float64(getattr(ctx, "theta_bc", 0.0))

    du = cp.float64(ctx.du)
    dv = cp.float64(ctx.dv)
    dz = cp.float64(ctx.dz)

    du2 = du * du
    dv2 = dv * dv
    dz2 = dz * dz

    th_yplus = cp.roll(th, -1, axis=1)
    th_yminus = cp.roll(th, +1, axis=1)

    th_xplus = cp.empty_like(th)
    th_xminus = cp.empty_like(th)
    th_xplus[:-1, :] = th[1:, :]
    th_xplus[-1, :] = theta_bc
    th_xminus[1:, :] = th[:-1, :]
    th_xminus[0, :] = theta_bc

    lap_xy = (th_xplus - 2.0 * th + th_xminus) / du2 + (th_yplus - 2.0 * th + th_yminus) / dv2
    lap_xy[0, :] = 0.0
    lap_xy[-1, :] = 0.0

    lap_z = cp.float64(gamma_z) * (tn - 2.0 * th + tp) / dz2
    lap_z[0, :] = 0.0
    lap_z[-1, :] = 0.0

    drive = (cp.float64(ctx.b) + cp.float64(ctx.bi) * I64) * cp.sin(2.0 * th)
    R = (lap_xy + lap_z + drive) / cp.float64(ctx.mobility)
    R[0, :] = 0.0
    R[-1, :] = 0.0
    return float(cp.sqrt(cp.mean(R * R)))

def thomas_batched_modevarying(off_lower, diag, off_upper, rhs, xp):
    rhs = xp.asarray(rhs)
    diag = xp.asarray(diag)
    Nx, Ny = rhs.shape

    dtype = xp.result_type(diag.dtype, rhs.dtype)
    a_in = xp.asarray(off_lower, dtype=dtype)
    c_in = xp.asarray(off_upper, dtype=dtype)

    if a_in.ndim == 1:
        a = xp.broadcast_to(a_in[:, None], (Nx - 1, Ny)).copy()
    else:
        a = xp.broadcast_to(a_in, (Nx - 1, Ny)).copy()

    if c_in.ndim == 1:
        c = xp.broadcast_to(c_in[:, None], (Nx - 1, Ny)).copy()
    else:
        c = xp.broadcast_to(c_in, (Nx - 1, Ny)).copy()

    bdiag = xp.asarray(diag, dtype=dtype).copy()
    d = xp.asarray(rhs, dtype=dtype).copy()

    for i in range(1, Nx):
        w = a[i - 1] / bdiag[i - 1]
        bdiag[i] = bdiag[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    x = xp.empty_like(d)
    x[-1] = d[-1] / bdiag[-1]
    for i in range(Nx - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / bdiag[i]

    return x

def _extract_theta_b_1d(theta_bias, xp):
    th = xp.asarray(theta_bias)
    if th.ndim == 1:
        return th
    if th.ndim == 2:
        return xp.mean(th, axis=1)
    raise ValueError("theta_bias must have shape (Nx,) or (Nx,Ny).")

def apply_Lpert_delta(delta, theta_b, b, du, dv, xp):
    theta_b = xp.asarray(theta_b)
    delta = xp.asarray(delta)

    qx = 2.0 * b * xp.cos(2.0 * theta_b)

    left = xp.zeros_like(delta)
    right = xp.zeros_like(delta)
    left[1:, :] = delta[:-1, :]
    right[:-1, :] = delta[1:, :]
    d2x = (left - 2.0 * delta + right) / (du * du)
    d2y = (xp.roll(delta, 1, axis=1) - 2.0 * delta + xp.roll(delta, -1, axis=1)) / (dv * dv)

    return d2x + d2y + qx[:, None] * delta

def solve_theta_perturbation_dirichletx_periody(
    theta_bias, Ixy, *, b, bi, du, dv,
    xp=None, return_theta=True, check_residual=True, verbose=False
):
    if xp is None:
        xp = cp.get_array_module(Ixy)

    Ixy = xp.asarray(Ixy, dtype=xp.float64)
    theta_b_1d = _extract_theta_b_1d(theta_bias, xp).astype(xp.float64, copy=False)

    Nx, Ny = Ixy.shape
    qx = 2.0 * b * xp.cos(2.0 * theta_b_1d)
    rhs = -bi * Ixy * xp.sin(2.0 * theta_b_1d)[:, None]

    qx_i = qx[1:-1]
    rhs_i = rhs[1:-1, :]

    rhs_hat_i = xp.fft.fft(rhs_i, axis=1)
    k = xp.arange(Ny, dtype=xp.float64)
    lam_y = (2.0 * xp.cos(2.0 * xp.pi * k / Ny) - 2.0) / (dv * dv)

    off = xp.full((Nx - 3,), 1.0 / (du * du), dtype=xp.float64)
    diag = (-2.0 / (du * du)) + qx_i[:, None] + lam_y[None, :]

    delta_hat_i = thomas_batched_modevarying(off, diag, off, rhs_hat_i, xp)
    delta_i = xp.fft.ifft(delta_hat_i, axis=1).real

    delta = xp.zeros((Nx, Ny), dtype=xp.float64)
    delta[1:-1, :] = delta_i

    out = {"delta": delta, "rhs": rhs, "lam_y": lam_y, "qx": qx}

    if return_theta:
        theta2d = theta_b_1d[:, None] + delta
        theta2d[0, :] = theta_b_1d[0]
        theta2d[-1, :] = theta_b_1d[-1]
        out["theta"] = theta2d

    if check_residual:
        lin_res = apply_Lpert_delta(delta, theta_b_1d, b=b, du=du, dv=dv, xp=xp) - rhs
        lin_res[0, :] = 0.0
        lin_res[-1, :] = 0.0
        out["linear_residual"] = lin_res

    return out

def _static_relax_to_resid_zcoupled(theta_seed, Ixy, theta_prev, theta_next, ctx, *, sS, offS, diagS, lamS):
    th = theta_seed.astype(cp.float32, copy=True)
    I32 = Ixy.astype(cp.float32, copy=False)
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))
    inv_dz2 = _get_inv_dz2(ctx)

    max_steps = int(getattr(ctx, "static_max_steps", 200))
    resid_every = int(getattr(ctx, "static_resid_every", 10))
    tol = float(getattr(ctx, "static_tol_resid", 0.08))
    omega = float(getattr(ctx, "static_relax_omega", 0.3))

    for it in range(max_steps):
        th_new = advance_theta_timestep_cn_trap_picard_prepared_zcoupled(
            th,
            dt=float(ctx.dtau_static),
            b=ctx.b,
            bi=ctx.bi,
            Ixy=I32,
            theta_prev=theta_prev,
            theta_next=theta_next,
            gamma_z=gamma_z,
            mobility=ctx.mobility,
            du=ctx.du,
            dv=ctx.dv,
            s=sS,
            off=offS,
            diag=diagS,
            lam_y=lamS,
            max_iter=ctx.picard_iters,
            tol_update=ctx.picard_tol_up,
            clamp=ctx.theta_clamp,
            inv_dz2=inv_dz2,
        )

        th = (1.0 - omega) * th + omega * th_new
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        th[0, :] = theta_bc
        th[-1, :] = theta_bc

        if ctx.theta_clamp is not None:
            th = cp.clip(th, cp.float32(ctx.theta_clamp[0]), cp.float32(ctx.theta_clamp[1]))

        if (it % resid_every) == 0 or it == (max_steps - 1):
            rloc = _slice_residual_rms_local3d_z(th, I32, theta_prev, theta_next, ctx, gamma_z)
            if rloc < tol:
                break

    return th

def strict_static_relax_slice(
    theta_seed, I_mid, tp, tn, ctx, *,
    sS, offS, diagS, lamS,
    use_linear_seed=True,
    early_accept_linear_seed=True,
    residual_tol_max=5e-2,
    residual_tol_rms=1e-2,
    max_outer_passes=8,
    verbose=False,
):
    if use_linear_seed:
        out_lin = solve_theta_perturbation_dirichletx_periody(
            theta_bias=ctx.theta_bias_2d,
            Ixy=I_mid,
            b=float(ctx.b),
            bi=float(ctx.bi),
            du=float(ctx.du),
            dv=float(ctx.dv),
            xp=cp,
            return_theta=True,
            check_residual=False,
            verbose=False,
        )
        theta = out_lin["theta"].astype(cp.float32, copy=False)
    else:
        theta = theta_seed.astype(cp.float32, copy=False)

    theta = enforce_theta_constraints(theta, ctx.theta_clamp)

    if early_accept_linear_seed:
        R2 = lc_residual2d_dirichletx_periody(
            theta, I_mid,
            b=float(ctx.b), bi=float(ctx.bi),
            du=float(ctx.du), dv=float(ctx.dv),
            mobility=float(ctx.mobility),
        )
        stats0 = residual_stats_2d(R2)
        if (stats0["max_interior"] <= residual_tol_max) and (stats0["rms_interior"] <= residual_tol_rms):
            return theta, {
                "converged": True,
                "n_outer_passes": 0,
                "max_interior": stats0["max_interior"],
                "rms_interior": stats0["rms_interior"],
            }

    history_max = []
    history_rms = []
    converged = False

    for outer in range(max_outer_passes):
        theta = _static_relax_to_resid_zcoupled(
            theta, I_mid, tp, tn, ctx,
            sS=sS, offS=offS, diagS=diagS, lamS=lamS
        ).astype(cp.float32, copy=False)

        if ctx.theta_clamp is not None:
            theta = cp.clip(theta, cp.float32(ctx.theta_clamp[0]), cp.float32(ctx.theta_clamp[1]))
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        theta[0, :] = theta_bc
        theta[-1, :] = theta_bc

        R2 = lc_residual2d_dirichletx_periody(
            theta, I_mid,
            b=float(ctx.b), bi=float(ctx.bi),
            du=float(ctx.du), dv=float(ctx.dv),
            mobility=float(ctx.mobility),
        )
        stats = residual_stats_2d(R2)
        history_max.append(stats["max_interior"])
        history_rms.append(stats["rms_interior"])

        if verbose:
            print(f"[strict static] pass={outer+1:2d} max={stats['max_interior']:.3e} rms={stats['rms_interior']:.3e}")

        if (stats["max_interior"] <= residual_tol_max) and (stats["rms_interior"] <= residual_tol_rms):
            converged = True
            break

    return theta, {
        "converged": converged,
        "n_outer_passes": outer + 1,
        "max_interior": history_max[-1],
        "rms_interior": history_rms[-1],
    }

def run_quasiglobal_td_branch(
    ctx,
    *,
    dt_j,
    store_I_dtype,
    I_mid_store,
    save_slices_fn,
    t_stride,
    jt,
    jt_iter,
    report_runtime,
    tmp_dir="/tmp",
    use_memmap_pred=False,
    z_chunk=16,
):
    """
    Run one quasi-global TD step and update ctx.theta_full in place.

    Returns
    -------
    td_resid_last : float
    td_resid_max  : float
    """
    theta_new, I_mid_store_new, amp_out = td_step_quasiglobal_fast(
        ctx,
        ctx.amp_time_state,
        dt_j,
        I_store_dtype=store_I_dtype,
        tmp_dir=tmp_dir,
        use_memmap_pred=use_memmap_pred,
        z_chunk=z_chunk,
    )

    ctx.theta_full[...] = theta_new
    ctx.amp_time_state[...] = amp_out

    if I_mid_store is not None:
        I_mid_store[...] = I_mid_store_new.astype(store_I_dtype, copy=False)

    if save_slices_fn is not None and ((jt % t_stride) == 0):
        slot = jt // t_stride
        for k in range(ctx.Nz):
            save_slices_fn(slot, k, I_mid_store_new[k], ctx.theta_full[k])

    td_resid_last = 0.0
    td_resid_max = 0.0

    if report_runtime:
        gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))

        for k in range(ctx.Nz):
            tp = ctx.theta_full[k - 1] if k > 0 else ctx.theta_full[k]
            tn = ctx.theta_full[k + 1] if (k + 1) < ctx.Nz else ctx.theta_full[k]

            rloc = _slice_residual_rms_local3d_z(
                ctx.theta_full[k],
                I_mid_store_new[k].astype(cp.float32, copy=False),
                tp,
                tn,
                ctx,
                gamma_z,
            )
            td_resid_last = rloc
            td_resid_max = max(td_resid_max, rloc)

        try:
            jt_iter.set_postfix(
                rlast=f"{td_resid_last:.2e}",
                rmax=f"{td_resid_max:.2e}",
            )
        except Exception:
            pass

    del theta_new, I_mid_store_new, amp_out


    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return td_resid_last, td_resid_max

def run_unified_td_static(
    ctx, *,
    timedep,
    time_steps=None,
    tsteps=None,
    t_stride=1,
    restart_theta=True,
    restart_static_from_bias=True,
    store_I_dtype=cp.float16,
    save_slices_fn=None,
    tqdm_timedep=True,
    tqdm_static_z=True,
    report_runtime=True,
    true_td=True,
    static_mode="strict_relax",      # "linear", "strict_relax"
    use_linear_seed_for_static_relax=True,
    early_accept_linear_seed=True,
    strict_residual_tol_max=5e-2,
    strict_residual_tol_rms=1e-2,
    strict_max_outer_passes=8,
    strict_verbose_every=100,
    frozen_I_mid_store=None,
    frozen_I_from_theta=False,
    use_quasiglobal_td=False,
    freeze_theta=False,
    checkpoint_dir=None,
    checkpoint_every=0,
    checkpoint_keep_history=False,
    checkpoint_prefix="lc",
    checkpoint_Ixz=None,
    checkpoint_Iyz=None,
    checkpoint_thetaxz=None,
    checkpoint_thetayz=None,
    checkpoint_theta_history_dir=None,
    checkpoint_theta_history_prefix="theta_hist",
    checkpoint_theta_history_dtype=np.float32,
):
    if static_mode not in ("linear", "strict_relax"):
        raise ValueError("static_mode must be 'linear', or 'strict_relax'")

    use_dual_grid = bool(getattr(ctx, "use_dual_grid", False))

    if use_dual_grid:
        ctx_t = ctx.ctx_theta
        dg = ctx.dual_grid
        print(
            f"[dual grid] optics=({ctx.Nx},{ctx.Ny}) "
            f"theta=({ctx_t.Nx},{ctx_t.Ny}) "
            f"rx,ry=({dg.rx},{dg.ry})"
        )
    else:
        ctx_t = ctx
        dg = None

    if timedep:
        if time_steps is None:
            raise ValueError("timedep=True requires time_steps")
        if tsteps is None:
            tsteps = len(time_steps)
    else:
        tsteps = 1

    t_model = 0.0

    if timedep and restart_theta:
        if use_dual_grid:
            ctx_t.theta_full[...] = ctx_t.theta_bias_2d[None, :, :]
        else:
            ctx.theta_full[...] = ctx.theta_bias_2d[None, :, :]
    elif (not timedep) and restart_static_from_bias:
        ctx.theta_full[...] = ctx.theta_bias_2d[None, :, :]

    if (ctx.theta_z_stride_um is None) or (float(ctx.theta_z_stride_um) <= 0.0):
        theta_k_stride = 1
    else:
        theta_k_stride = max(1, int(round(float(ctx.theta_z_stride_um) / float(ctx.dz))))


    Nsub, dz_sub, phi_est = choose_optics_substeps(
        ctx.dz,
        kout=ctx.kout,
        dn_max_est=ctx.dn_max_est,
        dz_opt_max_phi=ctx.dz_opt_max_phi,
        max_substeps=ctx.max_substeps
    )
    h_sub = get_h_for_dz(ctx, dz_sub)
    h_half = get_h_for_dz(ctx, 0.5 * dz_sub)
    ctx.Nsub = Nsub
    ctx.dz_sub = dz_sub
    quasi_report = True
    print(f"[theta stride] theta_z_stride_um={ctx.theta_z_stride_um} -> theta_k_stride={theta_k_stride} slices")
    print(f"[optics substep] dz={ctx.dz} um phi_est={phi_est:.3f} -> Nsub={Nsub} dz_sub={dz_sub} um")
    print(f"[theta z coupling] theta_z_gamma={float(getattr(ctx, 'theta_z_gamma', 0.0))}")
    if timedep:
        print(f"[TD mode] {'true physical-time integrator' if true_td else 'pseudo stepper'}")
    else:
        print(f"[static mode] {static_mode}")

    amp = cp.empty_like(ctx.amp0)
    Ahat = cp.empty_like(ctx.amp0)
    I_b = cp.empty((ctx.Nx, ctx.Ny), cp.float32)
    I_a = cp.empty((ctx.Nx, ctx.Ny), cp.float32)
    I_mid = cp.empty((ctx.Nx, ctx.Ny), cp.float32)

    need_full_I_store = (not timedep) or bool(getattr(ctx, "save_full_I_mid_store", False))
    I_mid_store = cp.empty((ctx.Nz, ctx.Nx, ctx.Ny), dtype=store_I_dtype) if need_full_I_store else None

    plan_f = spfft.get_fft_plan(Ahat, axes=(-2, -1))
    plan_i = spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C")

    sS, offS, diagS, lamS = prepare_cn_ky_operator(
        dt=float(ctx.dtau_static),
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        Ny=ctx.Ny
    )

    jt_iter = tqdm(range(tsteps), desc=("LC timedep" if timedep else "LC static"), dynamic_ncols=True) \
        if (tqdm_timedep and timedep) else range(tsteps)

    quasi_report = True

    if timedep and true_td and use_quasiglobal_td:
        if (not hasattr(ctx, "amp_time_state")) or (ctx.amp_time_state is None) or (ctx.amp_time_state.shape != ctx.amp0.shape):
            ctx.amp_time_state = ctx.amp0.copy()

    if not hasattr(ctx, "_amp_in_buf") or ctx._amp_in_buf.shape != amp.shape:
        ctx._amp_in_buf = cp.empty_like(amp)

    if not hasattr(ctx, "_amp_pred_buf") or ctx._amp_pred_buf.shape != amp.shape:
        ctx._amp_pred_buf = cp.empty_like(amp)

    if not hasattr(ctx, "_I_mid_pred_buf") or ctx._I_mid_pred_buf.shape != I_mid.shape:
        ctx._I_mid_pred_buf = cp.empty_like(I_mid)

    for jt in jt_iter:
        if timedep:
            dt_j = float(time_steps[jt])
            t_model += dt_j

            if true_td:
                a_ie, off, diag, lam_y = prepare_ie_ky_operator(
                    dt=dt_j,
                    mobility=float(ctx.mobility),
                    du=float(ctx.du),
                    dv=float(ctx.dv),
                    Ny=ctx.Ny
                )
                s = None
            else:
                s, off, diag, lam_y = prepare_cn_ky_operator(
                    dt=dt_j,
                    mobility=float(ctx.mobility),
                    du=float(ctx.du),
                    dv=float(ctx.dv),
                    Ny=ctx.Ny
                )
                a_ie = None

            if ctx.theta_prev_time is None or ctx.theta_prev_time.shape != ctx.theta_full.shape:
                ctx.theta_prev_time = cp.empty_like(ctx.theta_full)

            ctx.theta_prev_time[...] = ctx.theta_full
            theta_ref = ctx.theta_prev_time

            td_resid_last = 0.0
            td_resid_max = 0.0
        else:
            if ctx.theta_prev_time is None or ctx.theta_prev_time.shape != ctx.theta_full.shape:
                ctx.theta_prev_time = cp.empty_like(ctx.theta_full)

            ctx.theta_prev_time[...] = ctx.theta_full
            theta_ref = ctx.theta_prev_time
            dt_j = 0.0

        # =========================================================
        # QUASI-GLOBAL TD BRANCH
        # =========================================================
        if timedep and true_td and use_quasiglobal_td:
            if quasi_report:
                print("Quasi Global Branch")
                quasi_report = False

            td_resid_last, td_resid_max = run_quasiglobal_td_branch(
                ctx,
                dt_j=dt_j,
                store_I_dtype=store_I_dtype,
                I_mid_store=I_mid_store,
                save_slices_fn=save_slices_fn,
                t_stride=t_stride,
                jt=jt,
                jt_iter=jt_iter,
                report_runtime=report_runtime,
                tmp_dir="/tmp",
                use_memmap_pred=False,
                z_chunk=16,
            )

            continue
        # =====================================================
        linearized_td = bool(getattr(ctx, "linearized_td", False))
        amp[...] = ctx.amp0
        hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)

        k_iter = tqdm(range(ctx.Nz), desc="LC z", dynamic_ncols=True) \
            if (tqdm_static_z and (not timedep)) else range(ctx.Nz)

        if not timedep:
            theta_running = ctx.theta_bias_2d.copy() if restart_static_from_bias else ctx.theta_full[0].copy()
        else:
            theta_running = None

        for k in k_iter:

            do_theta_update = (theta_k_stride <= 1) or ((k % theta_k_stride) == 0)
            if freeze_theta:
                do_theta_update = False

            intens_into(I_b, amp, coh=ctx.coh)

            if timedep:
                theta_use = theta_ref[k]
                tp = theta_ref[k - 1] if k > 0 else theta_ref[k]
                tn = theta_ref[k + 1] if (k + 1) < ctx.Nz else theta_ref[k]
            else:
                if k > 0:
                    tp = 0.5 * (ctx.theta_full[k - 1] + theta_ref[k - 1])
                else:
                    tp = theta_running if restart_static_from_bias else ctx.theta_full[k]

                tn = theta_ref[k + 1] if (k + 1) < ctx.Nz else (
                    theta_running if restart_static_from_bias else ctx.theta_full[k]
                )

            # ---------------- TD path ----------------
            if timedep:
                gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))

                if frozen_I_mid_store is not None:
                    amp_for_pred = amp
                    td_get_slice_mid_intensity(
                        ctx, amp, theta_use, I_b, I_a, I_mid,
                        Nsub=Nsub, dz_sub=dz_sub, h_sub=h_sub,
                        Ahat=Ahat, plan_f=plan_f, plan_i=plan_i,
                        frozen_I_mid_k=frozen_I_mid_store[k],
                    )
                else:
                    do_full_pred = bool(getattr(ctx, "true_td_full_pred_optics", False))
                    amp_for_pred = None
                    if do_full_pred:
                        amp_for_pred = ctx._amp_in_buf
                        amp_for_pred[...] = amp

                    td_get_slice_mid_intensity(
                        ctx, amp, theta_use, I_b, I_a, I_mid,
                        Nsub=Nsub, dz_sub=dz_sub, h_sub=h_sub,
                        Ahat=Ahat, plan_f=plan_f, plan_i=plan_i,
                        frozen_I_mid_k=None,
                    )

                if I_mid_store is not None:
                    I_mid_store[k] = I_mid.astype(store_I_dtype, copy=False)

                if do_theta_update:

                    if linearized_td:
                        ctx.theta_full[k] = advance_theta_linearized_td_step(
                            ctx,
                            theta_ref[k],
                            I_mid,
                            tp,
                            tn,
                            dt_j=dt_j,
                            off=off,
                            diag=diag,
                        )
                    else:

                        ctx.theta_full[k] = td_update_theta_from_midintensity(
                            ctx,
                            theta_ref[k],
                            tp,
                            tn,
                            I_mid,
                            dt_j=dt_j,
                            true_td=true_td,
                            a_ie=a_ie,
                            s=s,
                            off=off,
                            diag=diag,
                            lam_y=lam_y,
                            Nsub=Nsub,
                            dz_sub=dz_sub,
                            h_sub=h_sub,
                            amp_for_pred=amp_for_pred,
                            I_b=I_b,
                            I_a=I_a,
                            Ahat=Ahat,
                            plan_f=plan_f,
                            plan_i=plan_i,
                        )
                else:
                    ctx.theta_full[k] = theta_ref[k]

                theta_save = ctx.theta_full[k]

                runtime_every_k = int(getattr(ctx, "runtime_every_k", 0))
                if report_runtime:
                    do_report_here = (
                        runtime_every_k <= 0
                        or (k % runtime_every_k) == 0
                        or (k == ctx.Nz - 1)
                    )
                    if do_report_here:
                        rloc = _slice_residual_rms_local3d_z(
                            ctx.theta_full[k], I_mid, tp, tn, ctx,
                            float(getattr(ctx, "theta_z_gamma", 0.0))
                        )
                        td_resid_last = rloc
                        td_resid_max = max(td_resid_max, rloc)

            # ---------------- static path ----------------
            else:
                I_mid[...] = I_b

                did_selfconsistent_strict = False
                if do_theta_update:


                    if static_mode == "linear":
                        out_lin = solve_theta_perturbation_dirichletx_periody(
                            theta_bias=ctx.theta_bias_2d,
                            Ixy=I_mid,
                            b=float(ctx.b),
                            bi=float(ctx.bi),
                            du=float(ctx.du),
                            dv=float(ctx.dv),
                            xp=cp,
                            return_theta=True,
                            check_residual=False,
                            verbose=False,
                        )
                        theta_running = out_lin["theta"].astype(cp.float32, copy=False)

                    elif static_mode == "strict_relax":
                        theta_seed = ctx.theta_bias_2d if restart_static_from_bias else (
                            theta_running if (k > 0) else ctx.theta_full[k]
                        )

                        theta_running, I_mid_sc, amp_sc, info = strict_static_relax_slice_selfconsistent(
                            amp_in=amp,
                            theta_seed=theta_seed,
                            tp=tp,
                            tn=tn,
                            ctx=ctx,
                            dz_sub=dz_sub,
                            Nsub=Nsub,
                            h_sub=h_sub,
                            Ahat=Ahat,
                            plan_f=plan_f,
                            plan_i=plan_i,
                            sS=sS,
                            offS=offS,
                            diagS=diagS,
                            lamS=lamS,
                            use_linear_seed=use_linear_seed_for_static_relax,
                            early_accept_linear_seed=early_accept_linear_seed,
                            residual_tol_max=strict_residual_tol_max,
                            residual_tol_rms=strict_residual_tol_rms,
                            max_outer_passes=strict_max_outer_passes,
                            max_selfcons_passes=int(getattr(ctx, "strict_selfcons_passes", 3)),
                            selfcons_tol_theta=float(getattr(ctx, "strict_selfcons_tol_theta", 1e-4)),
                            selfcons_tol_I=float(getattr(ctx, "strict_selfcons_tol_I", 1e-4)),

                            verbose=False,
                        )

                        if True:
                          I_mid[...] = I_mid_sc
                          amp[...] = amp_sc
                          did_selfconsistent_strict = True

                        if False:
                            I_mid_candidate = I_mid_sc
                            amp[...] = amp_sc



                            print(
                                "[amp diag]",
                                "shape=", amp.shape,
                                "dtype=", amp.dtype,
                                "iscomplex=", cp.iscomplexobj(amp),
                            )

                            print(
                                "[amp diag]",
                                "abs max=", float(cp.max(cp.abs(amp))),
                                "abs min=", float(cp.min(cp.abs(amp))),
                            )

                            print(
                                "[amp diag]",
                                "power=", float(cp.sum(cp.abs(amp)**2) * ctx.dx * ctx.dy),
                            )

                            tmpI = cp.abs(amp)**2

                            print(
                                "[amp diag]",
                                "|amp|^2 max=", float(cp.max(tmpI)),
                                "|amp|^2 sum=", float(cp.sum(tmpI) * ctx.dx * ctx.dy),
                            )






                            # For launch diagnostics: strict_static_relax_slice_selfconsistent
                            # currently returns a bad I_mid_sc even when amp_sc is valid.
                            if (not bool(cp.all(cp.isfinite(I_mid_sc)))) or float(cp.max(I_mid_sc)) < 1e-12:
                                intens_into(I_mid, amp, coh=ctx.coh)
                                print(
                                    "[I_mid patch] replaced bad I_mid_sc:",
                                    "I_mid_sc max=", float(cp.max(I_mid_sc)),
                                    "amp power=", float(cp.sum(cp.abs(amp)**2) * ctx.dx * ctx.dy),
                                    "new Imax=", float(cp.max(I_mid)),
                                    "new Isum=", float(cp.sum(I_mid) * ctx.dx * ctx.dy),
                                )
                            else:
                                I_mid[...] = I_mid_sc

                            did_selfconsistent_strict = True

                        if (strict_verbose_every is not None) and ((k % strict_verbose_every) == 0 or k == ctx.Nz - 1):
                            print(
                                f"[k={k:4d}] max={info['max_interior']:.3e}  "
                                f"rms={info['rms_interior']:.3e}  "
                                f"sc_passes={info['n_selfcons_passes']}  "
                                f"ok={info['converged']}"
                            )

                    theta_running = enforce_theta_constraints(theta_running, ctx.theta_clamp)

                    ctx.theta_full[k] = theta_running
                    theta_save = theta_running
                    theta_use = theta_running
                else:
                    theta_current = theta_running if theta_running is not None else ctx.theta_full[k]

                    ctx.theta_full[k] = theta_current
                    theta_save = theta_current
                    theta_use = theta_current

                if not did_selfconsistent_strict:
                    for _ in range(Nsub):
                        apply_nonlinear_phase_inplace(
                            amp, theta_use,
                            dz_step_um=dz_sub,
                            kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
                        )
                        hop_linear_inplace(amp, h_sub, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)

                    intens_into(I_a, amp, coh=ctx.coh)
                else:
                    intens_into(I_a, amp, coh=ctx.coh)

                if I_mid_store is not None:
                    I_mid_store[k] = I_mid.astype(store_I_dtype, copy=False)

            if (save_slices_fn is not None) and ((jt % t_stride) == 0):
                slot = jt // t_stride
                save_slices_fn(slot, k, I_a, theta_save)

        hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)

        if timedep and report_runtime and hasattr(jt_iter, "set_postfix"):
            try:
                jt_iter.set_postfix(
                    rlast=f"{td_resid_last:.2e}",
                    rmax=f"{td_resid_max:.2e}",
                )
            except Exception:
                pass

        if timedep and checkpoint_dir and checkpoint_every:
            if (((jt + 1) % int(checkpoint_every)) == 0) or (jt == (tsteps - 1)):
                last_movie_slot = None
                if jt >= 0:
                    last_movie_slot = jt // t_stride

                save_lc_checkpoint(
                    checkpoint_dir,
                    prefix=checkpoint_prefix,
                    step_kind="jt",
                    step_index=jt,
                    t_model=t_model,
                    theta_full=ctx.theta_full,
                    Ixz=checkpoint_Ixz,
                    Iyz=checkpoint_Iyz,
                    thetaxz=checkpoint_thetaxz,
                    thetayz=checkpoint_thetayz,
                    keep_history=checkpoint_keep_history,
                    last_movie_slot=last_movie_slot,
                )
    archive_dir = None
    if checkpoint_dir is not None:
        archive_dir = finalize_checkpoint_archive(
            checkpoint_dir,
            prefix=checkpoint_prefix
        )

        # save prdata if available
        if hasattr(ctx, "prdata"):
            save_prdata_json(
                ctx.prdata,
                os.path.join(archive_dir, "prdata.json")
            )

    return I_mid_store

def advance_theta_linearized_timestep_prepared(
    theta_n,
    *,
    Ixy,
    theta_bias,
    dt,
    b,
    bi,
    mobility,
    du,
    dv,
    off,
    diag,
    clamp=None,
):
    """
    Linearized TD step around theta_bias for gamma_z = 0.

    Evolves delta = theta - theta_bias using
        delta_t = (1/mobility) * [ L_xy delta + q(I) delta + f(I) ]

    where
        q(I) = 2 (b + bi I) cos(2 theta_bias)
        f(I) = L_xy(theta_bias) + (b + bi I) sin(2 theta_bias)

    Returns full theta_{n+1} = theta_bias + delta_{n+1}.
    """
    theta_n = theta_n.astype(cp.float32, copy=False)
    Ixy = Ixy.astype(cp.float32, copy=False)
    theta_b = theta_bias.astype(cp.float32, copy=False)

    delta_n = (theta_n - theta_b).astype(cp.float32, copy=False)

    lap_b = _laplacian_dirichletx_periody(
        theta_b, float(du), float(dv)
    ).astype(cp.float32, copy=False)

    alpha = cp.float32(b) + cp.float32(bi) * Ixy
    q = cp.float32(2.0) * alpha * cp.cos(cp.float32(2.0) * theta_b)
    f = lap_b + alpha * cp.sin(cp.float32(2.0) * theta_b)

    dt_over_m = cp.float32(float(dt) / float(mobility))

    rhs = delta_n + dt_over_m * (q * delta_n + f)
    rhs[0, :] = 0.0
    rhs[-1, :] = 0.0

    delta_np1 = _cn_solve_dirichletx_periody(rhs, off=off, diag=diag)
    theta_np1 = theta_b + delta_np1
    theta_np1 = enforce_theta_constraints(theta_np1, clamp)

    return theta_np1

def advance_theta_linearized_timestep_prepared_zcoupled(
    theta_n,
    *,
    Ixy,
    theta_bias,
    theta_prev_n,
    theta_next_n,
    gamma_z,
    dt,
    b,
    bi,
    mobility,
    du,
    dv,
    off,
    diag,
    inv_dz2,
    clamp=None,
):
    """
    Linearized TD step around theta_bias with z coupling.

    Evolves delta = theta - theta_bias using
        delta_t = (1/mobility) * [ L_xy delta + gamma_z delta_zz + q(I) delta + f(I) ]

    Returns full theta_{n+1} = theta_bias + delta_{n+1}.
    """
    theta_n = theta_n.astype(cp.float32, copy=False)
    Ixy = Ixy.astype(cp.float32, copy=False)
    theta_b = theta_bias.astype(cp.float32, copy=False)
    tp = theta_prev_n.astype(cp.float32, copy=False)
    tn = theta_next_n.astype(cp.float32, copy=False)

    delta_n = (theta_n - theta_b).astype(cp.float32, copy=False)
    delta_p = (tp - theta_b).astype(cp.float32, copy=False)
    delta_q = (tn - theta_b).astype(cp.float32, copy=False)

    lap_b = _laplacian_dirichletx_periody(
        theta_b, float(du), float(dv)
    ).astype(cp.float32, copy=False)

    alpha = cp.float32(b) + cp.float32(bi) * Ixy
    q = cp.float32(2.0) * alpha * cp.cos(cp.float32(2.0) * theta_b)
    f = lap_b + alpha * cp.sin(cp.float32(2.0) * theta_b)

    gam = cp.float32(gamma_z) * inv_dz2
    dt_over_m = cp.float32(float(dt) / float(mobility))

    rhs = delta_n + dt_over_m * (q * delta_n + f + gam * (delta_p + delta_q))
    rhs[0, :] = 0.0
    rhs[-1, :] = 0.0

    diag_eff = diag + cp.float32(2.0) * dt_over_m * gam
    delta_np1 = _cn_solve_dirichletx_periody(rhs, off=off, diag=diag_eff)

    theta_np1 = theta_b + delta_np1
    theta_np1 = enforce_theta_constraints(theta_np1, clamp)

    return theta_np1

def advance_theta_linearized_td_step(
    ctx,
    theta_n,
    I_mid,
    tp,
    tn,
    *,
    dt_j,
    off,
    diag,
):
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))

    if gamma_z == 0.0:
        return advance_theta_linearized_timestep_prepared(
            theta_n,
            Ixy=I_mid,
            theta_bias=ctx.theta_bias_2d,
            dt=dt_j,
            b=ctx.b,
            bi=ctx.bi,
            mobility=ctx.mobility,
            du=ctx.du,
            dv=ctx.dv,
            off=off,
            diag=diag,
            clamp=ctx.theta_clamp,
        )

    return advance_theta_linearized_timestep_prepared_zcoupled(
        theta_n,
        Ixy=I_mid,
        theta_bias=ctx.theta_bias_2d,
        theta_prev_n=tp,
        theta_next_n=tn,
        gamma_z=gamma_z,
        dt=dt_j,
        b=ctx.b,
        bi=ctx.bi,
        mobility=ctx.mobility,
        du=ctx.du,
        dv=ctx.dv,
        off=off,
        diag=diag,
        inv_dz2=_get_inv_dz2(ctx),
        clamp=ctx.theta_clamp,
    )

