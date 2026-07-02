"""Z-stack time-dependent theta updates for LC soliton propagation.

Clean port of the trusted production TD slice update from runner_core.py.
This module owns one numerical task: advance one 2-D theta slice from a
midpoint optical intensity and frozen neighboring z-slices.

Important contract
------------------
The input ``Ixy``/``I_mid`` is **plain optical intensity**.  This module
multiplies by ``bi`` internally through ``(b + bi*I)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

Array = Any


def _xp_from(*arrays: Array, xp: Any | None = None):
    """Return an array module, preferring explicit ``xp`` when supplied."""
    if xp is not None:
        return xp
    for a in arrays:
        mod = type(a).__module__.split(".")[0]
        if mod == "cupy":
            import cupy as cp  # type: ignore
            return cp
    return np


def _as_float_dtype(xp: Any, dtype: Any | None):
    """Return a floating dtype object/type appropriate for xp."""
    if dtype is not None:
        return dtype
    return xp.float32 if hasattr(xp, "float32") else np.float32


def _cast(value: Any, dtype: Any | None):
    """Cast scalar ``value`` to dtype without ever calling a dtype object."""
    if dtype is None:
        return value
    return np.dtype(dtype).type(value)


def _real_scalar(x: Any) -> float:
    """Convert numpy/cupy scalar to Python float without assuming backend."""
    try:
        if hasattr(x, "get"):
            return float(x.get())
    except Exception:
        pass
    return float(x)


def _theta_bc_value(theta: Array, ctx: Any | None = None, *, xp: Any | None = None):
    """Boundary value in theta's dtype, defaulting to theta[0,0]."""
    xp = _xp_from(theta, xp=xp)
    value = getattr(ctx, "theta_bc", None) if ctx is not None else None
    if value is None:
        value = theta[0, 0]
    return xp.asarray(value, dtype=theta.dtype)


def _lam_y_periodic_second_diff(Ny: int, dv: float, *, xp: Any = np, dtype: Any | None = None):
    """Eigenvalues of the periodic second-difference operator in y."""
    dtype = _as_float_dtype(xp, dtype)
    m = xp.arange(int(Ny), dtype=dtype)
    lam = -4.0 * xp.sin(xp.pi * m / int(Ny)) ** 2 / (float(dv) * float(dv))
    return lam.astype(dtype, copy=False)


def prepare_ie_ky_operator(*, dt: float, mobility: float, du: float, dv: float, Ny: int, xp: Any = np, dtype: Any | None = None):
    """Prepare the implicit-Euler x solve, periodic-y diagonalized."""
    dtype = _as_float_dtype(xp, dtype)
    a_ie = float(dt) / float(mobility)
    lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=xp, dtype=dtype)

    inv_du2 = 1.0 / (float(du) * float(du))
    off = _cast(-a_ie * inv_du2, dtype)
    diag = 1.0 + 2.0 * a_ie * inv_du2 - a_ie * lam_y
    diag = diag.astype(dtype, copy=False)
    return _cast(a_ie, dtype), off, diag, lam_y


def prepare_cn_ky_operator(*, dt: float, mobility: float, du: float, dv: float, Ny: int, xp: Any = np, dtype: Any | None = None):
    """Prepare the Crank-Nicolson x solve, periodic-y diagonalized."""
    dtype = _as_float_dtype(xp, dtype)
    s = float(dt) / (2.0 * float(mobility))
    lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=xp, dtype=dtype)

    inv_du2 = 1.0 / (float(du) * float(du))
    off = _cast(-s * inv_du2, dtype)
    diag = 1.0 + 2.0 * s * inv_du2 - s * lam_y
    diag = diag.astype(dtype, copy=False)
    return _cast(s, dtype), off, diag, lam_y


def enforce_theta_constraints(theta: Array, theta_clamp: Optional[Tuple[float, float]] = None, *, xp: Any | None = None):
    """Apply optional theta clamp. Boundary conditions are handled separately."""
    xp = _xp_from(theta, xp=xp)
    if theta_clamp is None:
        return theta
    lo, hi = theta_clamp
    return xp.clip(theta, lo, hi)


def _laplacian_dirichletx_periody(theta: Array, du: float, dv: float, *, xp: Any | None = None):
    """2-D Laplacian with Dirichlet rows excluded and periodic y."""
    xp = _xp_from(theta, xp=xp)
    th = theta
    lap = xp.zeros_like(th)
    du2 = float(du) * float(du)
    dv2 = float(dv) * float(dv)
    yp = xp.roll(th, -1, axis=1)
    ym = xp.roll(th, +1, axis=1)
    lap[1:-1, :] = (
        (th[2:, :] - 2.0 * th[1:-1, :] + th[:-2, :]) / du2
        + (yp[1:-1, :] - 2.0 * th[1:-1, :] + ym[1:-1, :]) / dv2
    )
    return lap


def thomas_batched_const_tridiag(off_l: Any, diag: Array, off_u: Any, d: Array, *, xp: Any | None = None):
    """Solve many constant-tridiagonal systems sharing scalar off-diagonals."""
    xp = _xp_from(d, diag, xp=xp)
    batch, n = d.shape
    if n <= 0:
        return xp.empty_like(d)

    dtype = d.dtype
    cpv = xp.empty((batch, n), dtype=dtype)
    dpv = xp.empty((batch, n), dtype=dtype)

    b0 = diag.astype(dtype, copy=False)
    a = xp.asarray(off_l, dtype=dtype)
    c = xp.asarray(off_u, dtype=dtype)

    denom = b0
    cpv[:, 0] = c / denom
    dpv[:, 0] = d[:, 0] / denom

    for j in range(1, n):
        denom = b0 - a * cpv[:, j - 1]
        cpv[:, j] = c / denom if j < n - 1 else xp.zeros((batch,), dtype=dtype)
        dpv[:, j] = (d[:, j] - a * dpv[:, j - 1]) / denom

    x = xp.empty_like(d)
    x[:, -1] = dpv[:, -1]
    for j in range(n - 2, -1, -1):
        x[:, j] = dpv[:, j] - cpv[:, j] * x[:, j + 1]
    return x


def cn_solve_dirichletx_periody(rhs: Array, *, off: Any, diag: Array, theta_bc: float | None = None, xp: Any | None = None):
    """Solve a CN/IE Helmholtz-like system with Dirichlet x and periodic y."""
    xp = _xp_from(rhs, diag, xp=xp)
    rhs2 = rhs.astype(rhs.dtype, copy=True)
    theta_bc_arr = xp.asarray(rhs2[0, 0] if theta_bc is None else theta_bc, dtype=rhs2.dtype)

    rhs2[0, :] = theta_bc_arr
    rhs2[-1, :] = theta_bc_arr

    fft_dtype = xp.complex64 if rhs2.dtype == xp.float32 else xp.complex128
    rhs_hat = xp.fft.fft(rhs2, axis=1).astype(fft_dtype, copy=False)
    d_hatB = rhs_hat[1:-1, :].T.copy()  # (Ny, Nx-2)

    Ny_local = rhs.shape[1]
    bc_hat = xp.fft.fft(xp.full((Ny_local,), theta_bc_arr, dtype=rhs2.dtype)).astype(rhs_hat.dtype, copy=False)
    d_hatB[:, 0] -= off * bc_hat
    d_hatB[:, -1] -= off * bc_hat

    x_hatB = thomas_batched_const_tridiag(off, diag, off, d_hatB, xp=xp)

    theta_hat = xp.zeros_like(rhs_hat)
    theta_hat[1:-1, :] = x_hatB.T
    th = xp.fft.ifft(theta_hat, axis=1).real.astype(rhs2.dtype, copy=False)
    th[0, :] = theta_bc_arr
    th[-1, :] = theta_bc_arr
    return th


def advance_theta_timestep_cn_fft_thomas_prepared(
    theta_n: Array,
    *,
    dt: float,
    b: float,
    bi: float,
    Ixy: Array,
    mobility: float,
    du: float,
    dv: float,
    s: Any,
    off: Any,
    diag: Array,
    lam_y: Array | None = None,
    clamp: Optional[Tuple[float, float]] = None,
    xp: Any | None = None,
):
    """One predictor CN step for 2-D theta, matching runner_core."""
    xp = _xp_from(theta_n, Ixy, diag, xp=xp)
    dtype = theta_n.dtype
    theta_n = theta_n.astype(dtype, copy=False)
    Ixy = Ixy.astype(dtype, copy=False)

    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv), xp=xp).astype(dtype, copy=False)
    alpha = _cast(b, dtype) + _cast(bi, dtype) * Ixy
    drive_n = alpha * xp.sin(2.0 * theta_n)
    rhs = theta_n + s * lap_n + _cast(float(dt) / float(mobility), dtype) * drive_n

    theta_bc = theta_n[0, 0]
    rhs[0, :] = theta_bc
    rhs[-1, :] = theta_bc
    out = cn_solve_dirichletx_periody(rhs, off=off, diag=diag, theta_bc=_real_scalar(theta_bc), xp=xp)
    return enforce_theta_constraints(out, clamp, xp=xp)


def advance_theta_timestep_cn_trap_picard_prepared(
    theta_n: Array,
    *,
    dt: float,
    b: float,
    bi: float,
    I_n: Array,
    I_pic: Array,
    mobility: float,
    du: float,
    dv: float,
    s: Any,
    off: Any,
    diag: Array,
    lam_y: Array | None = None,
    max_iter: int = 4,
    tol_update: float = 1e-6,
    clamp: Optional[Tuple[float, float]] = None,
    xp: Any | None = None,
):
    """Trapezoid/Picard CN theta update used by the production TD runner."""
    xp = _xp_from(theta_n, I_n, I_pic, diag, xp=xp)
    dtype = theta_n.dtype
    theta_n = theta_n.astype(dtype, copy=False)
    I_n = I_n.astype(dtype, copy=False)
    I_pic = I_pic.astype(dtype, copy=False)

    alpha_old = _cast(b, dtype) + _cast(bi, dtype) * I_n
    alpha_pic = _cast(b, dtype) + _cast(bi, dtype) * I_pic
    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv), xp=xp).astype(dtype, copy=False)

    half_dt_over_m = _cast(0.5 * float(dt) / float(mobility), dtype)
    N_n = alpha_old * xp.sin(2.0 * theta_n)
    rhs_base = theta_n + s * lap_n + half_dt_over_m * N_n
    theta_bc = theta_n[0, 0]
    rhs_base[0, :] = theta_bc
    rhs_base[-1, :] = theta_bc

    theta_g = advance_theta_timestep_cn_fft_thomas_prepared(
        theta_n,
        dt=dt,
        b=b,
        bi=bi,
        Ixy=I_n,
        mobility=mobility,
        du=du,
        dv=dv,
        s=s,
        off=off,
        diag=diag,
        lam_y=lam_y,
        clamp=clamp,
        xp=xp,
    )

    for _ in range(int(max_iter)):
        N_g = alpha_pic * xp.sin(2.0 * theta_g)
        rhs = rhs_base + half_dt_over_m * N_g
        rhs[0, :] = theta_bc
        rhs[-1, :] = theta_bc
        theta_new = cn_solve_dirichletx_periody(rhs, off=off, diag=diag, theta_bc=_real_scalar(theta_bc), xp=xp)
        theta_new = enforce_theta_constraints(theta_new, clamp, xp=xp)

        dth = theta_new - theta_g
        rms_up = _real_scalar(xp.sqrt(xp.mean(dth * dth)))
        theta_g = theta_new
        if rms_up < float(tol_update):
            break

    return theta_g


def advance_theta_physical_timestep_semiimplicit_zcoupled(
    theta_n: Array,
    *,
    dt: float,
    b: float,
    bi: float,
    Ixy: Array,
    theta_prev_n: Array,
    theta_next_n: Array,
    gamma_z: float,
    mobility: float,
    du: float,
    dv: float,
    a_ie: Any,
    off: Any,
    diag: Array,
    lam_y: Array | None = None,
    inv_dz2: Any = 1.0,
    clamp: Optional[Tuple[float, float]] = None,
    xp: Any | None = None,
):
    """Semi-implicit physical-time update with z-neighbor coupling."""
    xp = _xp_from(theta_n, Ixy, theta_prev_n, theta_next_n, diag, xp=xp)
    dtype = theta_n.dtype
    theta_n = theta_n.astype(dtype, copy=False)
    Ixy = Ixy.astype(dtype, copy=False)
    tp = theta_prev_n.astype(dtype, copy=False)
    tn = theta_next_n.astype(dtype, copy=False)

    alpha = _cast(b, dtype) + _cast(bi, dtype) * Ixy
    drive_n = alpha * xp.sin(2.0 * theta_n)

    gam = _cast(gamma_z, dtype) * inv_dz2
    dt_over_m = _cast(float(dt) / float(mobility), dtype)

    rhs = theta_n + dt_over_m * (drive_n + gam * (tp + tn))
    theta_bc = theta_n[0, 0]
    rhs[0, :] = theta_bc
    rhs[-1, :] = theta_bc
    diag_eff = diag + 2.0 * dt_over_m * gam
    out = cn_solve_dirichletx_periody(rhs, off=off, diag=diag_eff, theta_bc=_real_scalar(theta_bc), xp=xp)
    return enforce_theta_constraints(out, clamp, xp=xp)


def advance_theta_timestep_cn_fft_thomas_prepared_zcoupled(
    theta_n: Array,
    *,
    dt: float,
    b: float,
    bi: float,
    Ixy: Array,
    theta_prev: Array,
    theta_next: Array,
    gamma_z: float,
    mobility: float,
    du: float,
    dv: float,
    s: Any,
    off: Any,
    diag: Array,
    lam_y: Array | None = None,
    inv_dz2: Any = 1.0,
    xp: Any | None = None,
):
    """Predictor CN update with explicit z-neighbor contribution."""
    xp = _xp_from(theta_n, Ixy, theta_prev, theta_next, diag, xp=xp)
    dtype = theta_n.dtype
    theta_n = theta_n.astype(dtype, copy=False)
    Ixy = Ixy.astype(dtype, copy=False)
    tp = theta_prev.astype(dtype, copy=False)
    tn = theta_next.astype(dtype, copy=False)

    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv), xp=xp).astype(dtype, copy=False)
    alpha = _cast(b, dtype) + _cast(bi, dtype) * Ixy
    drive_n = alpha * xp.sin(2.0 * theta_n)
    gam = _cast(gamma_z, dtype) * inv_dz2
    dt_over_m = _cast(float(dt) / float(mobility), dtype)

    rhs = theta_n + s * lap_n + dt_over_m * (drive_n + gam * (tp + tn))
    theta_bc = theta_n[0, 0]
    rhs[0, :] = theta_bc
    rhs[-1, :] = theta_bc
    diag_eff = diag + 2.0 * dt_over_m * gam
    return cn_solve_dirichletx_periody(rhs, off=off, diag=diag_eff, theta_bc=_real_scalar(theta_bc), xp=xp)


def advance_theta_timestep_cn_trap_picard_prepared_zcoupled(
    theta_n: Array,
    *,
    dt: float,
    b: float,
    bi: float,
    Ixy: Array,
    theta_prev: Array,
    theta_next: Array,
    gamma_z: float,
    mobility: float,
    du: float,
    dv: float,
    s: Any,
    off: Any,
    diag: Array,
    lam_y: Array | None = None,
    max_iter: int = 4,
    tol_update: float = 1e-6,
    clamp: Optional[Tuple[float, float]] = None,
    inv_dz2: Any = 1.0,
    xp: Any | None = None,
):
    """Trapezoid/Picard CN update with explicit z-neighbor coupling."""
    xp = _xp_from(theta_n, Ixy, theta_prev, theta_next, diag, xp=xp)
    dtype = theta_n.dtype
    theta_n = theta_n.astype(dtype, copy=False)
    Ixy = Ixy.astype(dtype, copy=False)
    tp = theta_prev.astype(dtype, copy=False)
    tn = theta_next.astype(dtype, copy=False)

    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv), xp=xp).astype(dtype, copy=False)
    alpha = _cast(b, dtype) + _cast(bi, dtype) * Ixy
    N_n = alpha * xp.sin(2.0 * theta_n)
    gam = _cast(gamma_z, dtype) * inv_dz2
    dt_over_m = _cast(float(dt) / float(mobility), dtype)

    rhs_base = theta_n + s * lap_n + 0.5 * dt_over_m * N_n + dt_over_m * gam * (tp + tn)
    theta_bc = theta_n[0, 0]
    rhs_base[0, :] = theta_bc
    rhs_base[-1, :] = theta_bc
    diag_eff = diag + 2.0 * dt_over_m * gam

    theta_g = advance_theta_timestep_cn_fft_thomas_prepared_zcoupled(
        theta_n,
        dt=dt,
        b=b,
        bi=bi,
        Ixy=Ixy,
        theta_prev=tp,
        theta_next=tn,
        gamma_z=gamma_z,
        mobility=mobility,
        du=du,
        dv=dv,
        s=s,
        off=off,
        diag=diag,
        lam_y=lam_y,
        inv_dz2=inv_dz2,
        xp=xp,
    )

    for _ in range(int(max_iter)):
        N_g = alpha * xp.sin(2.0 * theta_g)
        rhs = rhs_base + 0.5 * dt_over_m * N_g
        rhs[0, :] = theta_bc
        rhs[-1, :] = theta_bc
        theta_new = cn_solve_dirichletx_periody(rhs, off=off, diag=diag_eff, theta_bc=_real_scalar(theta_bc), xp=xp)
        theta_new = enforce_theta_constraints(theta_new, clamp, xp=xp)
        dth = theta_new - theta_g
        rms_up = _real_scalar(xp.sqrt(xp.mean(dth * dth)))
        theta_g = theta_new
        if rms_up < float(tol_update):
            break
    return theta_g


def _get_inv_dz2(ctx: Any, *, xp: Any = np, dtype: Any | None = None):
    dz = float(getattr(ctx, "dz"))
    return _cast(1.0 / (dz * dz), _as_float_dtype(xp, dtype))


def relax_theta_to_current_I_td_zcoupled(
    theta_init: Array,
    Ixy: Array,
    theta_prev: Array,
    theta_next: Array,
    ctx: Any,
    *,
    dt: float,
    s: Any,
    off: Any,
    diag: Array,
    lam_y: Array | None = None,
    xp: Any | None = None,
):
    """Quasi-static relaxation branch from the trusted runner."""
    xp = _xp_from(theta_init, Ixy, theta_prev, theta_next, diag, xp=xp)
    dtype = theta_init.dtype
    th = theta_init.astype(dtype, copy=True)
    I32 = Ixy.astype(dtype, copy=False)

    nrelax = max(1, int(getattr(ctx, "td_theta_relax_steps", 1)))
    omega = float(getattr(ctx, "td_theta_relax_omega", 1.0))
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))
    inv_dz2 = _get_inv_dz2(ctx, xp=xp, dtype=dtype)

    for _ in range(nrelax):
        th_new = advance_theta_timestep_cn_trap_picard_prepared_zcoupled(
            th,
            dt=float(dt),
            b=float(ctx.b),
            bi=float(ctx.bi),
            Ixy=I32,
            theta_prev=theta_prev,
            theta_next=theta_next,
            gamma_z=gamma_z,
            mobility=float(ctx.mobility),
            du=float(ctx.du),
            dv=float(ctx.dv),
            s=s,
            off=off,
            diag=diag,
            lam_y=lam_y,
            max_iter=int(getattr(ctx, "picard_iters", 4)),
            tol_update=float(getattr(ctx, "picard_tol_up", 1e-6)),
            clamp=getattr(ctx, "theta_clamp", None),
            inv_dz2=inv_dz2,
            xp=xp,
        )
        th = (1.0 - omega) * th + omega * th_new
        theta_bc = xp.asarray(getattr(ctx, "theta_bc", 0.0), dtype=dtype)
        th[0, :] = theta_bc
        th[-1, :] = theta_bc
        th = enforce_theta_constraints(th, getattr(ctx, "theta_clamp", None), xp=xp)

    return th


@dataclass(frozen=True)
class TDThetaSliceInfo:
    """Minimal diagnostics for a z-stack theta slice update."""

    true_td: bool
    gamma_z: float
    predictor_only: bool
    full_pred_optics: bool


def td_update_theta_from_midintensity(
    ctx: Any,
    theta_ref_k: Array,
    tp: Array,
    tn: Array,
    I_mid: Array,
    *,
    dt_j: float,
    true_td: bool,
    a_ie: Any,
    s: Any,
    off: Any,
    diag: Array,
    lam_y: Array | None,
    Nsub: int = 1,
    dz_sub: float | None = None,
    h_sub: Array | None = None,
    amp_for_pred: Array | None = None,
    I_b: Array | None = None,
    I_a: Array | None = None,
    Ahat: Array | None = None,
    plan_f: Any | None = None,
    plan_i: Any | None = None,
    xp: Any | None = None,
):
    """Trusted production theta update for one z-slice."""
    xp = _xp_from(theta_ref_k, tp, tn, I_mid, diag, xp=xp)
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))
    predictor_only = bool(getattr(ctx, "true_td_predictor_only", False))
    do_full_pred = bool(getattr(ctx, "true_td_full_pred_optics", False))

    if not bool(true_td):
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
            xp=xp,
        )

    inv_dz2 = _get_inv_dz2(ctx, xp=xp, dtype=theta_ref_k.dtype)

    if gamma_z != 0.0:
        theta_new = advance_theta_physical_timestep_semiimplicit_zcoupled(
            theta_ref_k,
            dt=dt_j,
            b=float(ctx.b),
            bi=float(ctx.bi),
            Ixy=I_mid,
            theta_prev_n=tp,
            theta_next_n=tn,
            gamma_z=gamma_z,
            mobility=float(ctx.mobility),
            du=float(ctx.du),
            dv=float(ctx.dv),
            a_ie=a_ie,
            off=off,
            diag=diag,
            lam_y=lam_y,
            inv_dz2=inv_dz2,
            clamp=getattr(ctx, "theta_clamp", None),
            xp=xp,
        )
        theta_bc = xp.asarray(getattr(ctx, "theta_bc", 0.0), dtype=theta_ref_k.dtype)
        theta_new[0, :] = theta_bc
        theta_new[-1, :] = theta_bc
        return theta_new

    theta_pred_k = advance_theta_physical_timestep_semiimplicit_zcoupled(
        theta_ref_k,
        dt=dt_j,
        b=float(ctx.b),
        bi=float(ctx.bi),
        Ixy=I_mid,
        theta_prev_n=tp,
        theta_next_n=tn,
        gamma_z=0.0,
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        a_ie=a_ie,
        off=off,
        diag=diag,
        lam_y=lam_y,
        inv_dz2=inv_dz2,
        clamp=getattr(ctx, "theta_clamp", None),
        xp=xp,
    )

    if predictor_only:
        theta_bc = xp.asarray(getattr(ctx, "theta_bc", 0.0), dtype=theta_ref_k.dtype)
        theta_pred_k[0, :] = theta_bc
        theta_pred_k[-1, :] = theta_bc
        return theta_pred_k

    if do_full_pred:
        raise NotImplementedError(
            "true_td_full_pred_optics uses optics prediction buffers and should be "
            "ported via optics_engine.splitstep before enabling."
        )

    s_cn, off_cn, diag_cn, lam_y_cn = prepare_cn_ky_operator(
        dt=dt_j,
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        Ny=int(ctx.Ny),
        xp=xp,
        dtype=theta_ref_k.dtype,
    )

    theta_new = advance_theta_timestep_cn_trap_picard_prepared(
        theta_ref_k,
        dt=dt_j,
        b=float(ctx.b),
        bi=float(ctx.bi),
        I_n=I_mid,
        I_pic=I_mid,
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        s=s_cn,
        off=off_cn,
        diag=diag_cn,
        lam_y=lam_y_cn,
        max_iter=int(getattr(ctx, "picard_iters", 4)),
        tol_update=float(getattr(ctx, "picard_tol_up", 1e-6)),
        clamp=getattr(ctx, "theta_clamp", None),
        xp=xp,
    )

    theta_bc = xp.asarray(getattr(ctx, "theta_bc", 0.0), dtype=theta_ref_k.dtype)
    theta_new[0, :] = theta_bc
    theta_new[-1, :] = theta_bc
    return theta_new


def advance_theta_td_slice(
    theta_k: Array,
    I_mid: Array,
    theta_prev_z: Array,
    theta_next_z: Array,
    ctx: Any,
    *,
    dt: float,
    true_td: bool = True,
    prepared: tuple[Any, Any, Any, Any] | None = None,
    xp: Any | None = None,
):
    """Small public API for production z-stack workflows."""
    xp = _xp_from(theta_k, I_mid, theta_prev_z, theta_next_z, xp=xp)
    if prepared is None:
        a_ie, off, diag, lam_y = prepare_ie_ky_operator(
            dt=dt,
            mobility=float(ctx.mobility),
            du=float(ctx.du),
            dv=float(ctx.dv),
            Ny=int(ctx.Ny),
            xp=xp,
            dtype=theta_k.dtype,
        )
    else:
        a_ie, off, diag, lam_y = prepared

    s = a_ie  # old runner passes ``s`` here for the quasi-static branch
    return td_update_theta_from_midintensity(
        ctx,
        theta_k,
        theta_prev_z,
        theta_next_z,
        I_mid,
        dt_j=dt,
        true_td=true_td,
        a_ie=a_ie,
        s=s,
        off=off,
        diag=diag,
        lam_y=lam_y,
        xp=xp,
    )


__all__ = [
    "prepare_ie_ky_operator",
    "prepare_cn_ky_operator",
    "cn_solve_dirichletx_periody",
    "advance_theta_timestep_cn_fft_thomas_prepared",
    "advance_theta_timestep_cn_trap_picard_prepared",
    "advance_theta_physical_timestep_semiimplicit_zcoupled",
    "advance_theta_timestep_cn_fft_thomas_prepared_zcoupled",
    "advance_theta_timestep_cn_trap_picard_prepared_zcoupled",
    "relax_theta_to_current_I_td_zcoupled",
    "td_update_theta_from_midintensity",
    "advance_theta_td_slice",
    "TDThetaSliceInfo",
]
