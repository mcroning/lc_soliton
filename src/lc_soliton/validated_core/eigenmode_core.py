"""Clean extracted LC soliton core code from the latest working notebook.

This module intentionally keeps the numerics close to the trusted notebook while
removing notebook UI, ad-hoc execution cells, and Streamlit/package wrappers.
"""
from __future__ import annotations

import gc
import json
import math
import os
import platform
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pandas as pd
import numpy as np
import cupy as cp
import cupyx.scipy.fft as spfft
from cupyx.scipy.ndimage import gaussian_filter

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

from .runner_core import *
from .launch_core import *

def xy_um_from_ctx(ctx):
    """
    Physical transverse coordinates in microns.
    Assumes ctx.dx, ctx.dy are in microns.
    """
    x_um = (cp.arange(ctx.Nx, dtype=cp.float32) - cp.float32((ctx.Nx - 1) / 2)) * cp.float32(ctx.dx)
    y_um = (cp.arange(ctx.Ny, dtype=cp.float32) - cp.float32((ctx.Ny - 1) / 2)) * cp.float32(ctx.dy)
    return x_um, y_um

def set_lc_power_from_prdata(ctx, prdata, P_mW):
    """
    Recompute ctx.bi from physical power.

    prdata['d'] is in microns.
    prdata['K'] is SI elastic constant, N.
    P_mW is milliwatts.
    """
    c0 = 299792458.0

    ne = float(ctx.ne)
    no = float(ctx.no)
    d_um = float(prdata["d"])
    K_SI = float(prdata["K"])
    P_W = 1e-3 * float(P_mW)

    na2 = ne**2 - no**2

    # Equivalent to:
    # bi = (na2 * d_m**2 * 1e12 * P_W) / (8 c0 K)
    # since d_m**2 * 1e12 = d_um**2.
    ctx.bi = float((na2 * d_um**2 * P_W) / (8.0 * c0 * K_SI))

def require_ctx_fields(ctx, names):
    missing = [name for name in names if not hasattr(ctx, name)]
    if missing:
        raise AttributeError(f"ctx is missing required fields: {missing}")


#####################################
#Save helpers
#####################################

def _json_safe(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, cp.ndarray):
        return cp.asnumpy(v).tolist()
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)

def ctx_repro_dict(ctx):
    """
    Minimal-but-useful reproduction metadata.
    Large arrays are saved separately in NPZ.
    """
    names = [
        "Nx", "Ny", "Nz",
        "dx", "dy", "dz",
        "lm", "k0",
        "ne", "no", "refin", "n0",
        "b", "bi", "theta_bc",
        "theta_clamp",
        "theta_z_gamma",
        "use_dual_grid",
        "T_mode",
        "Nsub", "dz_sub",
        "static_tol_resid",
        "static_max_steps",
        "static_relax_omega",
        "picard_iters",
        "picard_tol_up",
        "xsamp", "ysamp", "rlen",
        "windowedge",
        "theta_bc",
        "theta_z_stride_um",
        "DG_factor",
        "DG_theta_z_stride",
    ]

    d = {}
    for name in names:
        if hasattr(ctx, name):
            d[name] = _json_safe(getattr(ctx, name))

    if bool(getattr(ctx, "use_dual_grid", False)) and hasattr(ctx, "ctx_theta"):
        d["ctx_theta"] = {}
        for name in names:
            if hasattr(ctx.ctx_theta, name):
                d["ctx_theta"][name] = _json_safe(getattr(ctx.ctx_theta, name))

        if hasattr(ctx, "dual_grid"):
            dg = ctx.dual_grid
            d["dual_grid"] = {}
            for name in ["theta_z_stride", "factor", "restrict_mode", "prolong_mode"]:
                if hasattr(dg, name):
                    d["dual_grid"][name] = _json_safe(getattr(dg, name))

    return d

def rebuild_ctx_from_repro(repro):
    prdata = dict(repro["prdata"])
    c = repro["ctx"]

    prdata["Nx"] = int(c["Nx"])
    prdata["Ny"] = int(c["Ny"])
    prdata["Nz"] = int(c["Nz"])

    prdata["dx"] = float(c["dx"])
    prdata["dy"] = float(c["dy"])
    prdata["dz"] = float(c["dz"])

    prdata["xsamp"] = int(c["Nx"])
    prdata["ysamp"] = int(c["Ny"])
    prdata["zsteps"] = int(c["Nz"])

    out = initialize_lc_from_prdata(prdata)
    ctx = out["ctx"] if isinstance(out, dict) else out

    # Do NOT restore b/bi from repro.
    # They can be stale/wrong, especially after continuation/polish.
    skip = {
        "ctx_theta",
        "dual_grid",
        "b",
        "bi",
        "use_dual_grid",
    }

    for k, v in c.items():
        if k in skip:
            continue
        try:
            setattr(ctx, k, v)
        except Exception:
            pass

    ctx.use_dual_grid = False
    return ctx, prdata

def assert_ctx_matches_mode(ctx, mode):
    shape = tuple(mode["A"].shape)

    checks = {
        "theta_bias_2d": getattr(ctx, "theta_bias_2d", None),
        "fxy2": getattr(ctx, "fxy2", None),
    }

    for name, arr in checks.items():
        if arr is not None and tuple(arr.shape) != shape:
            raise ValueError(
                f"ctx mismatch: saved mode shape {shape}, "
                f"but ctx.{name}.shape={tuple(arr.shape)}"
            )

    if int(ctx.Nx) != shape[0] or int(ctx.Ny) != shape[1]:
        raise ValueError(
            f"ctx mismatch: saved mode shape {shape}, "
            f"but ctx.Nx,Ny={(ctx.Nx, ctx.Ny)}"
        )

def write_repro_json(save_dir, *, prdata, ctx, branch_name, powers_mW,
                     subtract_carrier, checkpoint_prefix, solve_kwargs):
    save_dir = Path(save_dir)

    repro = {
        "created_unix": time.time(),
        "branch_name": branch_name,
        "checkpoint_prefix": checkpoint_prefix,
        "powers_mW": _json_safe(np.asarray(powers_mW)),
        "subtract_carrier": bool(subtract_carrier),
        "prdata": _json_safe(prdata),
        "ctx": ctx_repro_dict(ctx),
        "solve_kwargs": _json_safe(solve_kwargs),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    path = save_dir / f"{checkpoint_prefix}_repro.json"
    with open(path, "w") as f:
        json.dump(repro, f, indent=2)

    print(f"[repro] wrote {path}")
    return path



# ============================================================
# Optical eigen-operator matching angular-spectrum propagation
# ============================================================

def make_beta_symbol_from_ctx(ctx, subtract_carrier=True):
    """
    Nonparaxial angular-spectrum eigenvalue q(fx,fy).

    Your linear hop is of the form:
        h = exp(i q dz)

    For the transverse envelope eigenproblem it is usually better to remove
    the constant carrier q0 = 2*pi*refin/lm, so beta measures the excess
    propagation constant relative to the reference plane wave.

    If subtract_carrier=True:
        beta_symbol = q(fx,fy) - q0

    If False:
        beta_symbol = q(fx,fy)
    """
    require_ctx_fields(ctx, ["lm", "refin", "fxy2"])

    lm = float(ctx.lm)       # microns
    refin = float(ctx.refin)

    q0 = 2.0 * np.pi * refin / lm  # 1/um

    arg = 1.0 - (lm / refin)**2 * ctx.fxy2
    mask = arg > 0.0

    q = cp.zeros_like(ctx.fxy2, dtype=cp.float32)
    q[mask] = (q0 * cp.sqrt(arg[mask])).astype(cp.float32)

    if subtract_carrier:
        q = q - cp.float32(q0)

    return q, mask

def optical_potential_from_theta(ctx, theta):
    """
    V(theta) = kout * dn(theta), matching the propagation phase:
        A <- exp(i * kout * dz * dn(theta)) A
    """
    require_ctx_fields(ctx, ["ne", "no", "refin", "kout"])

    dn = lc_dn_from_theta(theta, ctx.ne, ctx.no, ctx.refin)
    return cp.float32(ctx.kout) * dn

def apply_optical_eigen_operator(ctx, A, beta_symbol, theta):
    """
    H(theta) A = q_perp(-i∇) A + V(theta) A
    """
    Ahat = cp.fft.fft2(A)
    HA = cp.fft.ifft2(beta_symbol * Ahat)
    HA += optical_potential_from_theta(ctx, theta) * A
    return HA

def normalize_A_unit_intensity(A, dx, dy, eps=1e-30):
    """
    Normalize so ∫ |A|^2 dx dy = 1.
    dx, dy are physical microns.
    """
    norm = cp.sqrt(cp.sum(cp.abs(A)**2) * cp.float32(dx * dy) + eps)
    return A / norm

def rayleigh_beta(ctx, A, beta_symbol, theta):
    HA = apply_optical_eigen_operator(ctx, A, beta_symbol, theta)
    area = cp.float32(ctx.dx * ctx.dy)
    num = cp.vdot(A.ravel(), HA.ravel()) * area
    den = cp.vdot(A.ravel(), A.ravel()) * area
    return float(cp.real(num / den))

def phase_align(A, Aref, eps=1e-30):
    c = cp.vdot(Aref.ravel(), A.ravel())
    ph = c / (cp.abs(c) + eps)
    return A / ph

# Allow for higher order modes

def make_mode_seed(ctx, mode="00", w0_um=4.0):
    x_um, y_um = xy_um_from_ctx(ctx)
    X, Y = cp.meshgrid(cp.asarray(x_um), cp.asarray(y_um), indexing="ij")

    G = cp.exp(-(X**2 + Y**2) / (2.0 * w0_um**2))

    mode = str(mode).upper()

    if mode in ("10", "TEM10"):
        A0 = (X / w0_um) * G
    elif mode in ("01", "TEM01"):
        A0 = (Y / w0_um) * G
    elif mode in ("11", "TEM11"):
        A0 = (X * Y / w0_um**2) * G
    else:
        A0 = G

    A0 = A0.astype(cp.complex64, copy=False)
    A0 = normalize_A_unit_intensity(A0, ctx.dx, ctx.dy)

    theta0 = ctx.theta_bias_2d.copy().astype(cp.float32, copy=False)

    return A0, theta0

def project_mode_parity(A, theta, mode):
    mode = str(mode).upper()

    if mode in ("10", "TEM10"):
        A = 0.5 * (A - A[::-1, :])      # odd x
        theta = 0.5 * (theta + theta[::-1, :])
    elif mode in ("01", "TEM01"):
        A = 0.5 * (A - A[:, ::-1])      # odd y
        theta = 0.5 * (theta + theta[:, ::-1])
    elif mode in ("11", "TEM11"):
        A = 0.25 * (
            A - A[::-1, :] - A[:, ::-1] + A[::-1, ::-1]
        )
        theta = 0.25 * (
            theta + theta[::-1, :] + theta[:, ::-1] + theta[::-1, ::-1]
        )

    A = normalize_A_unit_intensity(A.astype(cp.complex64, copy=False), ctx.dx, ctx.dy)
    theta = enforce_theta_constraints(theta.astype(cp.float32, copy=False), ctx.theta_clamp)

    return A, theta


# ============================================================
# Fixed-theta optical eigenmode solve
# ============================================================

def solve_optical_mode_for_theta(
    ctx,
    A0,
    theta,
    beta_symbol,
    *,
    dtau=0.02,
    nsteps=300,
    tol=1e-8,
    shift_beta=True,
    verbose=False,
):
    """
    Fixed-theta optical eigenmode iteration.

    This is a simple normalized flow. If it drifts to the wrong branch,
    use a better seed or reduce dtau.
    """
    require_ctx_fields(ctx, ["dx", "dy"])

    A = normalize_A_unit_intensity(A0.astype(cp.complex64, copy=False), ctx.dx, ctx.dy)

    beta_old = None

    for it in range(int(nsteps)):
        A_old = A.copy()

        HA = apply_optical_eigen_operator(ctx, A, beta_symbol, theta)

        if shift_beta:
            beta_now = rayleigh_beta(ctx, A, beta_symbol, theta)
            HA = HA - cp.float32(beta_now) * A

        A = A + cp.float32(dtau) * HA
        A = normalize_A_unit_intensity(A, ctx.dx, ctx.dy)
        A = phase_align(A, A_old).astype(cp.complex64, copy=False)

        beta = rayleigh_beta(ctx, A, beta_symbol, theta)

        err = float(
            cp.linalg.norm((A - A_old).ravel())
            / (cp.linalg.norm(A_old.ravel()) + 1e-30)
        )

        if verbose and (it % 25 == 0):
            print(f"    [opt] it={it:4d} beta={beta:.9e} err={err:.3e}")

        if beta_old is not None:
            db = abs(beta - beta_old) / max(1.0, abs(beta))
            if err < tol and db < tol:
                break

        beta_old = beta

    return A, beta


# ============================================================
# Frozen-I theta solve using your strict settled form
# ============================================================

def solve_theta_for_eigenmode_I(
    ctx,
    theta_seed,
    Ixy,
    *,
    residual_tol_max=5e-2,
    residual_tol_rms=1e-2,
    max_outer_passes=8,
    use_linear_seed=False,
    verbose=False,
):
    """
    Uses strict_static_relax_slice as a 2D frozen-I director solver.

    For eigensolitons, theta_z_gamma is temporarily forced to zero.
    """
    require_ctx_fields(
        ctx,
        [
            "dtau_static", "mobility", "du", "dv", "Ny",
            "theta_z_gamma"
        ],
    )

    old_gamma = float(getattr(ctx, "theta_z_gamma", 0.0))
    ctx.theta_z_gamma = 0.0

    sS, offS, diagS, lamS = prepare_cn_ky_operator(
        dt=float(ctx.dtau_static),
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        Ny=int(ctx.Ny),
    )

    theta, info = strict_static_relax_slice(
        theta_seed,
        Ixy.astype(cp.float32, copy=False),
        theta_seed,
        theta_seed,
        ctx,
        sS=sS,
        offS=offS,
        diagS=diagS,
        lamS=lamS,
        use_linear_seed=use_linear_seed,
        early_accept_linear_seed=True,
        residual_tol_max=residual_tol_max,
        residual_tol_rms=residual_tol_rms,
        max_outer_passes=max_outer_passes,
        verbose=verbose,
    )

    ctx.theta_z_gamma = old_gamma
    return theta, info


# ============================================================
# Coupled nonlinear optical eigensoliton solve
# ============================================================

def solve_lc_optical_eigensoliton_v2(
    ctx,
    A_seed,
    theta_seed,
    *,
    beta_symbol=None,
    max_outer=20,
    mix_theta=0.10,
    optical_dtau=0.02,
    optical_steps=150,
    optical_tol=1e-7,
    theta_residual_tol_max=2e-1,
    theta_residual_tol_rms=3e-2,
    theta_outer_passes=4,
    outer_tol_A=1e-5,
    outer_tol_theta=1e-5,
    outer_print_every=5,
    verbose=True,
    progress_callback=None,
):
    if beta_symbol is None:
        beta_symbol, _ = make_beta_symbol_from_ctx(ctx, subtract_carrier=True)

    A = normalize_A_unit_intensity(A_seed.astype(cp.complex64, copy=True), ctx.dx, ctx.dy)
    theta = enforce_theta_constraints(theta_seed.astype(cp.float32, copy=True), ctx.theta_clamp)

    beta = np.nan
    beta_old = None
    eA = np.inf
    eth = np.inf
    eb = np.inf
    theta_info = {"max_interior": np.nan, "rms_interior": np.nan}

    for outer in range(int(max_outer)):
        A_old = A.copy()
        theta_old = theta.copy()

        A, beta = solve_optical_mode_for_theta(
            ctx,
            A,
            theta,
            beta_symbol,
            dtau=float(optical_dtau),
            nsteps=int(optical_steps),
            tol=float(optical_tol),
            verbose=False,
        )

        if getattr(ctx, "T_mode", "00").upper() not in ("00", "TEM00"):
            A, theta = project_mode_parity(A, theta, ctx.T_mode)

        I = cp.abs(A) ** 2
        normalize_Ishape_inplace(I, ctx.dx, ctx.dy)

        theta_new, theta_info = solve_theta_for_eigenmode_I(
            ctx,
            theta,
            I,
            residual_tol_max=float(theta_residual_tol_max),
            residual_tol_rms=float(theta_residual_tol_rms),
            max_outer_passes=int(theta_outer_passes),
            use_linear_seed=False,
            verbose=False,
        )

        #theta = ((1.0 - mix_theta) * theta + mix_theta * theta_new).astype(cp.float32, copy=False)

        Rmax_new = float(theta_info.get("max_interior", np.inf))
        Rrms_new = float(theta_info.get("rms_interior", np.inf))

        bad_theta_update = (
            not np.isfinite(Rmax_new)
            or not np.isfinite(Rrms_new)
            or Rmax_new > 10.0
            or Rrms_new > 0.2
        )

        mix_use = float(mix_theta)

        if bad_theta_update:
            mix_use = 0.0
            if verbose:
                print(
                    f"[eig safeguard outer={outer:3d}] rejecting theta update: "
                    f"Rmax={Rmax_new:.3e} Rrms={Rrms_new:.3e}"
                )

        theta = ((1.0 - mix_use) * theta + mix_use * theta_new).astype(cp.float32, copy=False)



        
        theta = enforce_theta_constraints(theta, ctx.theta_clamp)

        eA = float(cp.linalg.norm((A - A_old).ravel()) / (cp.linalg.norm(A_old.ravel()) + 1e-30))
        eth = float(cp.linalg.norm((theta - theta_old).ravel()) / (cp.linalg.norm(theta_old.ravel()) + 1e-30))
        eb = np.inf if beta_old is None else abs(beta - beta_old) / max(1.0, abs(beta))

        if progress_callback is not None:
            progress_callback(
                outer=outer + 1,
                max_outer=int(max_outer),
                beta=float(beta),
                rrms=float(theta_info.get("rms_interior", np.nan)),
                rmax=float(theta_info.get("max_interior", np.nan)),
            )
        
        if verbose and (
            outer == 0
            or outer % outer_print_every == 0
            or outer == int(max_outer) - 1
            or (eA < outer_tol_A and eth < outer_tol_theta)
        ):
            print(
                f"[eig FULL outer={outer:3d}] beta={beta:.9e} "
                f"eA={eA:.3e} eth={eth:.3e} eb={eb:.3e} "
                f"Rmax={theta_info.get('max_interior', np.nan):.3e} "
                f"Rrms={theta_info.get('rms_interior', np.nan):.3e}"
            )

        if eA < outer_tol_A and eth < outer_tol_theta:
            break

        beta_old = beta

    I = cp.abs(A) ** 2
    normalize_Ishape_inplace(I, ctx.dx, ctx.dy)

    return {
        "A": A,
        "theta": theta,
        "I": I,
        "beta": beta,
        "outer_iters": outer + 1,
        "err_A": eA,
        "err_theta": eth,
        "err_beta": eb,
        "theta_residual_max": theta_info.get("max_interior", np.nan),
        "theta_residual_rms": theta_info.get("rms_interior", np.nan),
    }

def lc_eigensoliton_existence_curve_v2(
    ctx,
    prdata,
    powers_mW,
    A_seed,
    theta_seed,
    *,
    branch_name="fundamental",
    subtract_carrier=True,
    save_dir=None,
    checkpoint_prefix="lc_eigensoliton",
    checkpoint_every=1,
    save_profiles=True,
    profile_every=1,
    stop_on_bad_residual=True,
    bad_residual_rms=5e-2,
    bad_residual_max=10.0,
    live_plot=True,
    live_plot_every=1,
    progress_callback=None,
    **solve_kwargs,
):
    if save_dir is None:
        save_dir = Path(".")
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    profile_dir = save_dir / f"{checkpoint_prefix}_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / f"{checkpoint_prefix}_partial.csv"
    state_path = save_dir / f"{checkpoint_prefix}_last_state.npz"

    # Save reproduction metadata once at run start.
    write_repro_json(
        save_dir,
        prdata=prdata,
        ctx=ctx,
        branch_name=branch_name,
        powers_mW=powers_mW,
        subtract_carrier=subtract_carrier,
        checkpoint_prefix=checkpoint_prefix,
        solve_kwargs=solve_kwargs,
    )

    rows = []

    beta_symbol, _ = make_beta_symbol_from_ctx(
        ctx,
        subtract_carrier=subtract_carrier,
    )

    A = A_seed.copy()
    theta = theta_seed.copy()
    I = None

    bi0 = float(ctx.bi)
    x_um, y_um = xy_um_from_ctx(ctx)

    try:
        for j, P in enumerate(powers_mW):
            if progress_callback is not None:
                progress_callback(
                    current=j,
                    total=len(powers_mW),
                    power_mW=float(P),
                )
            print(f"\n=== eigensoliton branch point {j+1}/{len(powers_mW)}: P = {P:.6g} mW ===")

            set_lc_power_from_prdata(ctx, prdata, P)

            if bool(getattr(ctx, "use_dual_grid", False)):
                ctx.ctx_theta.bi = ctx.bi
                ctx.ctx_theta.b = ctx.b
                ctx.ctx_theta.theta_bc = ctx.theta_bc

                sol = solve_lc_optical_eigensoliton_dg(
                    ctx,
                    A,
                    theta,
                    beta_symbol=beta_symbol,
                    **solve_kwargs,
                )

                A = sol["A"]
                theta_c = sol["theta_c"]          # coarse continuation state
                theta = theta_c                  # continuation variable for next P
                theta_plot = sol["theta"]         # fine-grid theta
                theta_bias_plot = prolong_theta(ctx.ctx_theta.theta_bias_2d, ctx.dual_grid)
                theta_bias_c = ctx.ctx_theta.theta_bias_2d

                grid_mode = "DG"

            else:
                def point_progress(**kw):
                    if progress_callback is not None:
                        progress_callback(
                            current=j,
                            total=len(powers_mW),
                            power_mW=float(P),
                            phase="outer",
                            **kw,
                        )
                
                sol = solve_lc_optical_eigensoliton_v2(
                    ctx,
                    A,
                    theta,
                    beta_symbol=beta_symbol,
                    progress_callback=point_progress,
                    **solve_kwargs,
                )

                A = sol["A"]
                theta = sol["theta"]              # full-grid continuation state
                theta_c = None
                theta_plot = theta
                theta_bias_plot = ctx.theta_bias_2d
                theta_bias_c = None

                grid_mode = "FULL"

            I = sol["I"]
            dtheta = theta_plot - theta_bias_plot

            x0, y0, sx, sy = beam_moments(I, x_um, y_um)

            row = {
                "branch": branch_name,
                "grid_mode": grid_mode,
                "P_mW": float(P),
                "bi": float(ctx.bi),
                "b": float(ctx.b),
                "theta_bc": float(getattr(ctx, "theta_bc", 0.0)),
                "beta": sol["beta"],
                "dtheta_max": float(cp.max(dtheta)),
                "dtheta_absmax": float(cp.max(cp.abs(dtheta))),
                "Imax": float(cp.max(I)),
                "x0_um": x0,
                "y0_um": y0,
                "sx_um": sx,
                "sy_um": sy,
                "outer_iters": sol["outer_iters"],
                "err_A": sol["err_A"],
                "err_theta": sol["err_theta"],
                "err_beta": sol["err_beta"],
                "theta_residual_max": sol["theta_residual_max"],
                "theta_residual_rms": sol["theta_residual_rms"],
            }

            rows.append(row)
            if progress_callback is not None:
                progress_callback(j + 1, len(powers_mW), float(P), "done")

            pd.DataFrame(rows).to_csv(csv_path, index=False)

            # ------------------------------------------------
            # Per-branch-point profile save
            # ------------------------------------------------
            if save_profiles and ((j + 1) % profile_every == 0):
                profile_path = profile_dir / f"{checkpoint_prefix}_P_{float(P):08.4f}mW.npz"

                save_lc_mode_npz(
                    profile_path,
                    ctx=ctx,
                    A=A,
                    theta=theta_plot,
                    theta_cont=theta,
                    I=I,
                    beta=float(sol["beta"]),
                    P_mW=float(P),
                    grid_mode=grid_mode,
                    branch_name=branch_name,
                    subtract_carrier=subtract_carrier,
                    theta_bias=theta_bias_plot,
                    theta_c=theta_c,
                    theta_bias_c=theta_bias_c,
                    x_um=x_um,
                    y_um=y_um,
                    extra=dict(
                        dtheta_absmax=cp.array(row["dtheta_absmax"]),
                        sx_um=cp.array(row["sx_um"]),
                        sy_um=cp.array(row["sy_um"]),
                    ),
                    beta_tol=2e-4 if grid_mode == "DG" else 2e-4,
                    raise_on_fail=False #(grid_mode != "DG"),
                )



                print(f"[profile] wrote {profile_path}")

            # ------------------------------------------------
            # Last-state checkpoint save
            # ------------------------------------------------
            if checkpoint_every is not None and ((j + 1) % checkpoint_every == 0):
                save_lc_mode_npz(
                    state_path,
                    ctx=ctx,
                    A=A,
                    theta=theta_plot,
                    theta_cont=theta,
                    I=I,
                    beta=float(sol["beta"]),
                    P_mW=float(P),
                    grid_mode=grid_mode,
                    branch_name=branch_name,
                    subtract_carrier=subtract_carrier,
                    theta_bias=theta_bias_plot,
                    theta_c=theta_c,
                    theta_bias_c=theta_bias_c,
                    x_um=x_um,
                    y_um=y_um,
                    extra=dict(j=cp.array(j)),
                    beta_tol=2e-4 if grid_mode == "DG" else 2e-4,
                    raise_on_fail=False,
                )

                print(f"[checkpoint] wrote {csv_path}")
                print(f"[checkpoint] wrote {state_path}")








            if live_plot and ((j + 1) % live_plot_every == 0):
                plot_eigensoliton_core(
                    row,
                    I,
                    theta_plot,
                    ctx,
                    zoom_um=max(5.0, 6.0 * max(row["sx_um"], row["sy_um"])),
                    clear=True,
                )

            if stop_on_bad_residual:
                bad = (
                    row["theta_residual_rms"] > bad_residual_rms
                    or row["theta_residual_max"] > bad_residual_max
                )
                if bad:
                    print(
                        "\n[stop] Bad theta residual detected:\n"
                        f"       P = {row['P_mW']:.6g} mW\n"
                        f"       Rmax = {row['theta_residual_max']:.3e}\n"
                        f"       Rrms = {row['theta_residual_rms']:.3e}\n"
                        "       Stopping branch continuation."
                    )
                    break

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

    except KeyboardInterrupt:
        print("\n[interrupt] Caught KeyboardInterrupt. Returning completed points.")

        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            cp.savez(
                state_path,
                A=A,
                theta=theta_plot if "theta_plot" in locals() else theta,
                theta_cont=theta,
                I=I,
                P_mW=cp.array(float(rows[-1]["P_mW"])),
                bi=cp.array(float(rows[-1]["bi"])),
                b=cp.array(float(rows[-1]["b"])),
                theta_bc=cp.array(float(rows[-1]["theta_bc"])),
                beta=cp.array(float(rows[-1]["beta"])),
                x_um=cp.asarray(x_um),
                y_um=cp.asarray(y_um),
                grid_mode=np.array(grid_mode),
                branch=np.array(branch_name),
            )

            print(f"[checkpoint] wrote {csv_path}")
            print(f"[checkpoint] wrote {state_path}")
        else:
            print("[interrupt] No completed branch points to save.")

    finally:
        ctx.bi = bi0
        if bool(getattr(ctx, "use_dual_grid", False)) and hasattr(ctx, "ctx_theta"):
            ctx.ctx_theta.bi = bi0

    return pd.DataFrame(rows), {"A": A, "theta": theta, "I": I}




# ============================================================
# Branch continuation / existence curve
# ============================================================

def beam_moments(I, x_um, y_um, eps=1e-30):
    X = cp.asarray(x_um)[:, None]
    Y = cp.asarray(y_um)[None, :]

    W = cp.sum(I) + eps

    x0 = cp.sum(I * X) / W
    y0 = cp.sum(I * Y) / W
    sx = cp.sqrt(cp.sum(I * (X - x0)**2) / W)
    sy = cp.sqrt(cp.sum(I * (Y - y0)**2) / W)

    return float(x0), float(y0), float(sx), float(sy)

def live_plot_eigensoliton_profile(row, I, theta, ctx, *, every=1):
    """
    Live profile plot after each completed power point.
    I, theta are CuPy arrays.
    """
    Icpu = cp.asnumpy(I)
    thcpu = cp.asnumpy(theta - ctx.theta_bias_2d)

    x_um, y_um = xy_um_from_ctx(ctx)
    x = cp.asnumpy(x_um)
    y = cp.asnumpy(y_um)

    extent = [y[0], y[-1], x[0], x[-1]]

    clear_output(wait=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = ax[0].imshow(
        Icpu,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    ax[0].set_title(
        f"I, P={row['P_mW']:.3f} mW\n"
        f"sx={row['sx_um']:.3f}, sy={row['sy_um']:.3f} um"
    )
    ax[0].set_xlabel("y [um]")
    ax[0].set_ylabel("x [um]")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(
        thcpu,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    ax[1].set_title(
        rf"$\theta-\theta_b$, beta={row['beta']:.5f}"
    )
    ax[1].set_xlabel("y [um]")
    ax[1].set_ylabel("x [um]")
    fig.colorbar(im1, ax=ax[1])

    display(fig)
    plt.close(fig)

def plot_eigensoliton_core(row, I, theta, ctx, zoom_um=5.0, clear=True):
    import numpy as np
    import matplotlib.pyplot as plt

    if clear:
        clear_output(wait=True)

    I_cpu = cp.asnumpy(I)
    th_cpu = cp.asnumpy(theta - ctx.theta_bias_2d)

    x_um_cp, y_um_cp = xy_um_from_ctx(ctx)
    x_um = cp.asnumpy(x_um_cp)
    y_um = cp.asnumpy(y_um_cp)

    ix, iy = np.unravel_index(np.argmax(I_cpu), I_cpu.shape)
    x0 = x_um[ix]
    y0 = y_um[iy]

    xmask = (x_um >= x0 - zoom_um) & (x_um <= x0 + zoom_um)
    ymask = (y_um >= y0 - zoom_um) & (y_um <= y0 + zoom_um)

    I_zoom = I_cpu[np.ix_(xmask, ymask)]
    th_zoom = th_cpu[np.ix_(xmask, ymask)]

    extent = [
        y_um[ymask][0], y_um[ymask][-1],
        x_um[xmask][0], x_um[xmask][-1],
    ]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = ax[0].imshow(
        I_zoom,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax[0].set_title(
        f"I core, P={row['P_mW']:.3f} mW\n"
        f"sx={row['sx_um']:.3f}, sy={row['sy_um']:.3f} µm"
    )
    ax[0].set_xlabel("y [µm]")
    ax[0].set_ylabel("x [µm]")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(
        th_zoom,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax[1].set_title(
        rf"$\theta-\theta_b$, $\beta$={row['beta']:.5f}"
    )
    ax[1].set_xlabel("y [µm]")
    ax[1].set_ylabel("x [µm]")
    fig.colorbar(im1, ax=ax[1])

    display(fig)
    plt.close(fig)

def lc_profile_metadata(ctx, *, grid_mode, branch_name, subtract_carrier=True):
    """
    Authoritative per-profile metadata.
    Values are taken from the active ctx at the moment the profile is saved.
    """
    return dict(
        # grid
        Nx=cp.array(int(ctx.Nx)),
        Ny=cp.array(int(ctx.Ny)),
        Nz=cp.array(int(ctx.Nz)),
        dx=cp.array(float(ctx.dx)),
        dy=cp.array(float(ctx.dy)),
        dz=cp.array(float(ctx.dz)),

        # optics
        lm=cp.array(float(ctx.lm)),
        refin=cp.array(float(ctx.refin)),
        kout=cp.array(float(ctx.kout)),
        ne=cp.array(float(ctx.ne)),
        no=cp.array(float(ctx.no)),

        # LC
        b=cp.array(float(ctx.b)),
        bi=cp.array(float(ctx.bi)),
        theta_bc=cp.array(float(getattr(ctx, "theta_bc", 0.0))),

        # labels
        grid_mode=np.array(str(grid_mode)),
        branch=np.array(str(branch_name)),
        subtract_carrier=np.array(bool(subtract_carrier)),
    )

def verify_saved_profile_beta(
    profile_path,
    ctx,
    *,
    subtract_carrier=True,
    beta_tol=2e-4,
    raise_on_fail=False,
):
    z = cp.load(profile_path, allow_pickle=True)
    try:
        A = z["A"]
        theta = z["theta"]
        beta_saved = float(cp.asnumpy(z["beta"]).ravel()[0])

        beta_symbol, _ = make_beta_symbol_from_ctx(ctx, subtract_carrier=subtract_carrier)
        beta_check = rayleigh_beta(ctx, A, beta_symbol, theta)

        err = abs(beta_check - beta_saved)

        msg = (
            f"[verify] beta_saved={beta_saved:.9e} "
            f"beta_rayleigh={beta_check:.9e} "
            f"diff={err:.3e}"
        )

        if err > beta_tol:
            msg += "  [WARN]"
            print(msg)
            if raise_on_fail:
                raise RuntimeError(
                    f"Saved profile failed beta self-check: diff={err:.3e}"
                )
        else:
            print(msg)

        return {
            "beta_saved": beta_saved,
            "beta_rayleigh": beta_check,
            "beta_diff": err,
            "passed": err <= beta_tol,
        }

    finally:
        z.close()

def save_lc_mode_npz(
    path,
    *,
    ctx,
    A,
    theta,
    I,
    beta,
    P_mW,
    grid_mode,
    branch_name,
    subtract_carrier=True,
    theta_cont=None,
    theta_bias=None,
    theta_c=None,
    theta_bias_c=None,
    x_um=None,
    y_um=None,
    extra=None,
    beta_tol=None,
    raise_on_fail=False,
):
    path = Path(path)

    if theta_cont is None:
        theta_cont = theta
    if theta_bias is None:
        theta_bias = ctx.theta_bias_2d
    if x_um is None or y_um is None:
        x_um, y_um = xy_um_from_ctx(ctx)

    if beta_tol is None:
        beta_tol = 1e-4 if str(grid_mode).upper() == "DG" else 1e-5

    beta_symbol_check, _ = make_beta_symbol_from_ctx(
        ctx,
        subtract_carrier=subtract_carrier,
    )
    beta_rayleigh = rayleigh_beta(ctx, A, beta_symbol_check, theta)
    beta_diff = abs(float(beta_rayleigh) - float(beta))

    payload = dict(
        A=A,
        theta=theta,
        theta_cont=theta_cont,
        I=I,
        theta_bias=theta_bias,
        beta=cp.array(float(beta)),
        P_mW=cp.array(float(P_mW)),
        x_um=cp.asarray(x_um),
        y_um=cp.asarray(y_um),

        beta_verify_rayleigh=cp.array(float(beta_rayleigh)),
        beta_verify_diff=cp.array(float(beta_diff)),
        beta_verify_tol=cp.array(float(beta_tol)),
        beta_verify_passed=cp.array(bool(beta_diff <= beta_tol)),
    )

    payload.update(
        lc_profile_metadata(
            ctx,
            grid_mode=grid_mode,
            branch_name=branch_name,
            subtract_carrier=subtract_carrier,
        )
    )

    if theta_c is not None:
        payload["theta_c"] = theta_c
    if theta_bias_c is not None:
        payload["theta_bias_c"] = theta_bias_c
    if extra:
        payload.update(extra)

    cp.savez(path, **payload)

    msg = (
        f"[save] {path}\n"
        f"[verify] beta_saved={float(beta):.9e} "
        f"beta_rayleigh={float(beta_rayleigh):.9e} "
        f"diff={beta_diff:.3e} tol={beta_tol:.1e}"
    )
    if beta_diff > beta_tol:
        msg += "  [WARN]"
        print(msg)
        if raise_on_fail:
            raise RuntimeError(f"Saved profile failed beta self-check: diff={beta_diff:.3e}")
    else:
        print(msg)

    return payload

def find_eigensoliton_runs(root="./cglab/mcroning"):
    """
    Fast targeted finder. Does NOT recursively crawl all files.
    Finds:
      ./eigensoliton_prod/TEMxx_*/DG_stage
      ./eigensoliton_prod/TEMxx_*/FULL_polish
      ./eigensolitons_*
    """
    root = Path(root)
    runs = []

    def is_run_dir(p):

        return (

            (p / "lc_eigensoliton_profiles").exists()

            or bool(list(p.glob("lc_eigensoliton_P_*mW.npz")))

            or (p / "lc_eigensoliton_repro.json").exists()

        )

    # New structured production layout

    for prod in root.glob("eigensoliton_*"):

        if not prod.is_dir():

            continue

        for sub in prod.glob("*"):

            if not sub.is_dir():

                continue

            for stage in sub.glob("*"):

                if stage.is_dir() and is_run_dir(stage):

                    runs.append(stage)

    # Older flat layout
    for p in root.glob("eigensolitons_*"):
        if p.is_dir() and (p / "lc_eigensoliton_profiles").exists():
            runs.append(p)

    return sorted(set(runs))

def list_run_profiles(run_dir, checkpoint_prefix="lc_eigensoliton"):
    run_dir = Path(run_dir)

    # normal existence-curve layout
    prof_dir = run_dir / f"{checkpoint_prefix}_profiles"
    files = sorted(prof_dir.glob(f"{checkpoint_prefix}_P_*mW.npz"))

    # polish legacy layout, if profiles were saved directly in folder
    if not files:
        files = sorted(run_dir.glob(f"{checkpoint_prefix}_P_*mW.npz"))

    return files

def load_repro_json(run_dir, checkpoint_prefix="lc_eigensoliton"):
    path = Path(run_dir) / f"{checkpoint_prefix}_repro.json"
    if not path.exists():
        print(f"[warn] no repro json found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def _npz_keys(z):
    return set(z.npz_file.files)

def _npz_scalar(z, key, default=None):
    if key not in _npz_keys(z):
        return default
    return float(cp.asnumpy(z[key]).ravel()[0])

def _npz_string(z, key, default="LEGACY"):
    keys = _npz_keys(z)
    if key not in keys:
        return default

    # String arrays should be read from the underlying NumPy npz file,
    # not through CuPy, because CuPy does not support unicode dtype <U.
    v = z.npz_file[key]

    try:
        return str(v.item())
    except Exception:
        return str(v)

def read_profile_summary(profile_path):
    z = cp.load(profile_path, allow_pickle=True)
    try:
        out = {
            "path": Path(profile_path),
            "P_mW": _npz_scalar(z, "P_mW", np.nan),
            "bi": _npz_scalar(z, "bi", np.nan),
            "beta": _npz_scalar(z, "beta", np.nan),
            "grid_mode": _npz_string(z, "grid_mode", "LEGACY"),
            "branch": _npz_string(z, "branch", "LEGACY"),
        }
    finally:
        z.close()
    return out

def choose_saved_run(root="./cglab/mcroning"):
    runs = find_eigensoliton_runs(root)
    if not runs:
        raise FileNotFoundError(f"No eigensoliton run folders found under {root}")

    print("\nAvailable runs:")
    for i, r in enumerate(runs):
        print(f"  [{i}] {r}")

    i = int(input("Choose run index: "))
    return runs[i]

def choose_power_profile(run_dir, checkpoint_prefix="lc_eigensoliton"):
    profiles = list_run_profiles(run_dir, checkpoint_prefix)
    if not profiles:
        raise FileNotFoundError(f"No profiles found in {run_dir}")

    rows = [read_profile_summary(p) for p in profiles]

    print("\nAvailable saved powers:")
    for i, r in enumerate(rows):
        print(
            f"  [{i}] P={r['P_mW']:.6g} mW, "
            f"beta={r['beta']:.8g}, grid={r['grid_mode']}"
        )

    i = int(input("Choose power index: "))
    return rows[i]["path"]


# ------------------------------------------------------------
# Load one saved nonlinear mode, new or legacy format
# ------------------------------------------------------------

def load_saved_mode_profile(profile_path, ctx=None):
    z = cp.load(profile_path, allow_pickle=True)
    keys = _npz_keys(z)

    try:
        mode = {
            "path": Path(profile_path),
            "A": z["A"].astype(cp.complex64),
            "theta": z["theta"].astype(cp.float32),
            "I": z["I"].astype(cp.float32),
            "P_mW": _npz_scalar(z, "P_mW"),
            "bi": _npz_scalar(z, "bi"),
            "beta": _npz_scalar(z, "beta"),
        }

        # New format has theta_cont; legacy does not.
        mode["theta_cont"] = (
            z["theta_cont"].astype(cp.float32)
            if "theta_cont" in keys
            else mode["theta"].copy()
        )

        # Bias is essential; legacy fallback uses current ctx.
        if "theta_bias" in keys:
            mode["theta_bias"] = z["theta_bias"].astype(cp.float32)
        elif ctx is not None and hasattr(ctx, "theta_bias_2d"):
            mode["theta_bias"] = ctx.theta_bias_2d.copy()
        else:
            raise KeyError("Missing theta_bias and no ctx.theta_bias_2d fallback was provided.")

        mode["b"] = _npz_scalar(z, "b", float(getattr(ctx, "b", np.nan)))
        mode["theta_bc"] = _npz_scalar(
            z,
            "theta_bc",
            float(getattr(ctx, "theta_bc", 0.0)),
        )

        if "x_um" in keys and "y_um" in keys:
            mode["x_um"] = cp.asarray(z["x_um"])
            mode["y_um"] = cp.asarray(z["y_um"])
        elif ctx is not None:
            x_um, y_um = xy_um_from_ctx(ctx)
            mode["x_um"] = cp.asarray(x_um)
            mode["y_um"] = cp.asarray(y_um)
        else:
            mode["x_um"] = None
            mode["y_um"] = None

        mode["grid_mode"] = _npz_string(z, "grid_mode", "LEGACY")
        mode["branch"] = _npz_string(z, "branch", "LEGACY")

        if "theta_c" in keys:
            mode["theta_c"] = z["theta_c"].astype(cp.float32)
        if "theta_bias_c" in keys:
            mode["theta_bias_c"] = z["theta_bias_c"].astype(cp.float32)

    finally:
        z.close()

    return mode


# ------------------------------------------------------------
# Reinsert saved mode into ctx
# ------------------------------------------------------------

def install_saved_mode_into_ctx(ctx, mode, *, restore_b=False):
    """
    Sets b/bi, theta state, and DG coarse theta if available.

    Legacy DG profiles often do NOT contain theta_c. In that case, use
    ctx.use_dual_grid=False before calling stability_test_one_profile,
    or rerun with the updated save block.
    """
    ctx.bi = float(mode["bi"])
    if restore_b:
        ctx.b = float(mode["b"])
    ctx.theta_bc = float(mode["theta_bc"])

    if bool(getattr(ctx, "use_dual_grid", False)):
        if "theta_c" not in mode:
            print("[legacy warning] ctx.use_dual_grid=True but profile has no theta_c.")
            print("[legacy warning] Falling back to FULL-grid solve for this stability test.")
            ctx.use_dual_grid = False

    if bool(getattr(ctx, "use_dual_grid", False)):
        ctx.ctx_theta.bi = ctx.bi
        ctx.ctx_theta.b = ctx.b
        ctx.ctx_theta.theta_bc = ctx.theta_bc

        ctx.ctx_theta.theta_bias_2d = mode["theta_bias_c"].copy()
        ctx.ctx_theta.theta_full = cp.repeat(mode["theta_c"][None, :, :], ctx.Nz, axis=0)

        ctx.theta_full = cp.repeat(mode["theta"][None, :, :], ctx.Nz, axis=0)
        theta_seed = mode["theta_c"].copy()

    else:
        ctx.theta_bias_2d = mode["theta_bias"].copy()
        ctx.theta_full = cp.repeat(mode["theta"][None, :, :], ctx.Nz, axis=0)
        theta_seed = mode["theta"].copy()

    if hasattr(ctx, "theta_prev_time"):
        ctx.theta_prev_time = ctx.theta_full.copy()

    return theta_seed


# ------------------------------------------------------------
# Perturbations
# ------------------------------------------------------------

class LCDualGrid:
    ctx_f: object   # fine (optics)
    ctx_t: object   # coarse (theta)

    theta_z_stride: int = 1

    def __post_init__(self):
        self.use_dual_grid = True

        # sanity
        assert self.ctx_f.Nx % self.ctx_t.Nx == 0
        assert self.ctx_f.Ny % self.ctx_t.Ny == 0

        self.rx = self.ctx_f.Nx // self.ctx_t.Nx
        self.ry = self.ctx_f.Ny // self.ctx_t.Ny


# ============================================================
# Fine → coarse intensity (energy-preserving)
# ============================================================

def restrict_I(I_f, dg, *, normalize=False):
    ctx_f, ctx_t = dg.ctx_f, dg.ctx_t

    Nx_t, Ny_t = ctx_t.Nx, ctx_t.Ny
    rx, ry = dg.rx, dg.ry

    I_c = I_f.reshape(Nx_t, rx, Ny_t, ry).mean(axis=(1, 3))
    I_c = I_c.astype(cp.float32, copy=False)

    if normalize:
        normalize_Ishape_inplace(I_c, ctx_t.dx, ctx_t.dy)

    return I_c


# ============================================================
# Coarse → fine theta (smooth interpolation)
# ============================================================

def prolong_theta(theta_c, dg):
    ctx_f, ctx_t = dg.ctx_f, dg.ctx_t

    zx = ctx_f.Nx / ctx_t.Nx
    zy = ctx_f.Ny / ctx_t.Ny

    theta_f = cndi.zoom(theta_c, (zx, zy), order=3)

    theta_f = theta_f[:ctx_f.Nx, :ctx_f.Ny]

    theta_f[0, :]  = ctx_t.theta_bc
    theta_f[-1, :] = ctx_t.theta_bc

    return theta_f.astype(cp.float32, copy=False)

# ============================================================
# Dual-grid optics + theta update (single slice)
# ============================================================

def dualgrid_slice_update(dg, A_f, theta_c):

    ctx_f, ctx_t = dg.ctx_f, dg.ctx_t

    # --- coarse → fine for optics
    theta_f = prolong_theta(theta_c, dg)

    # --- optics step (UNCHANGED solver)
    A_out = propagate_one_slice(ctx_f, A_f, theta_f)

    # --- intensity
    I_f = cp.abs(A_out)**2

    # --- fine → coarse
    I_c = restrict_I(I_f, dg)

    # --- theta update (UNCHANGED solver)
    theta_new = update_theta_slice(ctx_t, theta_c, I_c)

    return A_out, theta_new

# ============================================================
# Dual-grid z propagation
# ============================================================

def run_dualgrid_z(dg, A0, theta0):

    ctx_f, ctx_t = dg.ctx_f, dg.ctx_t

    Nz_f = ctx_f.Nz
    Nz_t = ctx_t.Nz

    A = A0.copy()
    theta_c = theta0.copy()

    theta_stride = dg.theta_z_stride

    for kf in range(Nz_f):

        kt = kf // theta_stride

        A, theta_new = dualgrid_slice_update(dg, A, theta_c)

        if (kf % theta_stride) == 0:
            theta_c = theta_new

    return A, theta_c

# ============================================================
# Dual-grid z propagation using existing solvers
# ============================================================

def run_dualgrid_unified(
    dg,
    ctx_f,
    ctx_t,
    A0,
    theta0_c,
    *,
    theta_z_stride=2,
):
    """
    Minimal dual-grid wrapper using your existing machinery.
    """

    Nz_f = ctx_f.Nz

    A = A0.copy().astype(cp.complex64, copy=False)
    theta_c = theta0_c.copy().astype(cp.float32, copy=False)

    for k in range(Nz_f):

        # ----------------------------------------------------
        # 1. Prolong theta to fine grid
        # ----------------------------------------------------
        theta_f = prolong_theta(theta_c, dg)

        # ----------------------------------------------------
        # 2. Optical propagation (ONE slice)
        # ----------------------------------------------------
        A = propagate_one_slice(ctx_f, A, theta_f)

        # ----------------------------------------------------
        # 3. Compute intensity
        # ----------------------------------------------------
        I_f = cp.abs(A)**2

        # ----------------------------------------------------
        # 4. Every stride: update theta
        # ----------------------------------------------------
        if (k % theta_z_stride) == 0:

            I_c = restrict_I(I_f, dg)

            # --- call your EXISTING theta update
            theta_c = solve_theta_for_slice(
                ctx_t,
                theta_c,
                I_c
            )

    return A, theta_c

def propagate_one_slice(ctx, A, theta_f):
    """
    Wrap your existing angular spectrum step for ONE slice.
    """
    # This depends on your code, but likely:

    A_out = propagate_optics_substeps(
        ctx,
        A,
        theta_f
    )

    return A_out

def restrict_theta_fine_to_coarse(theta_f, ctx_f, ctx_t):
    class DG_tmp:
        pass

    dg_tmp = DG_tmp()
    dg_tmp.ctx_f = ctx_f
    dg_tmp.ctx_t = ctx_t
    dg_tmp.rx = ctx_f.Nx // ctx_t.Nx
    dg_tmp.ry = ctx_f.Ny // ctx_t.Ny

    Nx_t, Ny_t = ctx_t.Nx, ctx_t.Ny
    rx, ry = dg_tmp.rx, dg_tmp.ry

    theta_c = theta_f.reshape(Nx_t, rx, Ny_t, ry).mean(axis=(1, 3))
    theta_c = theta_c.astype(cp.float32, copy=False)

    theta_bc = cp.float32(getattr(ctx_t, "theta_bc", getattr(ctx_f, "theta_bc", 0.0)))
    theta_c[0, :] = theta_bc
    theta_c[-1, :] = theta_bc

    return theta_c

# ============================================================
# @title Dual-grid eigensoliton solver + existence curve
# Optics on fine ctx, theta on coarse ctx_t
# ============================================================

def solve_lc_optical_eigensoliton_dg(
    ctx,
    A_seed,
    theta_seed_c,
    *,
    beta_symbol=None,
    max_outer=60,
    mix_theta=0.2,
    optical_dtau=0.02,
    optical_steps=300,
    optical_tol=1e-8,
    theta_residual_tol_max=5e-2,
    theta_residual_tol_rms=1e-2,
    theta_outer_passes=8,
    outer_tol_A=1e-6,
    outer_tol_theta=1e-6,
    outer_tol_beta=1e-7,
    outer_print_every=5,
    verbose=True,
):
    assert bool(getattr(ctx, "use_dual_grid", False))
    ctx_f = ctx
    ctx_t = ctx.ctx_theta
    dg = ctx.dual_grid

    if beta_symbol is None:
        beta_symbol, _ = make_beta_symbol_from_ctx(ctx_f, subtract_carrier=True)

    A = normalize_A_unit_intensity(
        A_seed.astype(cp.complex64, copy=True),
        ctx_f.dx,
        ctx_f.dy,
    )

    theta_c = theta_seed_c.astype(cp.float32, copy=True)
    theta_c = enforce_theta_constraints(theta_c, ctx_t.theta_clamp)

    beta_old = None
    beta = np.nan
    eA = np.inf
    eth = np.inf
    eb = np.inf

    theta_info = {
        "max_interior": np.nan,
        "rms_interior": np.nan,
    }

    sS, offS, diagS, lamS = prepare_cn_ky_operator(
        dt=float(ctx_t.dtau_static),
        mobility=float(ctx_t.mobility),
        du=float(ctx_t.du),
        dv=float(ctx_t.dv),
        Ny=int(ctx_t.Ny),
    )

    for outer in range(int(max_outer)):
        A_old = A.copy()
        theta_old_c = theta_c.copy()

        # coarse theta -> fine optics
        theta_f = prolong_theta(theta_c, dg)

        # optical eigenmode on fine grid
        A, beta = solve_optical_mode_for_theta(
            ctx_f,
            A,
            theta_f,
            beta_symbol,
            dtau=optical_dtau,
            nsteps=optical_steps,
            tol=optical_tol,
            verbose=False,
        )

        if getattr(ctx_f, "T_mode", "00").upper() not in ("00", "TEM00"):
            A, theta_f = project_mode_parity(A, theta_f, ctx_f.T_mode)

        # fine intensity -> coarse theta grid
        I_f = cp.abs(A)**2
        normalize_Ishape_inplace(I_f, ctx_f.dx, ctx_f.dy)
        I_c = restrict_I(I_f, dg)
        if False:
            theta_new_c, theta_info = strict_static_relax_slice(
                theta_c,
                I_c,
                theta_c,
                theta_c,
                ctx_t,
                sS=sS,
                offS=offS,
                diagS=diagS,
                lamS=lamS,
                use_linear_seed=True,
                early_accept_linear_seed=True,
                residual_tol_max=theta_residual_tol_max,
                residual_tol_rms=theta_residual_tol_rms,
                max_outer_passes=theta_outer_passes,
                verbose=False,
            )

##temp
        # --- 1. branch-safe linear seed around FD bias
        # --- default (fallback)

        theta_new_c = theta_c.copy()
        #theta_new_c = enforce_theta_constraints(theta_new_c, ctx_t.theta_clamp)
        theta_info = {"max_interior": np.nan, "rms_interior": np.nan}
        out_lin = solve_theta_perturbation_dirichletx_periody(
            theta_bias=ctx_t.theta_bias_2d,
            Ixy=I_c,
            b=float(ctx_t.b),
            bi=float(ctx_t.bi),
            du=float(ctx_t.du),
            dv=float(ctx_t.dv),
            xp=cp,
            return_theta=True,
            check_residual=False,
            verbose=False,
        )

        theta_lin_c = out_lin["theta"].astype(cp.float32, copy=False)

        theta_bc = cp.float32(getattr(ctx_t, "theta_bc", 0.0))
        theta_lin_c[0, :] = theta_bc
        theta_lin_c[-1, :] = theta_bc

        # --- 2. nonlinear polish, seeded from branch-safe linear result
        theta_new_c, theta_info = strict_static_relax_slice(
            theta_lin_c,      # seed from linear branch
            I_c,
            theta_c,          # z-neighbor refs; gamma_z=0 in eigensolver anyway
            theta_c,
            ctx_t,
            sS=sS,
            offS=offS,
            diagS=diagS,
            lamS=lamS,
            use_linear_seed=False,          # do NOT override seed
            early_accept_linear_seed=False,
            residual_tol_max=theta_residual_tol_max,
            residual_tol_rms=theta_residual_tol_rms,
            max_outer_passes=2,             # start small
            verbose=False,
        )
        theta_new_c = enforce_theta_constraints(theta_new_c, ctx_t.theta_clamp)

        # --- 3. branch guard: reject if it collapses toward zero branch
        bias_norm = float(cp.sqrt(cp.mean(ctx_t.theta_bias_2d[1:-1, :]**2)))
        new_norm = float(cp.sqrt(cp.mean(theta_new_c[1:-1, :]**2)))

        if new_norm < 0.5 * bias_norm:
            print("[DG theta] nonlinear polish rejected: fell toward zero branch")
            theta_new_c = theta_lin_c
            theta_info = {"max_interior": np.nan, "rms_interior": np.nan}
##temp

        theta_c = ((1.0 - mix_theta) * theta_c + mix_theta * theta_new_c).astype(cp.float32, copy=False)
        theta_c = enforce_theta_constraints(theta_c, ctx_t.theta_clamp)

        eA = float(cp.linalg.norm((A - A_old).ravel()) / (cp.linalg.norm(A_old.ravel()) + 1e-30))
        eth = float(cp.linalg.norm((theta_c - theta_old_c).ravel()) / (cp.linalg.norm(theta_old_c.ravel()) + 1e-30))
        eb = np.inf if beta_old is None else abs(beta - beta_old) / max(1.0, abs(beta))

        if verbose and (
            outer == 0
            or outer % outer_print_every == 0
            or outer == max_outer - 1
            or (eA < outer_tol_A and eth < outer_tol_theta and eb < outer_tol_beta)
        ):
            print(
                f"[eig DG outer={outer:3d}] beta={beta:.9e} "
                f"eA={eA:.3e} eth={eth:.3e} eb={eb:.3e} "
                f"Rmax={theta_info['max_interior']:.3e} "
                f"Rrms={theta_info['rms_interior']:.3e}"
            )

        if eA < outer_tol_A and eth < outer_tol_theta and eb < outer_tol_beta:
            break

        beta_old = beta

    theta_f = prolong_theta(theta_c, dg)
    I_f = cp.abs(A)**2
    normalize_Ishape_inplace(I_f, ctx_f.dx, ctx_f.dy)

    I_c = restrict_I(I_f, dg)
    R = lc_residual2d_dirichletx_periody(
        theta_c,
        I_c,
        b=float(ctx_t.b),
        bi=float(ctx_t.bi),
        du=float(ctx_t.du),
        dv=float(ctx_t.dv),
        mobility=float(ctx_t.mobility),
    )
    stats = residual_stats_2d(R)

    return {
        "A": A,
        "theta": theta_f,       # fine-grid theta for plotting/optics
        "theta_c": theta_c,     # coarse-grid theta for continuation
        "I": I_f,
        "I_c": I_c,
        "beta": beta,
        "outer_iters": outer + 1,
        "err_A": eA,
        "err_theta": eth,
        "err_beta": eb,
        "theta_residual_max": stats["max_interior"],
        "theta_residual_rms": stats["rms_interior"],
    }

def polish_from_DG(ctx, prdata, dg, profile_path, *, n_outer=8):
    prof = cp.load(profile_path)
    keys = set(prof.npz_file.files)

    A = prof["A"].astype(cp.complex64, copy=True)

    if "theta_c" in keys:
        theta = prolong_theta(prof["theta_c"].astype(cp.float32), dg)
    else:
        theta_raw = prof["theta"].astype(cp.float32)
        if theta_raw.shape == (ctx.Nx, ctx.Ny):
            theta = theta_raw.copy()
        else:
            theta = prolong_theta(theta_raw, dg)

    theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
    theta[0, :] = theta_bc
    theta[-1, :] = theta_bc
    theta = enforce_theta_constraints(theta, ctx.theta_clamp)


    # switch to full grid
    ctx.use_dual_grid = False

    beta = float(cp.asnumpy(prof["beta"]).ravel()[0])

    for outer in range(n_outer):
        A_old = A.copy()
        theta_old = theta.copy()

        # --- optical update
        beta_symbol, _ = make_beta_symbol_from_ctx(
            ctx,
            subtract_carrier=True,   # or False, matching the DG run
        )

        A, beta = solve_optical_mode_for_theta(
            ctx,
            A,
            theta,
            beta_symbol=beta_symbol,
            dtau=0.02,
            nsteps=120,
            tol=1e-7,
            verbose=False,
        )

        # --- intensity
        I = cp.abs(A)**2
        normalize_Ishape_inplace(I, ctx.dx, ctx.dy)

        # --- FULL nonlinear theta solve
        theta_new, _ = solve_theta_for_eigenmode_I(
            ctx,
            theta,
            I,
            residual_tol_max=2e-1,
            residual_tol_rms=3e-2,
            max_outer_passes=4,
            use_linear_seed=False,
            verbose=False,
        )

        theta = 0.1 * theta_new + 0.9 * theta
        theta = enforce_theta_constraints(theta, ctx.theta_clamp)

        # --- convergence check
        eA = float(cp.linalg.norm((A - A_old).ravel()) /
                   (cp.linalg.norm(A_old.ravel()) + 1e-30))

        if eA < 1e-5:
            print(f"[polish] converged in {outer} steps")
            break

    return {
        "A": A,
        "theta": theta,
        "beta": float(beta),
    }

