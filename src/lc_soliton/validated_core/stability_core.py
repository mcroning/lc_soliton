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
from .eigenmode_core import *

def paired_stability_metrics(unpert_npz, pert_npz, ctx):
    u = cp.load(unpert_npz, allow_pickle=True)
    p = cp.load(pert_npz, allow_pickle=True)

    try:
        A_u = u["A_test"]
        A_p = p["A_test"]
        I_u = u["I_test"]
        I_p = p["I_test"]
        th_u = u["theta_test"]
        th_p = p["theta_test"]

        dA_pair = rel_l2(A_u, A_p)
        dI_pair = rel_l2(I_u, I_p)
        dth_pair = rel_l2(th_u, th_p)

        return {
            "paired_rel_A_L2": dA_pair,
            "paired_rel_I_L2": dI_pair,
            "paired_rel_theta_L2": dth_pair,
        }

    finally:
        u.close()
        p.close()

# @title Beam Builder and runner helpers

def amp_power(A, ctx):
    return cp.sum(cp.abs(A)**2) * ctx.dx * ctx.dy

def renorm_like(A, Aref, ctx):
    return A * cp.sqrt(amp_power(Aref, ctx) / amp_power(A, ctx))

def perturb_saved_A(
    A,
    ctx,
    *,
    kind="noise",
    eps=1e-5,
    seed=1,
    dx_shift_um=0.0,
    dy_shift_um=0.0,
    phase_tilt_x=0.0,
    phase_tilt_y=0.0,
):
    rng = cp.random.RandomState(seed)
    A0 = A.astype(cp.complex64, copy=True)
    Ap = A0.copy()

    if kind in ("noise", "mixed"):
        eta = rng.standard_normal(A0.shape) + 1j * rng.standard_normal(A0.shape)
        eta /= cp.sqrt(cp.mean(cp.abs(eta)**2))
        Ap = Ap + eps * cp.sqrt(cp.mean(cp.abs(A0)**2)) * eta

    if kind in ("phase", "mixed"):
        phi = rng.standard_normal(A0.shape)
        phi /= cp.std(phi)
        Ap = Ap * cp.exp(1j * eps * phi)

    if kind == "amplitude":
        eta = rng.standard_normal(A0.shape)
        eta /= cp.std(eta)
        Ap = Ap * (1.0 + eps * eta)

    if kind in ("shift", "mixed") and (dx_shift_um != 0 or dy_shift_um != 0):
        if not hasattr(ctx, "fx") or not hasattr(ctx, "fy"):
            raise AttributeError("Need ctx.fx and ctx.fy for Fourier shift.")
        FX, FY = cp.meshgrid(cp.asarray(ctx.fx), cp.asarray(ctx.fy), indexing="ij")
        sh = cp.exp(-2j * cp.pi * (FX * dx_shift_um + FY * dy_shift_um))
        Ap = cp.fft.ifft2(cp.fft.fft2(Ap) * sh)

    if kind in ("tilt", "mixed") and (phase_tilt_x != 0 or phase_tilt_y != 0):
        X, Y = cp.meshgrid(cp.asarray(ctx.x_um), cp.asarray(ctx.y_um), indexing="ij")
        Ap = Ap * cp.exp(1j * (phase_tilt_x * X + phase_tilt_y * Y))

    return renorm_like(Ap, A0, ctx)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def rel_l2(a, b):
    a = cp.asarray(a)
    b = cp.asarray(b)
    return float(cp.sqrt(cp.sum(cp.abs(a - b)**2) / (cp.sum(cp.abs(a)**2) + 1e-30)))

def overlap(a, b, ctx):
    a = cp.asarray(a)
    b = cp.asarray(b)
    num = cp.abs(cp.sum(cp.conj(a) * b) * ctx.dx * ctx.dy)**2
    den = amp_power(a, ctx) * amp_power(b, ctx)
    return float(num / (den + 1e-30))

def perturbation_norm(Aref, Ap, ctx):
    return rel_l2(Aref, Ap)

def stability_metrics(mode, sol, A_pert, ctx):
    A0 = mode["A"]
    I0 = mode["I"]
    th0 = mode["theta"]

    A1 = sol["A"]
    I1 = sol["I"]
    th1 = sol["theta"]

    d0 = perturbation_norm(A0, A_pert, ctx)
    dA = rel_l2(A0, A1)
    dI = rel_l2(I0, I1)
    dtheta = rel_l2(th0, th1)
    ov = overlap(A0, A1, ctx)

    gain_A = dA / max(d0, 1e-30)
    gain_I = dI / max(d0, 1e-30)
    gain_theta = dtheta / max(d0, 1e-30)

    return {
        "perturb_norm": d0,
        "rel_A_L2": dA,
        "rel_I_L2": dI,
        "rel_theta_L2": dtheta,
        "overlap": ov,
        "gain_A": gain_A,
        "gain_I": gain_I,
        "gain_theta": gain_theta,
        "log_gain_A": np.log(max(gain_A, 1e-300)),
        "log_gain_I": np.log(max(gain_I, 1e-300)),
        "log_gain_theta": np.log(max(gain_theta, 1e-300)),
    }


# ------------------------------------------------------------
# Solver wrapper
# ------------------------------------------------------------

def solve_saved_mode_once(
    ctx,
    mode,
    A_in,
    theta_seed,
    *,
    subtract_carrier=True,
    solve_kwargs=None,
):
    if solve_kwargs is None:
        solve_kwargs = {}
    bad_keys = {"stop_on_bad_residual", "live_plot", "save_dir"}
    solve_kwargs = {k: v for k, v in solve_kwargs.items() if k not in bad_keys}
    beta_symbol, _ = make_beta_symbol_from_ctx(ctx, subtract_carrier=subtract_carrier)

    if bool(getattr(ctx, "use_dual_grid", False)):
        sol = solve_lc_optical_eigensoliton_dg(
            ctx,
            A_in,
            theta_seed,
            beta_symbol=beta_symbol,
            **solve_kwargs,
        )
    else:
        sol = solve_lc_optical_eigensoliton_v2(
            ctx,
            A_in,
            theta_seed,
            beta_symbol=beta_symbol,
            **solve_kwargs,
        )

    return sol

def stability_test_one_profile(
    ctx_template,
    profile_path,
    *,
    perturbations=None,
    solve_kwargs=None,
    subtract_carrier=True,
    copy_ctx=True,
    save_dir=None,
    verbose=True,
):
    mode = load_saved_mode_profile(profile_path, ctx=ctx_template)

    if perturbations is None:
        perturbations = [
            dict(label="noise_1e-5", kind="noise", eps=1e-5, seed=1),
        ]

    rows = []

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for p in perturbations:
        label = p.get("label", p.get("kind", "perturb"))

        ctx = copy.deepcopy(ctx_template) if copy_ctx else ctx_template
        theta_seed = install_saved_mode_into_ctx(ctx, mode, restore_b=False)

        p_call = dict(p)
        p_call.pop("label", None)

        A_pert = perturb_saved_A(mode["A"], ctx, **p_call)

        if verbose:
            print(f"\n[P={mode['P_mW']:.6g} mW] perturb={label}")
            print(f"  grid_mode saved = {mode['grid_mode']}")
            print(f"  ctx.use_dual_grid = {getattr(ctx, 'use_dual_grid', False)}")

        sol = solve_saved_mode_once(
            ctx,
            mode,
            A_pert,
            theta_seed,
            subtract_carrier=subtract_carrier,
            solve_kwargs=solve_kwargs,
        )

        m = stability_metrics(mode, sol, A_pert, ctx)

        row = {
            "profile": Path(profile_path).name,
            "P_mW": mode["P_mW"],
            "beta_ref": mode["beta"],
            "grid_mode": mode["grid_mode"],
            "branch": mode["branch"],
            "perturb": label,
            **m,
            "beta_test": float(sol["beta"]),
            "err_A": float(sol.get("err_A", np.nan)),
            "err_theta": float(sol.get("err_theta", np.nan)),
            "err_beta": float(sol.get("err_beta", np.nan)),
            "theta_residual_max": float(sol.get("theta_residual_max", np.nan)),
            "theta_residual_rms": float(sol.get("theta_residual_rms", np.nan)),
            "outer_iters": int(sol.get("outer_iters", -1)),
        }

        rows.append(row)

        if save_dir is not None:
            out = save_dir / f"{Path(profile_path).stem}__stab__{label}.npz"
            cp.savez(
                out,
                A_ref=mode["A"],
                A_pert=A_pert,
                A_test=sol["A"],
                I_ref=mode["I"],
                I_test=sol["I"],
                theta_ref=mode["theta"],
                theta_test=sol["theta"],
                P_mW=cp.array(mode["P_mW"]),
                beta_ref=cp.array(mode["beta"]),
                beta_test=cp.array(float(sol["beta"])),
            )
            if verbose:
                print(f"[save] {out}")

        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    return pd.DataFrame(rows)

def rebuild_ctx_from_run(run_dir, checkpoint_prefix="lc_eigensoliton"):
    run_dir = Path(run_dir)
    repro_path = run_dir / f"{checkpoint_prefix}_repro.json"

    if not repro_path.exists():
        raise FileNotFoundError(f"Missing repro file: {repro_path}")

    with open(repro_path, "r") as f:
        repro = json.load(f)

    ctx, prdata = rebuild_ctx_from_repro(repro)

    # FULL_polish stability should be full-grid
    ctx.use_dual_grid = False
    for attr in ["ctx_theta", "dual_grid"]:
        if hasattr(ctx, attr):
            delattr(ctx, attr)

    return ctx, prdata

def assert_ctx_matches_profile(ctx, profile):
    shape = tuple(profile["A"].shape)

    if (ctx.Nx, ctx.Ny) != shape:
        raise ValueError(
            f"ctx grid {(ctx.Nx, ctx.Ny)} "
            f"does not match saved profile {shape}"
        )

    if tuple(ctx.fxy2.shape) != shape:
        raise ValueError(
            f"ctx.fxy2.shape={ctx.fxy2.shape} "
            f"does not match profile {shape}"
        )

    if tuple(ctx.theta_bias_2d.shape) != shape:
        raise ValueError(
            f"ctx.theta_bias_2d.shape={ctx.theta_bias_2d.shape} "
            f"does not match profile {shape}"
        )

# ============================================================
# @title Dual-grid LC context wrapper
# ============================================================

def load_verified_profile(profile_path):
    profile_path = Path(profile_path)

    z = cp.load(profile_path, allow_pickle=True)
    try:
        mode = dict(
            path=profile_path,
            A=z["A"].astype(cp.complex64),
            theta=z["theta"].astype(cp.float32),
            I=z["I"].astype(cp.float32),
            beta=_npz_float(z, "beta"),
            P_mW=_npz_float(z, "P_mW"),
            b=_npz_float(z, "b"),
            bi=_npz_float(z, "bi"),
            theta_bc=_npz_float(z, "theta_bc"),
            beta_verify_diff=_npz_float(z, "beta_verify_diff"),
            beta_verify_rayleigh=_npz_float(z, "beta_verify_rayleigh"),
            beta_verify_tol=_npz_float(z, "beta_verify_tol"),
        )
    finally:
        z.close()

    mode["grid_mode"] = _npz_string_np(profile_path, "grid_mode", "UNKNOWN")
    mode["branch"] = _npz_string_np(profile_path, "branch", "UNKNOWN")

    return mode

def install_launch_profile_into_ctx(ctx, mode):
    """
    Prepare ctx for launching this saved eigensoliton.
    Full-grid launch only.
    """
    ctx.use_dual_grid = False

    for attr in ["ctx_theta", "dual_grid"]:
        if hasattr(ctx, attr):
            delattr(ctx, attr)

    ctx.b = float(mode["b"])
    ctx.bi = float(mode["bi"])
    ctx.theta_bc = float(mode["theta_bc"])

    theta = mode["theta"].astype(cp.float32, copy=True)
    theta[0, :] = cp.float32(ctx.theta_bc)
    theta[-1, :] = cp.float32(ctx.theta_bc)
    theta = enforce_theta_constraints(theta, ctx.theta_clamp)

    # z-stack initialized to the eigensoliton director everywhere
    ctx.theta_full = cp.repeat(theta[None, :, :], ctx.Nz, axis=0)

    if hasattr(ctx, "theta_prev_time"):
        ctx.theta_prev_time = ctx.theta_full.copy()

    return mode["A"].astype(cp.complex64, copy=True), theta

def launch_perturb_A(A, ctx, *, label="noise", eps=1e-5, seed=1):
    """
    Small perturbation to launch field, normalized back to same total power shape.
    """
    if eps == 0:
        return A.copy()

    rng = cp.random.RandomState(seed)

    if label.startswith("noise"):
        eta = rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape)
        eta = eta.astype(cp.complex64)
        eta /= cp.sqrt(cp.mean(cp.abs(eta)**2))
        Ap = A + eps * cp.sqrt(cp.mean(cp.abs(A)**2)) * eta

    elif label.startswith("phase"):
        phi = rng.standard_normal(A.shape).astype(cp.float32)
        phi /= cp.std(phi)
        Ap = A * cp.exp(1j * eps * phi)

    else:
        raise ValueError(f"Unknown launch perturbation label={label}")

    # preserve ∫|A|² dxdy
    p0 = cp.sum(cp.abs(A)**2) * ctx.dx * ctx.dy
    p1 = cp.sum(cp.abs(Ap)**2) * ctx.dx * ctx.dy
    Ap *= cp.sqrt(p0 / p1)

    return Ap.astype(cp.complex64, copy=False)

def transverse_metrics_from_I(I, ctx):
    x_um, y_um = xy_um_from_ctx(ctx)
    X = cp.asarray(x_um)[:, None]
    Y = cp.asarray(y_um)[None, :]

    W = cp.sum(I) + 1e-30
    x0 = cp.sum(I * X) / W
    y0 = cp.sum(I * Y) / W
    sx = cp.sqrt(cp.sum(I * (X - x0)**2) / W)
    sy = cp.sqrt(cp.sum(I * (Y - y0)**2) / W)

    return float(x0), float(y0), float(sx), float(sy), float(cp.max(I))

def rel_l2_cp(a, b):
    return float(cp.sqrt(cp.sum(cp.abs(a - b)**2) / (cp.sum(cp.abs(a)**2) + 1e-30)))


# ------------------------------------------------------------
# You must connect this to your actual z-runner
# ------------------------------------------------------------

def run_launch_zmarch(ctx, A_launch, *, runner_kwargs=None):
    runner_kwargs = {} if runner_kwargs is None else dict(runner_kwargs)

    old = {}
    for name in ["amp0", "amp", "amp_time_state"]:
        if hasattr(ctx, name):
            old[name] = getattr(ctx, name)

    old_save_full = getattr(ctx, "save_full_I_mid_store", None)

    try:
        Aset2 = A_launch.astype(cp.complex64, copy=True)

        print("Aset2 shape, dim ", Aset2.shape, Aset2.ndim)

        if Aset2.ndim == 2:
            Aset = cp.zeros((2, int(ctx.Nx), int(ctx.Ny)), dtype=cp.complex64)
            Aset[0, :, :] = Aset2
        elif Aset2.ndim == 3 and Aset2.shape[0] == 1:
            Aset = cp.zeros((2, int(ctx.Nx), int(ctx.Ny)), dtype=cp.complex64)
            Aset[0, :, :] = Aset2[0]
        else:
            Aset = Aset2

        print("Aset shape, dim ", Aset.shape, Aset.ndim)

        ctx.amp0 = Aset.copy()
        ctx.amp = Aset.copy()
        ctx.amp_time_state = Aset.copy()
        ctx.save_full_I_mid_store = True

        print("launch power amp0:", float(cp.sum(cp.abs(ctx.amp0)**2) * ctx.dx * ctx.dy))
        print("launch power amp :", float(cp.sum(cp.abs(ctx.amp)**2) * ctx.dx * ctx.dy))
        print("launch power ats :", float(cp.sum(cp.abs(ctx.amp_time_state)**2) * ctx.dx * ctx.dy))

        I_mid_store = run_unified_td_static(
            ctx,
            timedep=False,
            true_td=False,
            restart_theta=False,
            restart_static_from_bias=False,
            static_mode=runner_kwargs.pop("static_mode", "strict_relax"),
            store_I_dtype=runner_kwargs.pop("store_I_dtype", cp.float32),
            tqdm_static_z=runner_kwargs.pop("tqdm_static_z", True),
            report_runtime=runner_kwargs.pop("report_runtime", True),
            strict_residual_tol_max=runner_kwargs.pop("strict_residual_tol_max", 5e-2),
            strict_residual_tol_rms=runner_kwargs.pop("strict_residual_tol_rms", 1e-2),
            strict_max_outer_passes=runner_kwargs.pop("strict_max_outer_passes", 8),
            strict_verbose_every=runner_kwargs.pop("strict_verbose_every", 100),
            **runner_kwargs,
        )

        if isinstance(I_mid_store, dict):
            I_mid_store = I_mid_store["I_mid_store"]
        elif isinstance(I_mid_store, tuple):
            I_mid_store = I_mid_store[0]

        I_mid_store = cp.asarray(I_mid_store)

        print(
            "[launch returned]",
            I_mid_store.shape,
            I_mid_store.dtype,
            "Imax=", float(cp.max(I_mid_store)),
            "Isum0=", float(cp.sum(I_mid_store[0]) * ctx.dx * ctx.dy),
        )

        return I_mid_store

    finally:
        for name, val in old.items():
            setattr(ctx, name, val)
        if old_save_full is not None:
            ctx.save_full_I_mid_store = old_save_full


# ------------------------------------------------------------
# Main paired launch test
# ------------------------------------------------------------

def launch_stability_vs_z(
    ctx,
    profile_path,
    *,
    perturbations=None,
    runner_kwargs=None,
    save_dir=None,
):
    mode = load_verified_profile(profile_path)
    A_ref, theta_ref = install_launch_profile_into_ctx(ctx, mode)

    if perturbations is None:
        perturbations = [
            dict(label="unperturbed", eps=0.0, seed=1),
            dict(label="noise_1e-6", eps=1e-6, seed=1),
            dict(label="noise_3e-6", eps=3e-6, seed=1),
            dict(label="noise_1e-5", eps=1e-5, seed=1),
        ]

    save_dir = Path(save_dir or (Path(profile_path).parent / "launch_stability"))
    save_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    rows = []

    for p in perturbations:
        label = p["label"]
        eps = float(p.get("eps", 0.0))

        print(f"\n[launch] {label}, eps={eps:g}")

        # Reset theta stack identically for each launch
        A_ref, theta_ref = install_launch_profile_into_ctx(ctx, mode)
        A_launch = launch_perturb_A(A_ref, ctx, **p)

        I_mid = run_launch_zmarch(
            ctx,
            A_launch,
            runner_kwargs=runner_kwargs,
        )

        runs[label] = I_mid

        # per-z scalar diagnostics
        for k in range(ctx.Nz):
            I = cp.asarray(I_mid[k])
            x0, y0, sx, sy, Imax = transverse_metrics_from_I(I, ctx)

            rows.append(dict(
                label=label,
                eps=eps,
                k=k,
                z_um=float(k * ctx.dz),
                x0_um=x0,
                y0_um=y0,
                sx_um=sx,
                sy_um=sy,
                Imax=Imax,
            ))

        cp.savez(
            save_dir / f"{Path(profile_path).stem}__launch__{label}.npz",
            I_mid=I_mid,
            P_mW=cp.array(mode["P_mW"]),
            beta=cp.array(mode["beta"]),
            eps=cp.array(eps),
        )

        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    # paired separation vs unperturbed
    I0 = runs["unperturbed"]
    pair_rows = []

    for p in perturbations:
        label = p["label"]
        if label == "unperturbed":
            continue

        eps = float(p.get("eps", 0.0))
        Ip = runs[label]

        for k in range(ctx.Nz):
            dI = rel_l2_cp(I0[k], Ip[k])
            pair_rows.append(dict(
                label=label,
                eps=eps,
                k=k,
                z_um=float(k * ctx.dz),
                paired_rel_I=dI,
                paired_gain_I=dI / max(eps, 1e-30),
            ))

    df = pd.DataFrame(rows)
    df_pair = pd.DataFrame(pair_rows)

    df.to_csv(save_dir / f"{Path(profile_path).stem}__launch_metrics.csv", index=False)
    df_pair.to_csv(save_dir / f"{Path(profile_path).stem}__launch_paired.csv", index=False)

    return df, df_pair

def match_shape_theta(theta, target_shape):
    if theta.shape == target_shape:
        return theta

    zx = target_shape[0] / theta.shape[0]
    zy = target_shape[1] / theta.shape[1]
    return cndi.zoom(theta.astype(cp.float32, copy=False), (zx, zy), order=3)[:target_shape[0], :target_shape[1]]

def compare_profile_sets(dir_A, dir_B, *, prefix="lc_eigensoliton"):
    dir_A = Path(dir_A)
    dir_B = Path(dir_B)

    files_A = sorted(dir_A.glob(f"{prefix}_P_*mW.npz"))
    files_B = sorted(dir_B.glob(f"{prefix}_P_*mW.npz"))

    rows = []

    for fa in files_A:
        Pa = _P_from_name(fa)
        fb = min(files_B, key=lambda f: abs(_P_from_name(f) - Pa))
        Pb = _P_from_name(fb)

        a = cp.load(fa)
        b = cp.load(fb)

        IA = a["I"].astype(cp.float32)
        IB = b["I"].astype(cp.float32)

        thA = a["theta"].astype(cp.float32)
        thB = b["theta"].astype(cp.float32)

        if thA.shape != thB.shape:
            thA = match_shape_theta(thA, thB.shape)

        # phase-insensitive field comparison via intensity
        I_rel = float(cp.linalg.norm((IA - IB).ravel()) /
                      (cp.linalg.norm(IB.ravel()) + 1e-30))

        th_rel = float(cp.linalg.norm((thA - thB).ravel()) /
                       (cp.linalg.norm(thB.ravel()) + 1e-30))

        rows.append({
            "P_DG": Pa,
            "P_full": Pb,
            "beta_DG": float(a["beta"]),
            "beta_full": float(b["beta"]),
            "beta_diff": float(a["beta"] - b["beta"]),
            "beta_rel": abs(float(a["beta"] - b["beta"])) / max(1.0, abs(float(b["beta"]))),
            "I_rel_L2": I_rel,
            "theta_rel_L2": th_rel,
            "Imax_DG": float(cp.max(IA)),
            "Imax_full": float(cp.max(IB)),
            "theta_max_DG": float(cp.max(thA)),
            "theta_max_full": float(cp.max(thB)),
        })

    return pd.DataFrame(rows)


# edit these to your actual profile folders

def axial_peak_from_Ixz(Ixz):
    """
    Ixz shape expected (Nt_or_1, Nz, Nx) or (Nz, Nx).
    Returns peak over x at each z.
    """
    A = cp.asarray(Ixz)
    if A.ndim == 3:
        A = A[-1]          # last saved TD frame
    return cp.asnumpy(cp.max(A, axis=-1))

def breakup_z_from_trace(z_um, Ipeak, *, frac=0.80, n_ref=200, min_z_um=100.0):
    """
    Breakup when axial peak falls below frac * upstream median.
    """
    z_um = np.asarray(z_um)
    Ipeak = np.asarray(Ipeak)

    n_ref = min(int(n_ref), len(Ipeak))
    I0 = np.median(Ipeak[:n_ref])
    thresh = frac * I0

    mask = (z_um >= min_z_um) & (Ipeak < thresh)

    if not np.any(mask):
        return np.nan, I0, thresh

    idx = np.argmax(mask)
    return float(z_um[idx]), float(I0), float(thresh)

def load_profile_for_power(profile_dir, prefix, P):
    """
    Finds nearest saved profile file for requested P.
    """
    files = sorted(profile_dir.glob(f"{prefix}_P_*mW.npz"))
    if not files:
        raise FileNotFoundError(f"No profiles found in {profile_dir}")

    Ps = []
    for f in files:
        s = f.stem.split("_P_")[-1].replace("mW", "")
        Ps.append(float(s))

    Ps = np.array(Ps)
    j = int(np.argmin(np.abs(Ps - float(P))))
    return files[j], float(Ps[j])

def td_breakup_sweep_from_profiles(
    ctx,
    prdata,
    *,
    run_dir,
    powers,
    checkpoint_prefix="lc_eigensoliton",
    dt=2.5e-4,
    tsteps=40,
    breakup_frac=0.80,
    n_ref=200,
    min_z_um=100.0,
    out_csv_name="td_breakup_sweep.csv",
    store_I_dtype=cp.float32,
    verbose=True,
):
    run_dir = Path(run_dir)
    profile_dir = run_dir / f"{checkpoint_prefix}_profiles"
    out_csv = run_dir / out_csv_name

    rows = []

    old_save_full = getattr(ctx, "save_full_I_mid_store", False)

    ctx.save_full_I_mid_store = True

    time_steps = cp.full((int(tsteps),), cp.float32(dt), dtype=cp.float32)
    z_um = cp.asnumpy(cp.arange(ctx.Nz, dtype=cp.float32) * cp.float32(ctx.dz))

    for P_req in powers:
        profile_path, P_prof = load_profile_for_power(profile_dir, checkpoint_prefix, P_req)

        if verbose:
            print(f"\n=== TD breakup test P_req={P_req:.6g} mW, profile P={P_prof:.6g} ===")
            print(profile_path)

        prof = cp.load(profile_path)
        A0 = prof["A"].astype(cp.complex64, copy=False)
        theta0 = prof["theta"].astype(cp.float32, copy=False)

        set_lc_power_from_prdata(ctx, prdata, P_prof)

        ctx.amp0[...] = 0
        ctx.amp0[0] = A0

        ctx.theta_full[...] = theta0[None, :, :]
        if ctx.theta_prev_time is not None:
            ctx.theta_prev_time[...] = ctx.theta_full

        I_mid_td = run_unified_td_static(
            ctx,
            timedep=True,
            time_steps=time_steps,
            tsteps=int(tsteps),
            t_stride=1,
            restart_theta=False,
            true_td=True,
            store_I_dtype=store_I_dtype,
            tqdm_timedep=True,
            tqdm_static_z=False,
            report_runtime=True,
            use_quasiglobal_td=False,
            freeze_theta=False,
        )

        # axial peak from stored I_mid stack
        Ipeak = cp.asnumpy(cp.max(I_mid_td.astype(cp.float32), axis=(1, 2)))
        z_break, I0, Ith = breakup_z_from_trace(
            z_um,
            Ipeak,
            frac=breakup_frac,
            n_ref=n_ref,
            min_z_um=min_z_um,
        )

        row = {
            "P_req_mW": float(P_req),
            "P_profile_mW": float(P_prof),
            "theta_bc": float(getattr(ctx, "theta_bc", prdata.get("theta_bc", 0.0))),
            "b": float(ctx.b),
            "bi": float(ctx.bi),
            "dt": float(dt),
            "tsteps": int(tsteps),
            "breakup_frac": float(breakup_frac),
            "I0_ref": float(I0),
            "I_threshold": float(Ith),
            "z_break_um": z_break,
            "Ipeak_start": float(Ipeak[0]),
            "Ipeak_end": float(Ipeak[-1]),
            "Ipeak_min": float(np.min(Ipeak)),
            "Ipeak_max": float(np.max(Ipeak)),
            "profile_path": str(profile_path),
        }

        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        if verbose:
            print(
                f"[breakup] P={P_prof:.4g} mW  "
                f"z_break={z_break} um  I0={I0:.4e}  threshold={Ith:.4e}"
            )

        del prof, A0, theta0, I_mid_td
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    ctx.save_full_I_mid_store = old_save_full

    return pd.DataFrame(rows)


# -----------------------------
# Example run
# -----------------------------

def find_profile_near_power(profile_dir, target_mW):
    profile_dir = Path(profile_dir)
    files = sorted(profile_dir.glob("*.npz"))

    best = None
    best_err = np.inf

    for f in files:
        m = re.search(r"P_?([0-9.]+)mW", f.name)
        if m is None:
            continue

        P = float(m.group(1))
        err = abs(P - target_mW)

        if err < best_err:
            best = f
            best_err = err

    if best is None:
        raise FileNotFoundError(f"No profile files found in {profile_dir}")

    return best

def core_error(theta_full, theta_ref):
  return np.sqrt(np.mean((theta_ref - theta_full)[:,:,ctx.Ny//2-128:ctx.Ny//2+127]**2))

def advance_theta_perturbation_about_reference_prepared(
    delta_n,
    *,
    theta_ref,
    I_ref,
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
    Linear perturbation step about a frozen reference state theta_ref, gamma_z = 0.

    Evolves delta from:
        delta_t = (1/mobility) * [ L_xy delta + q_ref * delta + f_ref ]

    where
        q_ref = 2 (b + bi I_ref) cos(2 theta_ref)
        f_ref = L_xy(theta_ref) + (b + bi I_ref) sin(2 theta_ref)

    If theta_ref is an exact frozen-optics steady state, f_ref ~ 0.
    """
    delta_n = delta_n.astype(cp.float32, copy=False)
    theta_ref = theta_ref.astype(cp.float32, copy=False)
    I_ref = I_ref.astype(cp.float32, copy=False)

    lap_ref = _laplacian_dirichletx_periody(
        theta_ref, float(du), float(dv)
    ).astype(cp.float32, copy=False)

    alpha_ref = cp.float32(b) + cp.float32(bi) * I_ref
    q_ref = cp.float32(2.0) * alpha_ref * cp.cos(cp.float32(2.0) * theta_ref)
    f_ref = lap_ref + alpha_ref * cp.sin(cp.float32(2.0) * theta_ref)

    dt_over_m = cp.float32(float(dt) / float(mobility))

    rhs = delta_n + dt_over_m * (q_ref * delta_n + f_ref)
    rhs[0, :] = 0.0
    rhs[-1, :] = 0.0

    delta_np1 = _cn_solve_dirichletx_periody(rhs, off=off, diag=diag)

    # clamp should apply to the reconstructed theta, not delta itself
    theta_np1 = theta_ref + delta_np1
    theta_np1 = enforce_theta_constraints(theta_np1, clamp)
    delta_np1 = theta_np1 - theta_ref

    return delta_np1

def advance_theta_perturbation_about_reference_prepared_zcoupled(
    delta_n,
    *,
    theta_ref,
    I_ref,
    delta_prev_n,
    delta_next_n,
    dt,
    b,
    bi,
    gamma_z,
    mobility,
    du,
    dv,
    off,
    diag,
    inv_dz2,
    clamp=None,
):
    """
    Linear perturbation step about a frozen reference state theta_ref, with z coupling.

    Evolves delta from:
        delta_t = (1/mobility) * [ L_xy delta + gamma_z delta_zz + q_ref delta + f_ref ]

    where
        q_ref = 2 (b + bi I_ref) cos(2 theta_ref)
        f_ref = L_xy(theta_ref) + (b + bi I_ref) sin(2 theta_ref)

    If theta_ref is an exact frozen-optics steady state, f_ref ~ 0.
    """
    delta_n = delta_n.astype(cp.float32, copy=False)
    theta_ref = theta_ref.astype(cp.float32, copy=False)
    I_ref = I_ref.astype(cp.float32, copy=False)
    dp = delta_prev_n.astype(cp.float32, copy=False)
    dn = delta_next_n.astype(cp.float32, copy=False)

    lap_ref = _laplacian_dirichletx_periody(
        theta_ref, float(du), float(dv)
    ).astype(cp.float32, copy=False)

    alpha_ref = cp.float32(b) + cp.float32(bi) * I_ref
    q_ref = cp.float32(2.0) * alpha_ref * cp.cos(cp.float32(2.0) * theta_ref)
    f_ref = lap_ref + alpha_ref * cp.sin(cp.float32(2.0) * theta_ref)

    gam = cp.float32(gamma_z) * inv_dz2
    dt_over_m = cp.float32(float(dt) / float(mobility))

    rhs = delta_n + dt_over_m * (q_ref * delta_n + f_ref + gam * (dp + dn))
    rhs[0, :] = 0.0
    rhs[-1, :] = 0.0

    diag_eff = diag + cp.float32(2.0) * dt_over_m * gam
    delta_np1 = _cn_solve_dirichletx_periody(rhs, off=off, diag=diag_eff)

    theta_np1 = theta_ref + delta_np1
    theta_np1 = enforce_theta_constraints(theta_np1, clamp)
    delta_np1 = theta_np1 - theta_ref

    return delta_np1

def advance_theta_perturbation_about_reference_step(
    ctx,
    delta_n,
    theta_ref_k,
    I_ref_k,
    delta_prev_n,
    delta_next_n,
    *,
    dt_j,
    off,
    diag,
):
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))

    if gamma_z == 0.0:
        return advance_theta_perturbation_about_reference_prepared(
            delta_n,
            theta_ref=theta_ref_k,
            I_ref=I_ref_k,
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

    return advance_theta_perturbation_about_reference_prepared_zcoupled(
        delta_n,
        theta_ref=theta_ref_k,
        I_ref=I_ref_k,
        delta_prev_n=delta_prev_n,
        delta_next_n=delta_next_n,
        dt=dt_j,
        b=ctx.b,
        bi=ctx.bi,
        gamma_z=gamma_z,
        mobility=ctx.mobility,
        du=ctx.du,
        dv=ctx.dv,
        off=off,
        diag=diag,
        inv_dz2=_get_inv_dz2(ctx),
        clamp=ctx.theta_clamp,
    )

def run_linearized_stability_about_reference(
    ctx,
    theta_ref_full,
    I_ref_store,
    *,
    time_steps,
    delta0=None,
    t_stride=1,
    save_slices_fn=None,
    store_delta_history=False,
    tqdm_enable=True,
):
    """
    Evolve small perturbations delta about a frozen reference state theta_ref_full
    under frozen self-consistent optics I_ref_store.

    Parameters
    ----------
    theta_ref_full : cp.ndarray, shape (Nz, Nx, Ny)
        Frozen reference theta field.
    I_ref_store : cp.ndarray, shape (Nz, Nx, Ny)
        Frozen midpoint intensity store consistent with theta_ref_full.
    time_steps : sequence of float
        Time steps to evolve.
    delta0 : cp.ndarray or None
        Initial perturbation, shape (Nz, Nx, Ny). If None, starts from zero.
    store_delta_history : bool
        If True, store a snapshot every t_stride steps.

    Returns
    -------
    delta_full : cp.ndarray
        Final perturbation field.
    theta_full_lin : cp.ndarray
        Final reconstructed theta field = theta_ref_full + delta_full.
    growth_hist : list[float]
        RMS perturbation norm at each time step.
    delta_hist : cp.ndarray or None
        Stored perturbation snapshots if requested.
    """
    Nz, Nx, Ny = theta_ref_full.shape

    if delta0 is None:
        delta_full = cp.zeros_like(theta_ref_full, dtype=cp.float32)
    else:
        delta_full = delta0.astype(cp.float32, copy=True)

    delta_prev_time = cp.empty_like(delta_full)
    theta_full_lin = cp.empty_like(theta_ref_full, dtype=cp.float32)

    growth_hist = []

    nframes = (len(time_steps) + t_stride - 1) // t_stride if store_delta_history else 0
    delta_hist = cp.empty((nframes, Nz, Nx, Ny), dtype=cp.float16) if store_delta_history else None

    jt_iter = tqdm(range(len(time_steps)), desc="linearized stability", dynamic_ncols=True) if tqdm_enable else range(len(time_steps))

    for jt in jt_iter:
        dt_j = float(time_steps[jt])

        a_ie, off, diag, lam_y = prepare_ie_ky_operator(
            dt=dt_j,
            mobility=float(ctx.mobility),
            du=float(ctx.du),
            dv=float(ctx.dv),
            Ny=ctx.Ny,
        )

        delta_prev_time[...] = delta_full

        for k in range(Nz):
            dp = delta_prev_time[k - 1] if k > 0 else delta_prev_time[k]
            dn = delta_prev_time[k + 1] if (k + 1) < Nz else delta_prev_time[k]

            delta_full[k] = advance_theta_perturbation_about_reference_step(
                ctx,
                delta_prev_time[k],
                theta_ref_full[k],
                I_ref_store[k].astype(cp.float32, copy=False),
                dp,
                dn,
                dt_j=dt_j,
                off=off,
                diag=diag,
            )

            theta_full_lin[k] = enforce_theta_constraints(
                theta_ref_full[k] + delta_full[k],
                ctx.theta_clamp,
            )

        growth = float(cp.sqrt(cp.mean(delta_full * delta_full)))
        growth_hist.append(growth)

        try:
            jt_iter.set_postfix(delta_rms=f"{growth:.2e}")
        except Exception:
            pass

        if store_delta_history and ((jt % t_stride) == 0):
            slot = jt // t_stride
            delta_hist[slot] = delta_full.astype(cp.float16, copy=False)

        if save_slices_fn is not None and ((jt % t_stride) == 0):
            slot = jt // t_stride
            for k in range(Nz):
                save_slices_fn(slot, k, I_ref_store[k], theta_full_lin[k])

    return delta_full, theta_full_lin, growth_hist, delta_hist

# set save directory

