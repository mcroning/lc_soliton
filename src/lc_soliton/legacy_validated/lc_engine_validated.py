"""
lc_engine_validated.py

Thin orchestration layer for the LC soliton validated runtime.

The trusted GPU physics kernels live in:
    lc_soliton.validated_core.runner_core

This module owns:
    - backend selection
    - parameter/context construction
    - lightweight scalar/movie storage
    - mode dispatch and progress reporting

It deliberately avoids re-implementing the trusted static/TD kernels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import cupy as _cupy  # type: ignore
    import cupyx.scipy.fft as _cupy_fft  # type: ignore
    import cupyx.scipy.fft as _spfft  # type: ignore
    _HAS_CUPY = True
except Exception:  # pragma: no cover
    _cupy = None
    _cupy_fft = None
    _spfft = None
    _HAS_CUPY = False

from ..validated_core.runner_core import (
    apply_nonlinear_phase_inplace,
    choose_optics_substeps as core_choose_optics_substeps,
    get_h_for_dz as core_get_h_for_dz,
    hop_linear as core_hop_linear,
    hop_linear_inplace,
    intens_into,
    prepare_cn_ky_operator,
    prepare_ie_ky_operator,
    residual_stats_2d,
    lc_residual64,
    strict_static_relax_slice_selfconsistent,
    td_get_slice_mid_intensity,
    td_update_theta_from_midintensity,
)

from ..core.backend import (
    _HAS_CUPY,
    _cupy,
    asnumpy,
    free_backend_memory,
    get_backend,
    is_cupy_array,
)

from ..core.bias import (
    compute_neff,
    lc_b_from_voltage,
    voltage_from_lc_b,
    theta0_from_b_zero_bc,
    b_from_theta0_zero_bc,
    theta_bias_1d_exact_zero_bc,
    theta_bias_1d_relax_bc,
    theta_bias_2d_from_params,
)

from ..core.context import (
    LCParams,
    LCContext,
    DualGrid,
    LegacyPlans,
    apply_legacy_context_aliases,
    intensity,
    make_context,
    prolong_repeat,
    restrict_block_mean,
)

from ..core.storage import LightStore


# -----------------------------------------------------------------------------
# Local package-smoke fallback numerics
# -----------------------------------------------------------------------------


def _fft2(xp, a, axes=(-2, -1)):
    return _cupy_fft.fft2(a, axes=axes) if (_HAS_CUPY and xp is _cupy) else np.fft.fft2(a, axes=axes)


def _ifft2(xp, a, axes=(-2, -1)):
    return _cupy_fft.ifft2(a, axes=axes) if (_HAS_CUPY and xp is _cupy) else np.fft.ifft2(a, axes=axes)


def _compute_h_local(ctx: LCContext, dz_um: float):
    key = round(float(dz_um), 12)
    if key in ctx._h_cache:
        return ctx._h_cache[key]
    xp = ctx.xp
    lm = float(ctx.p.wavelength_um)
    refin = float(ctx.refin)
    arg = 1.0 - (lm / refin) ** 2 * ctx.fxy2
    h = xp.where(arg > 0, xp.exp(2j * xp.pi * refin * dz_um / lm * xp.sqrt(xp.maximum(arg, 0))), 0)
    ctx._h_cache[key] = h.astype(xp.complex64)
    return ctx._h_cache[key]


def _hop_linear_local(ctx: LCContext, amp, h):
    xp = ctx.xp
    A = _fft2(xp, amp, axes=(-2, -1))
    A *= h
    out = _ifft2(xp, A, axes=(-2, -1)).astype(xp.complex64)
    if ctx.windowxy is not None:
        out *= ctx.windowxy[None, :, :]
    return out


def _lc_dn_local(theta, ctx: LCContext):
    return (compute_neff(theta, ne=ctx.p.ne, no=ctx.p.no, xp=ctx.xp) - ctx.refin).astype(ctx.xp.float32)


def _propagate_slice_local(ctx: LCContext, amp, theta, *, nsub: int, dz_sub: float, h_sub):
    xp = ctx.xp
    for _ in range(int(nsub)):
        phase = xp.exp((1j * xp.float32(ctx.kout * dz_sub)) * _lc_dn_local(theta, ctx)).astype(xp.complex64)
        amp = (amp * phase[None, :, :]).astype(xp.complex64)
        amp = _hop_linear_local(ctx, amp, h_sub)
    return amp


def _laplacian_local(theta, du: float, dv: float, xp):
    th = theta.astype(xp.float32, copy=False)
    lap = xp.zeros_like(th)
    yp = xp.roll(th, -1, axis=1)
    ym = xp.roll(th, 1, axis=1)
    lap[1:-1, :] = ((th[2:, :] - 2 * th[1:-1, :] + th[:-2, :]) / (du * du) + (yp[1:-1, :] - 2 * th[1:-1, :] + ym[1:-1, :]) / (dv * dv))
    return lap


def _residual_local(theta, I, ctx: LCContext):
    p = ctx.p
    xp = ctx.xp
    R = (_laplacian_local(theta, ctx.du_lc, ctx.dv_lc, xp) + (p.b + p.bi * I) * xp.sin(2 * theta)) / p.mobility
    R[0, :] = 0
    R[-1, :] = 0
    return R.astype(xp.float32)


def _simple_static_smoke_slice(ctx: LCContext, amp, theta_seed, *, nsub: int, dz_sub: float, h_sub):
    """Small CPU-safe smoke path. Not the trusted physics branch."""
    xp = ctx.xp
    p = ctx.p
    I_b = intensity(amp, p.coherent, xp=xp)
    theta = theta_seed.astype(xp.float32, copy=True)
    theta[0, :] = xp.float32(p.theta_bc)
    theta[-1, :] = xp.float32(p.theta_bc)

    # Use a very conservative explicit relaxation. This exists only so CPU tests
    # can exercise IO/request plumbing; trusted static uses validated_core on GPU.
    dtau = min(float(p.dtau_static), 1e-5)
    for _ in range(max(1, min(int(p.static_max_steps), 20))):
        R = _residual_local(theta, I_b, ctx)
        theta[1:-1, :] += xp.float32(dtau) * R[1:-1, :]
        theta = xp.clip(theta, p.theta_clamp_min, p.theta_clamp_max).astype(xp.float32)
        theta[0, :] = xp.float32(p.theta_bc)
        theta[-1, :] = xp.float32(p.theta_bc)

    amp_out = _propagate_slice_local(ctx, amp.copy(), theta, nsub=nsub, dz_sub=dz_sub, h_sub=h_sub)
    I_a = intensity(amp_out, p.coherent, xp=xp)
    I_mid = 0.5 * (I_b + I_a)
    R = _residual_local(theta, I_mid, ctx)
    Rint = R[1:-1, :]
    info = dict(
        rrms=float(asnumpy(xp.sqrt(xp.mean(Rint * Rint)))),
        rmax=float(asnumpy(xp.max(xp.abs(Rint)))),
        converged=True,
        method="package_smoke",
    )
    return theta, I_mid, amp_out, info


# -----------------------------------------------------------------------------
# Runner utilities
# -----------------------------------------------------------------------------


def normalize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(info or {})
    if "rrms" not in out and "rms_interior" in out:
        out["rrms"] = out["rms_interior"]
    if "rmax" not in out and "max_interior" in out:
        out["rmax"] = out["max_interior"]
    if "converged" not in out:
        out["converged"] = bool(out.get("accepted", False) or out.get("early_accepted", False) or out.get("n_selfcons_passes", 0) > 0)
    return out


def _prepare_legacy_plans(ctx: LCContext) -> LegacyPlans:
    if not (_HAS_CUPY and ctx.xp is _cupy):
        raise RuntimeError("Validated legacy kernels require CuPy/GPU.")
    if _spfft is None:
        raise RuntimeError("cupyx.scipy.fft is unavailable.")

    Ahat = ctx.xp.empty_like(ctx.amp0)
    return LegacyPlans(
        Ahat=Ahat,
        plan_f=_spfft.get_fft_plan(Ahat, axes=(-2, -1)),
        plan_i=_spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C"),
    )


def _prepare_substeps(ctx: LCContext, params: LCParams, *, use_core: bool):
    if use_core:
        nsub, dz_sub, phi = core_choose_optics_substeps(
            float(ctx.dz),
            kout=float(ctx.kout),
            dn_max_est=float(params.dn_max_est),
            dz_opt_max_phi=float(params.dz_opt_max_phi),
            max_substeps=int(params.max_substeps),
        )
        return int(nsub), float(dz_sub), float(phi), core_get_h_for_dz(ctx, dz_sub), core_get_h_for_dz(ctx, 0.5 * dz_sub)

    phi = abs(ctx.kout * ctx.dz * params.dn_max_est)
    nsub = max(1, min(int(params.max_substeps), int(math.ceil(phi / max(params.dz_opt_max_phi, 1e-12)))))
    dz_sub = ctx.dz / nsub
    return int(nsub), float(dz_sub), float(phi), _compute_h_local(ctx, dz_sub), _compute_h_local(ctx, 0.5 * dz_sub)


def _make_scalar_info_from_residual(theta, I_mid, ctx: LCContext) -> Dict[str, Any]:
    R = lc_residual64(theta, I_mid, b=ctx.b, bi=ctx.bi, du=ctx.du, dv=ctx.dv, mobility=ctx.mobility)
    stats = residual_stats_2d(R)
    return normalize_info(dict(rrms=stats["rms_interior"], rmax=stats["max_interior"], converged=True))


# -----------------------------------------------------------------------------
# Mode implementations
# -----------------------------------------------------------------------------


def _run_static(
    *,
    ctx: LCContext,
    params: LCParams,
    store: LightStore,
    progress: Optional[Callable[[str], None]],
    should_stop: Optional[Callable[[], bool]],
    use_legacy_static: bool,
    save_full: bool,
) -> bool:
    xp = ctx.xp
    nsub, dz_sub, phi, h_sub, h_half = _prepare_substeps(ctx, params, use_core=use_legacy_static)
    plans = _prepare_legacy_plans(ctx) if use_legacy_static else LegacyPlans()

    if use_legacy_static:
        plans.sS, plans.offS, plans.diagS, plans.lamS = prepare_cn_ky_operator(
            dt=float(params.dtau_static),
            mobility=float(ctx.mobility),
            du=float(ctx.du),
            dv=float(ctx.dv),
            Ny=int(ctx.Ny),
        )

    amp = core_hop_linear(ctx.amp0.copy(), h_half, ctx.windowxy) if use_legacy_static else _hop_linear_local(ctx, ctx.amp0.copy(), h_half)
    stopped = False

    for k in range(ctx.Nz):
        if should_stop is not None and should_stop():
            stopped = True
            if progress:
                progress("run stopped by user")
            break

        theta_seed = ctx.theta_full[k - 1] if k > 0 else ctx.theta_bias_2d.copy()

        if use_legacy_static:
            tp = ctx.theta_full[k - 1] if k > 0 else ctx.theta_full[k]
            tn = ctx.theta_full[k + 1] if (k + 1) < ctx.Nz else ctx.theta_full[k]
            theta, I_mid, amp, info = strict_static_relax_slice_selfconsistent(
                amp,
                theta_seed,
                tp,
                tn,
                ctx,
                dz_sub=float(dz_sub),
                Nsub=int(nsub),
                h_sub=h_sub,
                Ahat=plans.Ahat,
                plan_f=plans.plan_f,
                plan_i=plans.plan_i,
                sS=plans.sS,
                offS=plans.offS,
                diagS=plans.diagS,
                lamS=plans.lamS,
                use_linear_seed=True,
                early_accept_linear_seed=True,
                residual_tol_max=float(params.static_tol_max),
                residual_tol_rms=float(params.static_tol_rms),
                max_outer_passes=8,
                max_selfcons_passes=int(params.static_selfcons_passes),
                selfcons_tol_theta=1e-4,
                selfcons_tol_I=1e-4,
                verbose=False,
            )
            info = normalize_info(info)
        else:
            theta, I_mid, amp, info = _simple_static_smoke_slice(ctx, amp, theta_seed, nsub=nsub, dz_sub=dz_sub, h_sub=h_sub)

        ctx.theta_full[k] = theta
        store.save(0, k, I_mid, theta, info)

        if progress and (k == 0 or (k + 1) % max(1, ctx.Nz // 20) == 0 or k == ctx.Nz - 1):
            progress(f"static z {k+1}/{ctx.Nz}  Imax={float(asnumpy(xp.max(I_mid))):.3e}  Rrms={float(info.get('rrms', np.nan)):.3e}")

    return stopped


def _run_td_predictor(
    *,
    ctx: LCContext,
    params: LCParams,
    store: LightStore,
    Nt: int,
    dt: float,
    t_stride: int,
    progress: Optional[Callable[[str], None]],
    should_stop: Optional[Callable[[], bool]],
) -> bool:
    if not (_HAS_CUPY and ctx.xp is _cupy):
        raise RuntimeError("Validated TD predictor currently requires CuPy/GPU.")

    xp = ctx.xp
    nsub, dz_sub, phi, h_sub, h_half = _prepare_substeps(ctx, params, use_core=True)
    plans = _prepare_legacy_plans(ctx)

    # Buffers used by validated_core TD predictor/corrector path.
    ctx._amp_in_buf = xp.empty_like(ctx.amp0)
    ctx._amp_pred_buf = xp.empty_like(ctx.amp0)
    ctx._I_mid_pred_buf = xp.empty((ctx.Nx, ctx.Ny), dtype=xp.float32)

    stopped = False
    Nt_eff = int(Nt)

    for jt in range(Nt_eff):
        if should_stop is not None and should_stop():
            stopped = True
            if progress:
                progress("run stopped by user")
            break

        a_ie, offTD, diagTD, lam_yTD = prepare_ie_ky_operator(
            dt=float(dt),
            mobility=float(ctx.mobility),
            du=float(ctx.du),
            dv=float(ctx.dv),
            Ny=int(ctx.Ny),
        )

        theta_ref = ctx.theta_full.copy()
        amp = core_hop_linear(ctx.amp0.copy(), h_half, ctx.windowxy)
        slot = jt // max(1, int(t_stride))
        last_Imax = np.nan
        last_rrms = np.nan

        for k in range(ctx.Nz):
            if should_stop is not None and should_stop():
                stopped = True
                if progress:
                    progress("run stopped by user")
                break

            theta_old = theta_ref[k]
            tp = theta_ref[k - 1] if k > 0 else theta_old
            tn = theta_ref[k + 1] if k + 1 < ctx.Nz else theta_old

            I_b = xp.empty((ctx.Nx, ctx.Ny), dtype=xp.float32)
            I_a = xp.empty((ctx.Nx, ctx.Ny), dtype=xp.float32)
            I_mid = xp.empty((ctx.Nx, ctx.Ny), dtype=xp.float32)
            intens_into(I_b, amp, coh=ctx.coh)

            # This advances amp through the slice using theta_old and fills I_mid.
            amp = td_get_slice_mid_intensity(
                ctx,
                amp,
                theta_old,
                I_b,
                I_a,
                I_mid,
                Nsub=int(nsub),
                dz_sub=float(dz_sub),
                h_sub=h_sub,
                Ahat=plans.Ahat,
                plan_f=plans.plan_f,
                plan_i=plans.plan_i,
                frozen_I_mid_k=None,
            )

            theta = td_update_theta_from_midintensity(
                ctx,
                theta_old,
                tp,
                tn,
                I_mid,
                dt_j=float(dt),
                true_td=True,
                a_ie=a_ie,
                s=None,
                off=offTD,
                diag=diagTD,
                lam_y=lam_yTD,
                Nsub=int(nsub),
                dz_sub=float(dz_sub),
                h_sub=h_sub,
                amp_for_pred=None,
                I_b=I_b,
                I_a=I_a,
                Ahat=plans.Ahat,
                plan_f=plans.plan_f,
                plan_i=plans.plan_i,
            )

            ctx.theta_full[k] = theta
            info = _make_scalar_info_from_residual(theta, I_mid, ctx)

            if (jt % max(1, int(t_stride))) == 0:
                store.save(slot, k, I_mid, theta, info)

            last_Imax = float(asnumpy(xp.max(I_mid)))
            last_rrms = float(info.get("rrms", np.nan))

        if stopped:
            break

        if progress:
            progress(f"time step {jt+1}/{Nt_eff}  mode=td_predictor_only  t={(jt+1)*dt:.4g}  Imax={last_Imax:.3e}  Rrms={last_rrms:.3e}")
        free_backend_memory(xp)

    return stopped


def _run_dg_td_smoke(
    *,
    ctx: LCContext,
    dg: DualGrid,
    params: LCParams,
    store: LightStore,
    Nt: int,
    dt: float,
    t_stride: int,
    progress: Optional[Callable[[str], None]],
    should_stop: Optional[Callable[[], bool]],
) -> bool:
    """Temporary DG smoke path. Full legacy DG requires the original DG context."""
    xp = ctx.xp
    nsub, dz_sub, phi, h_sub, h_half = _prepare_substeps(ctx, params, use_core=False)
    theta_c_stack = xp.stack([dg.theta_bias_c.copy() for _ in range(ctx.Nz)], axis=0)
    stopped = False

    for jt in range(int(Nt)):
        if should_stop is not None and should_stop():
            stopped = True
            break

        amp = _hop_linear_local(ctx, ctx.amp0.copy(), h_half)
        slot = jt // max(1, int(t_stride))
        last_Imax = np.nan
        last_rrms = np.nan

        for k in range(ctx.Nz):
            theta_f = prolong_repeat(theta_c_stack[k], dg.factor, target_shape=(ctx.Nx, ctx.Ny), xp=xp)
            I_b = intensity(amp, params.coherent, xp=xp)
            amp = _propagate_slice_local(ctx, amp, theta_f, nsub=nsub, dz_sub=dz_sub, h_sub=h_sub)
            I_a = intensity(amp, params.coherent, xp=xp)
            I_mid_f = 0.5 * (I_b + I_a)
            I_mid_c = restrict_block_mean(I_mid_f, dg.factor, xp=xp)

            # Smoke update: keep the coarse biased branch and add a small optical response.
            theta_c = theta_c_stack[k] + xp.float32(dt * params.bi / params.mobility) * I_mid_c * xp.sin(2.0 * theta_c_stack[k])
            theta_c = xp.clip(theta_c, params.theta_clamp_min, params.theta_clamp_max).astype(xp.float32)
            theta_c[0, :] = xp.float32(params.theta_bc)
            theta_c[-1, :] = xp.float32(params.theta_bc)
            theta_c_stack[k] = theta_c
            theta_f = prolong_repeat(theta_c, dg.factor, target_shape=(ctx.Nx, ctx.Ny), xp=xp)
            ctx.theta_full[k] = theta_f

            R = _residual_local(theta_f, I_mid_f, ctx)
            Rint = R[1:-1, :]
            info = normalize_info(dict(
                rrms=float(asnumpy(xp.sqrt(xp.mean(Rint * Rint)))),
                rmax=float(asnumpy(xp.max(xp.abs(Rint)))),
                converged=True,
            ))

            if (jt % max(1, int(t_stride))) == 0:
                store.save(slot, k, I_mid_f, theta_f, info)

            last_Imax = float(asnumpy(xp.max(I_mid_f)))
            last_rrms = float(info.get("rrms", np.nan))

        if progress:
            progress(f"time step {jt+1}/{int(Nt)}  mode=dg_td_predictor  t={(jt+1)*dt:.4g}  Imax={last_Imax:.3e}  Rrms={last_rrms:.3e}")
        free_backend_memory(xp)

    return stopped


# -----------------------------------------------------------------------------
# Public runner
# -----------------------------------------------------------------------------


def run_lc_validated(
    params: LCParams,
    *,
    run_dir: str | Path,
    mode: str = "strict_static",
    Nt: int = 100,
    dt: float = 5e-4,
    t_stride: int = 1,
    save_slices: bool = True,
    save_full: bool = False,
    progress: Optional[Callable[[str], None]] = print,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    mode = mode.lower().strip()
    if mode not in {"static_fast", "strict_static", "td_predictor_only", "dg_td_predictor"}:
        raise ValueError("mode must be static_fast, strict_static, td_predictor_only, or dg_td_predictor")

    if mode == "dg_td_predictor":
        params = LCParams(**{**asdict(params), "use_dual_grid": True})

    ctx, dg = make_context(params)
    xp = ctx.xp
    apply_legacy_context_aliases(ctx, params)

    ctx.theta_full = xp.stack([ctx.theta_bias_2d.copy() for _ in range(ctx.Nz)], axis=0).astype(xp.float32, copy=False)

    use_legacy_static = _HAS_CUPY and xp is _cupy and mode == "strict_static"
    use_legacy_td = _HAS_CUPY and xp is _cupy and mode == "td_predictor_only"
    use_legacy_dg_td = False

    if mode in {"strict_static", "td_predictor_only"} and not (_HAS_CUPY and xp is _cupy):
        if mode == "td_predictor_only":
            raise RuntimeError("Validated TD predictor requires CuPy/GPU. Use backend='auto' on a GPU node.")

    Nt_eff = int(Nt) if mode in {"td_predictor_only", "dg_td_predictor"} else 1
    t_stride = max(1, int(t_stride))
    Nt_out = (Nt_eff + t_stride - 1) // t_stride

    run_dir = Path(run_dir)
    store = LightStore(run_dir, ctx, Nt_out=Nt_out, save_slices=save_slices, save_full=save_full)

    # Metadata uses the same substep selector as the branch that will run.
    nsub_meta, dz_sub_meta, phi_meta, _, _ = _prepare_substeps(ctx, params, use_core=(_HAS_CUPY and xp is _cupy and mode in {"strict_static", "td_predictor_only"}))
    meta = asdict(params)
    meta.update(
        dict(
            mode=mode,
            Nt=Nt_eff,
            dt=dt,
            t_stride=t_stride,
            Nt_out=Nt_out,
            nsub=nsub_meta,
            dz_sub=dz_sub_meta,
            phi_est=phi_meta,
            backend="cupy" if (_HAS_CUPY and xp is _cupy) else "numpy",
            strict_static_kernel="validated_core" if use_legacy_static else "package_smoke",
            td_kernel="validated_core" if use_legacy_td else "package_smoke",
            dg_td_kernel="validated_core" if use_legacy_dg_td else "package_smoke",
        )
    )

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    stopped = False
    try:
        if mode in {"static_fast", "strict_static"}:
            stopped = _run_static(
                ctx=ctx,
                params=params,
                store=store,
                progress=progress,
                should_stop=should_stop,
                use_legacy_static=use_legacy_static,
                save_full=save_full,
            )
        elif mode == "td_predictor_only":
            stopped = _run_td_predictor(
                ctx=ctx,
                params=params,
                store=store,
                Nt=Nt_eff,
                dt=dt,
                t_stride=t_stride,
                progress=progress,
                should_stop=should_stop,
            )
        else:
            if dg is None:
                raise RuntimeError("Dual-grid mode requested but no dual-grid context was built.")
            stopped = _run_dg_td_smoke(
                ctx=ctx,
                dg=dg,
                params=params,
                store=store,
                Nt=Nt_eff,
                dt=dt,
                t_stride=t_stride,
                progress=progress,
                should_stop=should_stop,
            )

        np.savez(run_dir / "final_summary.npz", elapsed_s=time.time() - t0, stopped=stopped)
        if stopped and progress:
            progress("run stopped by user")
    finally:
        store.close()

    return {
        "run_dir": str(run_dir),
        "metadata": str(run_dir / "metadata.json"),
        "scalar_log": str(run_dir / "scalar_log.csv"),
    }


# -----------------------------------------------------------------------------
# Viewer helpers
# -----------------------------------------------------------------------------


def load_run_arrays(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    Nx, Ny, Nz = int(meta["Nx"]), int(meta["Ny"]), int(meta["Nz"])
    Nt_out = int(meta.get("Nt_out", (int(meta.get("Nt", 1)) + int(meta.get("t_stride", 1)) - 1) // int(meta.get("t_stride", 1))))
    out = {"metadata": meta, "run_dir": run_dir}
    for name, shape in {
        "Ixz": (Nt_out, Nz, Nx),
        "Iyz": (Nt_out, Nz, Ny),
        "dthetaxz": (Nt_out, Nz, Nx),
        "dthetayz": (Nt_out, Nz, Ny),
    }.items():
        path = run_dir / f"{name}.dat"
        if path.exists():
            out[name] = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
    return out


__all__ = [
    "LCParams",
    "LCContext",
    "DualGrid",
    "get_backend",
    "run_lc_validated",
    "load_run_arrays",
    "lc_b_from_voltage",
    "voltage_from_lc_b",
    "theta0_from_b_zero_bc",
    "b_from_theta0_zero_bc",
]
