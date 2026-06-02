from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..core.backend import _HAS_CUPY, _cupy, asnumpy, free_backend_memory
from ..core.context import LCContext, LCParams, DualGrid
from ..core.storage import LightStore
from ..core.launch import intensity
from ..core.dual_grid import restrict_block_mean, prolong_repeat

from ..validated_core.runner_core import (
    hop_linear as core_hop_linear,
    intens_into,
    prepare_ie_ky_operator,
    td_get_slice_mid_intensity,
    td_update_theta_from_midintensity,
    lc_residual64,
    residual_stats_2d,
)



from .runner_utils import (
    normalize_info,
    _prepare_substeps,
    _prepare_legacy_plans,
)

__all__ = [
    "_run_td_predictor",
    "_run_dg_td_experimental",
]

def _make_scalar_info_from_residual(theta, I_mid, ctx) -> dict:
    R = lc_residual64(
        theta,
        I_mid,
        b=ctx.b,
        bi=ctx.bi,
        du=ctx.du,
        dv=ctx.dv,
        mobility=ctx.mobility,
    )
    stats = residual_stats_2d(R)
    return normalize_info(
        dict(
            rrms=stats["rms_interior"],
            rmax=stats["max_interior"],
            converged=True,
        )
    )


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


def _run_dg_td_experimental(
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

