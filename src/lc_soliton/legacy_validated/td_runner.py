from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..core.backend import _HAS_CUPY, _cupy, asnumpy, free_backend_memory
from ..core.context import LCContext, LCParams, DualGrid, LegacyPlans
from ..core.storage import LightStore
from ..core.launch import intensity
from ..core.dual_grid import restrict_block_mean, prolong_repeat
from .runner_utils import residual_quality_info
from ..validated_core.runner_core import (
    hop_linear as core_hop_linear,
    intens_into,
    prepare_ie_ky_operator,
    prepare_cn_ky_operator,
    td_get_slice_mid_intensity,
    td_update_theta_from_midintensity,
    lc_residual64,
    residual_stats_2d,
    make_coarse_theta_ctx
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

class IdentityThetaGrid:
    """
    Adapter for the ordinary full-grid TD case.

    This makes full-grid TD the identity special case of the future
    dual-grid TD path.
    """

    def __init__(self, ctx: LCContext):
        self.ctx_theta = ctx
        self.theta_stack = ctx.theta_full

    def snapshot(self):
        return self.theta_stack.copy()

    def theta_for_optics(self, theta_k):
        return theta_k

    def restrict_I(self, I_full):
        return I_full

    def store_theta(self, k: int, theta_new):
        self.theta_stack[k] = theta_new
        return theta_new

class DualGridThetaGrid:
    def __init__(self, ctx, dg):
        self.ctx_full = ctx
        self.dg = dg
        self.ctx_theta = make_coarse_theta_ctx(ctx, factor=dg.factor)

        xp = ctx.xp
        self.theta_stack = xp.stack(
            [dg.theta_bias_c.copy() for _ in range(ctx.Nz)],
            axis=0,
        )

    def snapshot(self):
        return self.theta_stack.copy()

    def theta_for_optics(self, theta_c):
        return prolong_repeat(
            theta_c,
            self.dg.factor,
            target_shape=(self.ctx_full.Nx, self.ctx_full.Ny),
            xp=self.ctx_full.xp,
        )

    def restrict_I(self, I_full):
        return restrict_block_mean(
            I_full,
            self.dg.factor,
            xp=self.ctx_full.xp,
        )

    def store_theta(self, k, theta_c_new):
        self.theta_stack[k] = theta_c_new

        theta_f = self.theta_for_optics(theta_c_new)
        self.ctx_full.theta_full[k] = theta_f

        return theta_f

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
    return residual_quality_info(theta, I_mid, ctx, R)


def _run_td_predictor(
    *,
    ctx: LCContext,
    dg: DualGrid | None = None,
    params: LCParams,
    store: LightStore,
    Nt: int,
    dt: float,
    t_stride: int,
    progress: Optional[Callable[[str], None]],
    should_stop: Optional[Callable[[], bool]],
) -> bool:
    xp = ctx.xp
    
    cpu_td = not (_HAS_CUPY and xp is _cupy)
    
    if cpu_td:
        print(
            "[CPU TD experimental mode] validated TD predictor running on NumPy/SciPy backend."
        )
    nsub, dz_sub, phi, h_sub, h_half = _prepare_substeps(
        ctx, params, use_core=not cpu_td
    )
    plans = _prepare_legacy_plans(ctx) if not cpu_td else LegacyPlans()

    theta_grid = (
        DualGridThetaGrid(ctx, dg)
        if dg is not None
        else IdentityThetaGrid(ctx)
    )
    print("[TD] theta grid adapter =", type(theta_grid).__name__)
    theta_ctx = theta_grid.ctx_theta

    plans.sS, plans.offS, plans.diagS, plans.lamS = prepare_cn_ky_operator(
        dt=float(params.dtau_static),
        mobility=float(theta_ctx.mobility),
        du=float(theta_ctx.du),
        dv=float(theta_ctx.dv),
        Ny=int(theta_ctx.Ny),
    )

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
        theta_ctx = theta_grid.ctx_theta
        a_ie, offTD, diagTD, lam_yTD = prepare_ie_ky_operator(
            dt=float(dt),
            mobility=float(theta_ctx.mobility),
            du=float(theta_ctx.du),
            dv=float(theta_ctx.dv),
            Ny=int(theta_ctx.Ny),
        )

        theta_ref = theta_grid.snapshot()
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
            theta_opt = theta_grid.theta_for_optics(theta_old)
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
                theta_opt,
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
            I_mid_theta = theta_grid.restrict_I(I_mid)
            theta = td_update_theta_from_midintensity(
                theta_grid.ctx_theta,
                theta_old,
                tp,
                tn,
                I_mid_theta,
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
            theta_save = theta_grid.store_theta(k, theta)
            if k == 0 and jt == 0:
                print(
                    "[TD residual debug]",
                    "theta_save", getattr(theta_save, "shape", None),
                    "I_mid", getattr(I_mid, "shape", None),
                    "ctx", ctx.Nx, ctx.Ny,
                    "theta_grid", type(theta_grid).__name__,
                )


            
            if isinstance(theta_grid, DualGridThetaGrid):
                info = _make_scalar_info_from_residual(
                    theta,
                    I_mid_theta,
                    theta_grid.ctx_theta,
                )
            else:
                info = _make_scalar_info_from_residual(theta_save, I_mid, ctx)

            if (jt % max(1, int(t_stride))) == 0:
                store.save(slot, k, I_mid, theta_save, info)

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

