from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..core.context import LegacyPlans
from ..core.backend import _HAS_CUPY, _cupy, asnumpy
from ..validated_core.runner_core import (
    hop_linear as core_hop_linear,
    prepare_cn_ky_operator,
    strict_static_relax_slice_selfconsistent,
    lc_residual64,
)

from .runner_utils import (
    normalize_info,
    _prepare_substeps,
    _prepare_legacy_plans,
    residual_quality_info,
    _hop_linear_local,
)

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

    plans.sS, plans.offS, plans.diagS, plans.lamS = prepare_cn_ky_operator(
        dt=float(params.dtau_static),
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        Ny=int(ctx.Ny),
    )

    amp = (
        core_hop_linear(ctx.amp0.copy(), h_half, ctx.windowxy)
        if use_legacy_static
        else _hop_linear_local(ctx, ctx.amp0.copy(), h_half)
    )
    stopped = False

    for k in range(ctx.Nz):
        if should_stop is not None and should_stop():
            stopped = True
            if progress:
                progress("run stopped by user")
            break
    
        theta_seed = ctx.theta_full[k - 1] if k > 0 else ctx.theta_bias_2d.copy()
    
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
        )

    
        info = normalize_info(info)
    
        ctx.theta_full[k] = theta

        R = lc_residual64(
            theta,
            I_mid,
            b=ctx.b,
            bi=ctx.bi,
            du=ctx.du,
            dv=ctx.dv,
            mobility=ctx.mobility,
        )
        info.update(residual_quality_info(theta, I_mid, ctx, R))
        
        store.save(0, k, I_mid, theta, info)
    
        if progress and (
            k == 0
            or (k + 1) % max(1, ctx.Nz // 20) == 0
            or k == ctx.Nz - 1
        ):
            progress(
                f"static z {k+1}/{ctx.Nz}  "
                f"Imax={float(asnumpy(xp.max(I_mid))):.3e}  "
                f"Rrms={float(info.get('rrms', np.nan)):.3e}"
            )
    
    return stopped
