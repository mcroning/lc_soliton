# lc_core/static_solver.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .residuals import (
    lc_residual2d_ctx,
    lc_residual_slice3d_ctx,
    residual_stats_2d,
)


@dataclass
class StaticOperatorBundle:
    """
    CN/trap-Picard operator bundle for the theta static relaxer.

    The contents are intentionally named like the legacy strict solver expects:
        sS, offS, diagS, lamS
    """
    sS: Any
    offS: Any
    diagS: Any
    lamS: Any


@dataclass
class StaticSliceResult:
    theta: Any
    info: dict


def prepare_static_operator_bundle(ctx, *, prepare_cn_ky_operator):
    """
    Prepare the static theta CN operator using the legacy validated operator builder.

    This is a bridge step. Later, prepare_cn_ky_operator can be moved fully into
    lc_core.
    """
    sS, offS, diagS, lamS = prepare_cn_ky_operator(
        dt=float(ctx.dtau_static),
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        Ny=int(ctx.Ny),
    )
    return StaticOperatorBundle(sS=sS, offS=offS, diagS=diagS, lamS=lamS)


def strict_relax_theta_slice_bridge(
    ctx,
    I_mid,
    *,
    theta_seed=None,
    theta_prev=None,
    theta_next=None,
    operators: StaticOperatorBundle | None = None,
    prepare_cn_ky_operator=None,
    strict_static_relax_slice,
    use_linear_seed=True,
    early_accept_linear_seed=True,
    residual_tol_max=5e-2,
    residual_tol_rms=1e-2,
    max_outer_passes=8,
    verbose=False,
):
    """
    Solve one fixed-intensity theta slice using the legacy strict_static_relax_slice.

    This intentionally isolates the legacy dependency:
        strict_static_relax_slice(...)

    Parameters
    ----------
    ctx:
        New LCContextGPU.
    I_mid:
        Fixed forcing intensity for this z-slice.
    theta_seed:
        Initial theta guess. Defaults to ctx.theta_bias_2d.
    theta_prev, theta_next:
        z neighbors. Defaults to theta_seed, which is appropriate when
        theta_z_gamma = 0 or for a local smoke test.
    operators:
        Optional StaticOperatorBundle.
    prepare_cn_ky_operator:
        Required only if operators is None.

    Returns
    -------
    StaticSliceResult
    """
    if theta_seed is None:
        theta_seed = ctx.theta_bias_2d

    if theta_prev is None:
        theta_prev = theta_seed
    if theta_next is None:
        theta_next = theta_seed

    if operators is None:
        if prepare_cn_ky_operator is None:
            raise ValueError("prepare_cn_ky_operator is required when operators is None.")
        operators = prepare_static_operator_bundle(
            ctx,
            prepare_cn_ky_operator=prepare_cn_ky_operator,
        )

    theta, info = strict_static_relax_slice(
        theta_seed,
        I_mid,
        theta_prev,
        theta_next,
        ctx,
        sS=operators.sS,
        offS=operators.offS,
        diagS=operators.diagS,
        lamS=operators.lamS,
        use_linear_seed=bool(use_linear_seed),
        early_accept_linear_seed=bool(early_accept_linear_seed),
        residual_tol_max=float(residual_tol_max),
        residual_tol_rms=float(residual_tol_rms),
        max_outer_passes=int(max_outer_passes),
        verbose=bool(verbose),
    )

    # Add new-core residual report so we have a common diagnostic target.
    R2 = lc_residual2d_ctx(ctx, theta, I_mid)
    info = dict(info)
    info["new_core_residual"] = residual_stats_2d(R2)

    return StaticSliceResult(theta=theta, info=info)


def static_slice_smoke_test(
    ctx,
    launch_arrays,
    *,
    prepare_cn_ky_operator,
    strict_static_relax_slice,
    residual_tol_max=5e-2,
    residual_tol_rms=1e-2,
    max_outer_passes=8,
    use_linear_seed=True,
    verbose=False,
):
    """
    Minimal static-theta smoke test.

    It solves theta for the initial launch intensity only; it does not propagate
    optics or march in z.

    This should be the first static solver test.
    """
    operators = prepare_static_operator_bundle(
        ctx,
        prepare_cn_ky_operator=prepare_cn_ky_operator,
    )

    I_mid = launch_arrays.I0.astype(ctx.xp.float32, copy=False)

    out = strict_relax_theta_slice_bridge(
        ctx,
        I_mid,
        theta_seed=ctx.theta_bias_2d,
        theta_prev=ctx.theta_bias_2d,
        theta_next=ctx.theta_bias_2d,
        operators=operators,
        strict_static_relax_slice=strict_static_relax_slice,
        use_linear_seed=use_linear_seed,
        early_accept_linear_seed=True,
        residual_tol_max=residual_tol_max,
        residual_tol_rms=residual_tol_rms,
        max_outer_passes=max_outer_passes,
        verbose=verbose,
    )

    return out


def summarize_static_slice_result(result: StaticSliceResult) -> dict:
    """
    JSON-safe summary of a static slice result.
    """
    info = dict(result.info)
    theta = result.theta

    xp = _xp_of(theta)
    info["theta_shape"] = list(theta.shape)
    info["theta_min"] = float(xp.min(theta).get() if hasattr(xp.min(theta), "get") else xp.min(theta))
    info["theta_max"] = float(xp.max(theta).get() if hasattr(xp.max(theta), "get") else xp.max(theta))
    info["theta_mean"] = float(xp.mean(theta).get() if hasattr(xp.mean(theta), "get") else xp.mean(theta))

    return info


def write_static_slice_summary(result: StaticSliceResult, run_dir, filename="static_slice_summary.json"):
    """
    Write JSON-safe static slice summary.
    """
    import json
    from pathlib import Path

    path = Path(run_dir) / filename
    path.write_text(json.dumps(summarize_static_slice_result(result), indent=2))
    return path


def _xp_of(arr):
    mod = type(arr).__module__.split(".")[0]
    if mod == "cupy":
        import cupy as cp
        return cp
    import numpy as np
    return np
