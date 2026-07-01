"""
lc_soliton.workflows.timedependent

Thin workflow layer for time-dependent theta evolution.

This workflow intentionally contains no PDE discretization details.
It delegates the numerical work to theta_engine.advance_theta_cn().

Initial scope:
    frozen-I theta-only TD evolution.

Later, the same workflow shape can be extended to coupled optics by
replacing the fixed intensity with an intensity update between theta steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lc_soliton.theta_engine.engine import advance_theta_cn
from lc_soliton.theta_engine.residuals import pde_residual, residual_stats


@dataclass
class TimeDependentThetaResult:
    """Result container for theta-only time-dependent evolution."""

    theta: Any
    history: List[Dict[str, float]]
    info: Dict[str, Any]


def _pde_stats(theta, intensity, ctx, *, xp) -> Dict[str, float]:
    """Compute steady-PDE residual statistics for the current theta state."""
    R = pde_residual(
        theta,
        intensity,
        du=float(ctx.du),
        dv=float(ctx.dv),
        b=float(ctx.b),
        bi=float(ctx.bi),
        mobility=float(getattr(ctx, "mobility", 1.0)),
        xp=xp,
    )
    return residual_stats(R, xp=xp)


def run_theta_td_frozen_intensity(
    theta_seed,
    intensity,
    ctx,
    *,
    dt: Optional[float] = None,
    nsteps: Optional[int] = None,
    dtype=None,
    xp,
    record_every: int = 1,
) -> TimeDependentThetaResult:
    """
    Run theta-only time-dependent evolution with frozen intensity.

    Parameters
    ----------
    theta_seed
        Initial theta field.
    intensity
        Fixed optical intensity field.
    ctx
        Context-like object with du, dv, b, bi, mobility, theta_bc, and
        Picard/CN tolerance attributes.
    dt
        Physical timestep. If None, uses ctx.dt.
    nsteps
        Number of TD steps. If None, uses ctx.nt or ctx.Nt if present.
    dtype
        Optional dtype conversion for theta/intensity, e.g. cp.float64.
    xp
        NumPy or CuPy module.
    record_every
        Store diagnostics every this many steps.

    Returns
    -------
    TimeDependentThetaResult
        theta, diagnostic history, and summary info.

    Notes
    -----
    This is physical TD in the sense that each accepted step is the full
    accurate CN solution. There is no under-relaxation of the accepted state.
    """

    if dtype is not None:
        theta = theta_seed.astype(dtype, copy=True)
        intensity = intensity.astype(dtype, copy=False)
    else:
        theta = theta_seed.copy()

    if dt is None:
        dt = float(getattr(ctx, "dt"))
    else:
        dt = float(dt)

    if nsteps is None:
        nsteps = int(getattr(ctx, "nt", getattr(ctx, "Nt", 1)))
    else:
        nsteps = int(nsteps)

    record_every = max(1, int(record_every))

    history: List[Dict[str, float]] = []

    st0 = _pde_stats(theta, intensity, ctx, xp=xp)
    history.append(
        {
            "step": 0,
            "time": 0.0,
            "pde_rms": st0["rms_interior"],
            "pde_max": st0["max_interior"],
            "dtheta_rms": 0.0,
            "cn_rms": 0.0,
            "picard_iters": 0,
        }
    )

    last_cn_info: Dict[str, Any] = {}

    for step in range(1, nsteps + 1):
        theta_old = theta

        theta, cn_info = advance_theta_cn(
            theta_old,
            intensity,
            ctx,
            dt=dt,
            dtype=dtype,
            xp=xp,
        )

        last_cn_info = cn_info

        if step % record_every == 0 or step == nsteps:
            D = theta - theta_old
            dtheta_rms = float(xp.sqrt(xp.mean(D[1:-1, :] * D[1:-1, :])))
            st = _pde_stats(theta, intensity, ctx, xp=xp)

            history.append(
                {
                    "step": int(step),
                    "time": float(step * dt),
                    "pde_rms": st["rms_interior"],
                    "pde_max": st["max_interior"],
                    "dtheta_rms": dtheta_rms,
                    "cn_rms": float(cn_info.get("cn_rms", float("nan"))),
                    "picard_iters": int(cn_info.get("niter", -1)),
                }
            )

    info = {
        "nsteps": nsteps,
        "dt": dt,
        "final_pde_rms": history[-1]["pde_rms"],
        "final_pde_max": history[-1]["pde_max"],
        "last_cn_info": last_cn_info,
    }

    return TimeDependentThetaResult(theta=theta, history=history, info=info)
