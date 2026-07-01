"""
lc_soliton.workflows.timedependent

Time-dependent theta workflow.

This initial workflow is intentionally theta-only with frozen intensity.
It is the clean acceptance test for physical CN timestepping before optics
and global-z coupling are connected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lc_soliton.theta_engine.engine import advance_theta_cn
from lc_soliton.diagnostics.theta import (
    theta_state_diagnostics,
    theta_transition_diagnostics,
    theta_history_row,
)


@dataclass
class TimeDependentThetaResult:
    """Result returned by run_timedependent_theta_frozenI."""

    theta: Any
    history: List[Dict[str, Any]]
    info: Dict[str, Any]


def run_timedependent_theta_frozenI(
    theta_seed,
    intensity,
    ctx,
    *,
    dt: Optional[float] = None,
    steps: int = 100,
    dtype=None,
    xp,
) -> TimeDependentThetaResult:
    """
    Advance theta for a fixed intensity using accurate CN timesteps.

    This is a physical time-integration workflow: every accepted step is the
    full CN solution from theta_engine.advance_theta_cn.  There is no outer
    under-relaxation in this workflow.
    """

    theta = theta_seed.astype(dtype, copy=True) if dtype is not None else theta_seed.copy()
    I = intensity.astype(dtype, copy=False) if dtype is not None else intensity

    dt_use = float(getattr(ctx, "dt", 0.0) if dt is None else dt)

    history: List[Dict[str, Any]] = []

    diag0 = theta_state_diagnostics(theta, I, ctx, xp=xp)
    history.append(
        theta_history_row(
            step=0,
            state_diagnostics=diag0,
            transition_diagnostics=None,
            cn_info=None,
        )
    )

    last_info = None

    for step in range(1, int(steps) + 1):
        theta_old = theta

        theta, cn_info = advance_theta_cn(
            theta_old,
            I,
            ctx,
            dt=dt_use,
            dtype=dtype,
            xp=xp,
        )

        state_diag = theta_state_diagnostics(theta, I, ctx, xp=xp)
        trans_diag = theta_transition_diagnostics(theta, theta_old, I, ctx, dt=dt_use, xp=xp)

        history.append(
            theta_history_row(
                step=step,
                state_diagnostics=state_diag,
                transition_diagnostics=trans_diag,
                cn_info=cn_info,
            )
        )

        last_info = cn_info

    info = {
        "nsteps": int(steps),
        "dt": dt_use,
        "final_pde_rms": history[-1]["pde_rms"],
        "final_pde_max": history[-1]["pde_max"],
        "last_cn_info": last_info,
    }

    return TimeDependentThetaResult(
        theta=theta,
        history=history,
        info=info,
    )
