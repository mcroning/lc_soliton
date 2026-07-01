"""
lc_soliton.diagnostics.theta

Shared theta diagnostics.

These functions centralize the measurements used by static, TD,
eigensoliton, and stability workflows.  They intentionally do not solve
anything.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lc_soliton.theta_engine.residuals import (
    pde_residual,
    cn_residual,
    residual_stats,
)


def theta_state_diagnostics(theta, intensity, ctx, *, xp) -> Dict[str, Any]:
    """
    Diagnostics for a single theta state.

    Returns only PDE residual statistics.  CN and update quantities are
    transition diagnostics, not state diagnostics.
    """

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
    st = residual_stats(R, xp=xp)

    return {
        "pde_rms": st["rms_interior"],
        "pde_max": st["max_interior"],
        "pde_rms_full": st["rms_full"],
        "pde_max_full": st["max_full"],
    }


def theta_transition_diagnostics(theta_new, theta_old, intensity, ctx, *, dt, xp) -> Dict[str, Any]:
    """
    Diagnostics for one accepted theta transition.
    """

    dtheta = theta_new - theta_old
    di = dtheta[1:-1, :]

    update_rms = float(xp.sqrt(xp.mean(di * di)))
    update_max = float(xp.max(xp.abs(di)))

    Rcn = cn_residual(
        theta_new,
        theta_old,
        intensity,
        dt=float(dt),
        du=float(ctx.du),
        dv=float(ctx.dv),
        b=float(ctx.b),
        bi=float(ctx.bi),
        mobility=float(getattr(ctx, "mobility", 1.0)),
        xp=xp,
    )
    st_cn = residual_stats(Rcn, xp=xp)

    return {
        "update_rms": update_rms,
        "update_max": update_max,
        "cn_rms": st_cn["rms_interior"],
        "cn_max": st_cn["max_interior"],
        "cn_rms_full": st_cn["rms_full"],
        "cn_max_full": st_cn["max_full"],
    }


def theta_history_row(
    *,
    step: int,
    state_diagnostics: Dict[str, Any],
    transition_diagnostics: Optional[Dict[str, Any]] = None,
    cn_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a flat scalar-history row.
    """

    transition_diagnostics = transition_diagnostics or {}
    cn_info = cn_info or {}

    return {
        "step": int(step),
        "pde_rms": state_diagnostics.get("pde_rms"),
        "pde_max": state_diagnostics.get("pde_max"),
        "update_rms": transition_diagnostics.get("update_rms"),
        "update_max": transition_diagnostics.get("update_max"),
        "cn_rms": transition_diagnostics.get("cn_rms"),
        "cn_max": transition_diagnostics.get("cn_max"),
        "cn_rms_full": transition_diagnostics.get("cn_rms_full"),
        "cn_max_full": transition_diagnostics.get("cn_max_full"),
        "picard_iters": cn_info.get("niter"),
        "picard_update_rms": cn_info.get("update_rms"),
        "picard_update_max": cn_info.get("update_max"),
        "picard_converged_update": cn_info.get("converged_update"),
        "picard_converged_cn": cn_info.get("converged_cn"),
    }
