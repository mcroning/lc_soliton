"""
lc_soliton.diagnostics.theta

Shared theta diagnostics for the new LC soliton architecture.

This module contains measurements only.  It does not advance theta,
solve nonlinear equations, propagate optics, or know about workflows.

All routines are backend/dtype transparent: callers provide `xp`
(NumPy or CuPy), and arrays determine precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from lc_soliton.theta_engine.residuals import (
    cn_residual,
    pde_residual,
    residual_stats,
)


@dataclass
class ThetaDiagnostics:
    """Compact summary of theta diagnostics for one state or step."""

    pde_rms: float
    pde_max: float
    cn_rms: Optional[float] = None
    cn_max: Optional[float] = None
    update_rms: Optional[float] = None
    update_max: Optional[float] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "pde_rms": self.pde_rms,
            "pde_max": self.pde_max,
            "cn_rms": self.cn_rms,
            "cn_max": self.cn_max,
            "update_rms": self.update_rms,
            "update_max": self.update_max,
        }


def theta_update_stats(theta_new, theta_old, *, xp) -> Dict[str, float]:
    """
    RMS and maximum update size on the interior x rows.
    """

    dtheta = theta_new - theta_old
    dint = dtheta[1:-1, :]

    return {
        "update_rms": float(xp.sqrt(xp.mean(dint * dint))),
        "update_max": float(xp.max(xp.abs(dint))),
    }


def theta_pde_stats(theta, intensity, ctx, *, xp) -> Dict[str, float]:
    """
    PDE residual statistics for a theta/intensity state.
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


def theta_cn_stats(theta_new, theta_old, intensity, ctx, *, dt, xp) -> Dict[str, float]:
    """
    Crank-Nicolson equation residual statistics for one accepted theta step.
    """

    R = cn_residual(
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

    st = residual_stats(R, xp=xp)

    return {
        "cn_rms": st["rms_interior"],
        "cn_max": st["max_interior"],
        "cn_rms_full": st["rms_full"],
        "cn_max_full": st["max_full"],
    }


def summarize_theta_state(theta, intensity, ctx, *, xp) -> ThetaDiagnostics:
    """
    Diagnostics for a single theta/intensity state.
    """

    pde = theta_pde_stats(theta, intensity, ctx, xp=xp)

    return ThetaDiagnostics(
        pde_rms=pde["pde_rms"],
        pde_max=pde["pde_max"],
    )


def summarize_theta_step(theta_new, theta_old, intensity, ctx, *, dt, xp) -> ThetaDiagnostics:
    """
    Diagnostics for one accepted theta step.

    Includes PDE residual of theta_new, CN residual for the step, and
    update norm theta_new-theta_old.
    """

    pde = theta_pde_stats(theta_new, intensity, ctx, xp=xp)
    cn = theta_cn_stats(theta_new, theta_old, intensity, ctx, dt=dt, xp=xp)
    upd = theta_update_stats(theta_new, theta_old, xp=xp)

    return ThetaDiagnostics(
        pde_rms=pde["pde_rms"],
        pde_max=pde["pde_max"],
        cn_rms=cn["cn_rms"],
        cn_max=cn["cn_max"],
        update_rms=upd["update_rms"],
        update_max=upd["update_max"],
    )


def theta_history_row(
    *,
    step: int,
    theta,
    intensity,
    ctx,
    xp,
    theta_old=None,
    dt: Optional[float] = None,
    cn_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Standard row for theta workflow histories.

    If `theta_old` and `dt` are supplied, include update and CN residual
    diagnostics.  If `cn_info` is supplied, include nonlinear-solver details.
    """

    pde = theta_pde_stats(theta, intensity, ctx, xp=xp)

    row: Dict[str, Any] = {
        "step": int(step),
        "pde_rms": pde["pde_rms"],
        "pde_max": pde["pde_max"],
    }

    if theta_old is not None:
        row.update(theta_update_stats(theta, theta_old, xp=xp))

    if theta_old is not None and dt is not None:
        row.update(theta_cn_stats(theta, theta_old, intensity, ctx, dt=float(dt), xp=xp))
    else:
        row.update({"cn_rms": None, "cn_max": None})

    if cn_info is not None:
        row["picard_iters"] = cn_info.get("niter")
        row["picard_update_rms"] = cn_info.get("update_rms")
        row["picard_update_max"] = cn_info.get("update_max")
        row["picard_converged_update"] = cn_info.get("converged_update")
        row["picard_converged_cn"] = cn_info.get("converged_cn")
    else:
        row["picard_iters"] = None

    return row
