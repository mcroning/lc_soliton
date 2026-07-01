"""
lc_soliton.workflows.static

Static theta workflow.

This module intentionally contains no PDE discretization, no CN algebra,
and no Picard logic.  It orchestrates the validated theta engine and the
shared diagnostics layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from lc_soliton.theta_engine.engine import relax_theta_steady
from lc_soliton.diagnostics.theta import theta_state_diagnostics


@dataclass
class StaticThetaResult:
    """Result returned by run_static_theta."""

    theta: Any
    info: Dict[str, Any]
    diagnostics_initial: Dict[str, Any]
    diagnostics_final: Dict[str, Any]


def run_static_theta(
    theta_seed,
    intensity,
    ctx,
    *,
    dtau: Optional[float] = None,
    max_steps: Optional[int] = None,
    residual_tol_rms: Optional[float] = None,
    residual_tol_max: Optional[float] = None,
    dtype=None,
    xp,
) -> StaticThetaResult:
    """
    Relax theta toward a steady solution for a fixed intensity.

    Parameters
    ----------
    theta_seed, intensity
        Initial theta and fixed optical intensity.
    ctx
        Context-like object with du, dv, b, bi, mobility, theta_bc, etc.
    dtau, max_steps, residual_tol_rms, residual_tol_max
        Optional overrides for the pseudo-time relaxer.
    dtype, xp
        Precision/backend policy.  The caller chooses these explicitly.

    Returns
    -------
    StaticThetaResult
        The relaxed theta, engine info, and standardized diagnostics.
    """

    theta0 = theta_seed.astype(dtype, copy=False) if dtype is not None else theta_seed
    I = intensity.astype(dtype, copy=False) if dtype is not None else intensity

    diag0 = theta_state_diagnostics(
        theta0,
        I,
        ctx,
        xp=xp,
    )

    theta, info = relax_theta_steady(
        theta0,
        I,
        ctx,
        dtau=dtau,
        max_steps=max_steps,
        residual_tol_rms=residual_tol_rms,
        residual_tol_max=residual_tol_max,
        dtype=dtype,
        xp=xp,
    )

    diag1 = theta_state_diagnostics(
        theta,
        I,
        ctx,
        xp=xp,
    )

    return StaticThetaResult(
        theta=theta,
        info=info,
        diagnostics_initial=diag0,
        diagnostics_final=diag1,
    )
