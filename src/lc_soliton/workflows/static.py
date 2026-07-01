"""
lc_soliton.workflows.static

Thin static workflow layer for theta relaxation.

This module deliberately contains no numerical stencil or solver logic.
It orchestrates the validated theta_engine public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from lc_soliton.theta_engine.engine import relax_theta_steady
from lc_soliton.theta_engine.residuals import pde_residual, residual_stats


@dataclass
class StaticThetaResult:
    """Result container for a frozen-I static theta relaxation."""

    theta: Any
    info: dict
    initial_pde: dict
    final_pde: dict


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
    Relax theta toward a steady solution for fixed intensity.

    Parameters
    ----------
    theta_seed : array
        Initial theta field.
    intensity : array
        Fixed optical intensity on the theta grid.
    ctx : object
        Context-like object with du, dv, b, bi, mobility, theta_bc, etc.
    dtau, max_steps, residual_tol_rms, residual_tol_max : optional
        Overrides for pseudo-time relaxation settings.
    dtype : optional
        If supplied, theta/intensity are cast to this dtype before solving.
    xp : module
        NumPy or CuPy module.

    Returns
    -------
    StaticThetaResult
    """

    if dtype is not None:
        theta0 = theta_seed.astype(dtype, copy=False)
        I = intensity.astype(dtype, copy=False)
    else:
        theta0 = theta_seed
        I = intensity

    R0 = pde_residual(
        theta0,
        I,
        du=float(ctx.du),
        dv=float(ctx.dv),
        b=float(ctx.b),
        bi=float(ctx.bi),
        mobility=float(getattr(ctx, "mobility", 1.0)),
        xp=xp,
    )
    initial_pde = residual_stats(R0, xp=xp)

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

    R1 = pde_residual(
        theta,
        I,
        du=float(ctx.du),
        dv=float(ctx.dv),
        b=float(ctx.b),
        bi=float(ctx.bi),
        mobility=float(getattr(ctx, "mobility", 1.0)),
        xp=xp,
    )
    final_pde = residual_stats(R1, xp=xp)

    return StaticThetaResult(
        theta=theta,
        info=info,
        initial_pde=initial_pde,
        final_pde=final_pde,
    )
