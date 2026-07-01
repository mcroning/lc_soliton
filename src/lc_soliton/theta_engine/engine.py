"""
theta_engine/engine.py

Public API for theta evolution/relaxation.
"""

from __future__ import annotations

from .nonlinear import solve_theta_cn_picard
from .residuals import pde_residual, residual_stats


def advance_theta_cn(theta, intensity, ctx, *, dt=None, dtype=None, xp):
    """
    One physical CN timestep.
    Full accepted step: omega = 1 by construction.
    """

    if dtype is not None:
        theta = theta.astype(dtype, copy=False)
        intensity = intensity.astype(dtype, copy=False)

    dt = float(ctx.dt if dt is None else dt)

    return solve_theta_cn_picard(
        theta,
        intensity,
        dt=dt,
        du=float(ctx.du),
        dv=float(ctx.dv),
        b=float(ctx.b),
        bi=float(ctx.bi),
        mobility=float(getattr(ctx, "mobility", 1.0)),
        theta_bc=float(getattr(ctx, "theta_bc", 0.0)),
        theta_clamp=getattr(ctx, "theta_clamp", None),
        max_iter=int(getattr(ctx, "picard_iters", 20)),
        tol_update=float(getattr(ctx, "picard_tol_up", 1e-14)),
        tol_cn=float(getattr(ctx, "picard_tol_cn", 1e-10)),
        monitor_cn=True,
        save_history=False,
        xp=xp,
    )


def relax_theta_steady(
    theta,
    intensity,
    ctx,
    *,
    dtau=None,
    max_steps=None,
    residual_tol_rms=None,
    residual_tol_max=None,
    dtype=None,
    xp,
):
    """
    Pseudo-time steady theta solve.

    Uses repeated accurate CN steps and stops on PDE residual.
    """

    if dtype is not None:
        theta = theta.astype(dtype, copy=True)
        intensity = intensity.astype(dtype, copy=False)
    else:
        theta = theta.copy()

    dtau = float(getattr(ctx, "dtau_static", 0.02) if dtau is None else dtau)
    max_steps = int(getattr(ctx, "static_max_steps", 1000) if max_steps is None else max_steps)
    residual_tol_rms = float(getattr(ctx, "static_tol_rms", 5e-3) if residual_tol_rms is None else residual_tol_rms)
    residual_tol_max = float(getattr(ctx, "static_tol_max", 5e-2) if residual_tol_max is None else residual_tol_max)

    last_info = None

    for step in range(max_steps):
        theta, last_info = solve_theta_cn_picard(
            theta,
            intensity,
            dt=dtau,
            du=float(ctx.du),
            dv=float(ctx.dv),
            b=float(ctx.b),
            bi=float(ctx.bi),
            mobility=float(getattr(ctx, "mobility", 1.0)),
            theta_bc=float(getattr(ctx, "theta_bc", 0.0)),
            theta_clamp=getattr(ctx, "theta_clamp", None),
            max_iter=int(getattr(ctx, "picard_iters", 20)),
            tol_update=float(getattr(ctx, "picard_tol_up", 1e-14)),
            tol_cn=float(getattr(ctx, "picard_tol_cn", 1e-10)),
            monitor_cn=True,
            save_history=False,
            xp=xp,
        )

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

        if (
            st["rms_interior"] <= residual_tol_rms
            and st["max_interior"] <= residual_tol_max
        ):
            return theta, {
                "converged": True,
                "nsteps": step + 1,
                "pde_rms": st["rms_interior"],
                "pde_max": st["max_interior"],
                "last_cn_info": last_info,
            }

    return theta, {
        "converged": False,
        "nsteps": max_steps,
        "pde_rms": st["rms_interior"],
        "pde_max": st["max_interior"],
        "last_cn_info": last_info,
    }
solve_theta_steady = relax_theta_steady  ##old test compatibility