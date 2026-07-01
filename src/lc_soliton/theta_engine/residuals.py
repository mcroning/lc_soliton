"""
theta_engine/residuals.py

Residual evaluators for the LC director equation.

These routines are diagnostics only:
    - PDE residual: steady-state equation
    - CN residual: implicit timestep equation

No hard-coded precision. Arrays determine dtype.
"""

from __future__ import annotations

from .operators import theta_operator, theta_operator_z


def pde_residual(
    theta,
    intensity,
    *,
    du,
    dv,
    b,
    bi,
    mobility=1.0,
    xp,
):
    """
    Steady PDE residual:

        R = [L(theta) + (b+bi I) sin(2 theta)] / mobility
    """

    R = theta_operator(
        theta,
        intensity,
        du=du,
        dv=dv,
        b=b,
        bi=bi,
        xp=xp,
    ) / mobility

    R[0, :] = 0
    R[-1, :] = 0

    return R


def pde_residual_z(
    theta_prev,
    theta,
    theta_next,
    intensity,
    *,
    du,
    dv,
    dz,
    gamma_z,
    b,
    bi,
    mobility=1.0,
    xp,
):
    """
    z-coupled steady PDE residual.
    """

    R = theta_operator_z(
        theta_prev,
        theta,
        theta_next,
        intensity,
        du=du,
        dv=dv,
        dz=dz,
        gamma_z=gamma_z,
        b=b,
        bi=bi,
        xp=xp,
    ) / mobility

    R[0, :] = 0
    R[-1, :] = 0

    return R


def cn_residual(
    theta_new,
    theta_old,
    intensity,
    *,
    dt,
    du,
    dv,
    b,
    bi,
    mobility=1.0,
    xp,
):
    """
    Residual of the Crank-Nicolson equation:

        (theta_new - theta_old)/dt
        =
        0.5 * [F(theta_old) + F(theta_new)] / mobility
    """

    F_old = theta_operator(
        theta_old,
        intensity,
        du=du,
        dv=dv,
        b=b,
        bi=bi,
        xp=xp,
    )

    F_new = theta_operator(
        theta_new,
        intensity,
        du=du,
        dv=dv,
        b=b,
        bi=bi,
        xp=xp,
    )

    R = (
        (theta_new - theta_old) / dt
        - 0.5 * (F_old + F_new) / mobility
    )

    R[0, :] = 0
    R[-1, :] = 0

    return R


def residual_stats(R, *, xp):
    """
    Standard residual statistics.
    """

    Rint = R[1:-1, :]

    return {
        "max_full": float(xp.max(xp.abs(R))),
        "rms_full": float(xp.sqrt(xp.mean(R * R))),
        "max_interior": float(xp.max(xp.abs(Rint))),
        "rms_interior": float(xp.sqrt(xp.mean(Rint * Rint))),
    }