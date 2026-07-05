"""Crank-Nicolson theta updates with z-neighbor coupling.

This module adds the one genuinely new numerical kernel needed by the strict
static z-stack workflow.  It is the clean equivalent of the legacy
``advance_theta_timestep_cn_*_zcoupled`` routines.

Equation form
-------------
For one z slice, the extra coupling contributes

    gamma_z * inv_dz2 * (theta_prev - 2 theta + theta_next)

which changes the implicit x/y CN solve by adding

    2 * dt/mobility * gamma_z * inv_dz2

to the effective diagonal, while the neighboring slices enter the RHS.
"""

from __future__ import annotations

from typing import Any

from .theta_cn import (
    laplacian_dirichletx_periody,
    solve_dirichletx_periody,
)

Array = Any


def _call_solve_dirichletx_periody(
    rhs: Array,
    *,
    off: Array,
    diag: Array,
    theta_bc: Any,
    tridiag_solver: Any | None,
    xp: Any,
) -> Array:
    """Call solve_dirichletx_periody across minor signature changes."""

    try:
        return solve_dirichletx_periody(
            rhs,
            off=off,
            diag=diag,
            theta_bc=theta_bc,
            tridiag_solver=tridiag_solver,
            xp=xp,
        )
    except TypeError:
        pass

    try:
        return solve_dirichletx_periody(
            rhs,
            off=off,
            diag=diag,
            theta_bc=theta_bc,
            tridiag_solver=tridiag_solver,
        )
    except TypeError:
        pass

    try:
        return solve_dirichletx_periody(
            rhs,
            off=off,
            diag=diag,
            theta_bc=theta_bc,
        )
    except TypeError:
        return solve_dirichletx_periody(rhs, off=off, diag=diag)


def _enforce(theta: Array, *, clamp: tuple[float, float] | None, theta_bc: Any, xp: Any) -> Array:
    """Apply clamp and x-Dirichlet boundary rows."""
    if clamp is not None:
        theta = xp.clip(theta, clamp[0], clamp[1])
    theta[0, :] = theta_bc
    theta[-1, :] = theta_bc
    return theta


def cn_predictor_step_zcoupled(
    theta_n: Array,
    *,
    dt: float,
    b: float,
    bi: float,
    Ixy: Array,
    theta_prev: Array,
    theta_next: Array,
    gamma_z: float,
    mobility: float,
    dx: float,
    dy: float,
    s: Any,
    off: Array,
    diag: Array,
    inv_dz2: float,
    clamp: tuple[float, float] | None = None,
    tridiag_solver: Any | None = None,
    xp: Any,
) -> Array:
    """One CN predictor step with z-neighbor coupling.

    ``dx`` and ``dy`` are the theta-PDE grid spacings, normally ``du`` and
    ``dv``.
    """

    dtype = theta_n.dtype
    theta_n = theta_n.astype(dtype, copy=False)
    Ixy = Ixy.astype(dtype, copy=False)
    tp = theta_prev.astype(dtype, copy=False)
    tn = theta_next.astype(dtype, copy=False)

    lap_n = laplacian_dirichletx_periody(theta_n, dx, dy, xp=xp).astype(dtype, copy=False)
    alpha = dtype.type(b) + dtype.type(bi) * Ixy
    drive_n = alpha * xp.sin(dtype.type(2.0) * theta_n)

    gam = dtype.type(gamma_z) * dtype.type(inv_dz2)
    dt_over_m = dtype.type(float(dt) / float(mobility))
    theta_bc = theta_n[0, 0].astype(dtype, copy=False)

    rhs = theta_n + s * lap_n + dt_over_m * (drive_n + gam * (tp + tn))
    rhs[0, :] = theta_bc
    rhs[-1, :] = theta_bc

    diag_eff = diag + dtype.type(2.0) * dt_over_m * gam

    out = _call_solve_dirichletx_periody(
        rhs,
        off=off,
        diag=diag_eff,
        theta_bc=theta_bc,
        tridiag_solver=tridiag_solver,
        xp=xp,
    )
    return _enforce(out, clamp=clamp, theta_bc=theta_bc, xp=xp)


def cn_trapezoid_picard_step_zcoupled(
    theta_n: Array,
    Ixy: Array,
    *,
    dt: float,
    b: float,
    bi: float,
    theta_prev: Array,
    theta_next: Array,
    gamma_z: float,
    mobility: float,
    dx: float,
    dy: float,
    s: Any,
    off: Array,
    diag: Array,
    inv_dz2: float,
    max_iter: int = 4,
    tol_update: float = 1e-6,
    clamp: tuple[float, float] | None = None,
    tridiag_solver: Any | None = None,
    xp: Any,
) -> Array:
    """Trapezoid/Picard CN step with z-neighbor coupling."""

    dtype = theta_n.dtype
    theta_n = theta_n.astype(dtype, copy=False)
    Ixy = Ixy.astype(dtype, copy=False)
    tp = theta_prev.astype(dtype, copy=False)
    tn = theta_next.astype(dtype, copy=False)

    lap_n = laplacian_dirichletx_periody(theta_n, dx, dy, xp=xp).astype(dtype, copy=False)
    alpha = dtype.type(b) + dtype.type(bi) * Ixy
    N_n = alpha * xp.sin(dtype.type(2.0) * theta_n)

    gam = dtype.type(gamma_z) * dtype.type(inv_dz2)
    dt_over_m = dtype.type(float(dt) / float(mobility))
    theta_bc = theta_n[0, 0].astype(dtype, copy=False)

    rhs_base = (
        theta_n
        + s * lap_n
        + dtype.type(0.5) * dt_over_m * N_n
        + dt_over_m * gam * (tp + tn)
    )
    rhs_base[0, :] = theta_bc
    rhs_base[-1, :] = theta_bc

    diag_eff = diag + dtype.type(2.0) * dt_over_m * gam

    theta_g = cn_predictor_step_zcoupled(
        theta_n,
        dt=dt,
        b=b,
        bi=bi,
        Ixy=Ixy,
        theta_prev=tp,
        theta_next=tn,
        gamma_z=gamma_z,
        mobility=mobility,
        dx=dx,
        dy=dy,
        s=s,
        off=off,
        diag=diag,
        inv_dz2=inv_dz2,
        clamp=clamp,
        tridiag_solver=tridiag_solver,
        xp=xp,
    )

    for _ in range(int(max_iter)):
        N_g = alpha * xp.sin(dtype.type(2.0) * theta_g)
        rhs = rhs_base + dtype.type(0.5) * dt_over_m * N_g
        rhs[0, :] = theta_bc
        rhs[-1, :] = theta_bc

        theta_new = _call_solve_dirichletx_periody(
            rhs,
            off=off,
            diag=diag_eff,
            theta_bc=theta_bc,
            tridiag_solver=tridiag_solver,
            xp=xp,
        )
        theta_new = _enforce(theta_new, clamp=clamp, theta_bc=theta_bc, xp=xp)

        d = theta_new - theta_g
        rms = float(xp.sqrt(xp.mean(d * d)))
        theta_g = theta_new
        if rms < float(tol_update):
            break

    return theta_g


__all__ = [
    "Array",
    "cn_predictor_step_zcoupled",
    "cn_trapezoid_picard_step_zcoupled",
]
