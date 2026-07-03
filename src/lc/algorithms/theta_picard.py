"""Picard-corrected Crank-Nicolson theta steps.

This module is a small numerical black box. It consumes arrays and numerical
coefficients only. It knows nothing about liquid-crystal experiments, beam
geometry, GUI state, files, continuation, or diagnostics.

The intensity arguments are plain optical intensities; this module applies
``bi`` internally through ``(b + bi*I) sin(2 theta)``.
"""

from __future__ import annotations

from typing import Any

from .thomas import solve_const_offdiag_batched

import numpy as np

from .theta_cn import (
    cn_predictor_step,
    laplacian_dirichletx_periody,
    solve_dirichletx_periody,
    theta_drive,
)

Array = Any


def _xp_from(*arrays: Array, xp: Any | None = None):
    """Return NumPy/CuPy-like array module, preferring explicit ``xp``."""
    if xp is not None:
        return xp
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy as cp  # type: ignore

            return cp
    return np


def _to_float(x: Any, xp: Any) -> float:
    """Backend-independent scalar conversion."""
    if hasattr(xp, "asnumpy"):
        return float(xp.asnumpy(x))
    return float(x)


def cn_trapezoid_picard_step(
    theta: Array,
    intensity_old: Array,
    intensity_picard: Array,
    *,
    b: float,
    bi: float,
    dt: float,
    mobility: float,
    dx: float,
    dy: float,
    s: Any,
    off: Any,
    diag: Array,
    max_iter: int = 4,
    tol_update: float = 1e-6,
    clamp: tuple[float, float] | None = None,
    tridiag_solver=solve_const_offdiag_batched,
    xp: Any | None = None,
) -> Array:
    """Trapezoid/Picard corrected CN theta step.

    Parameters
    ----------
    theta
        Previous theta slice, shape ``(Nx, Ny)``.
    intensity_old
        Intensity used for the old nonlinear term.
    intensity_picard
        Intensity used for the Picard-corrected new nonlinear term.
    b, bi
        Director-drive coefficients in ``(b + bi I) sin(2 theta)``.
    s, off, diag
        Prepared CN operator coefficients from ``theta_cn.prepare_cn_operator``.

    Notes
    -----
    This preserves the trusted runner pattern:

    1. build the trapezoid RHS base from ``theta_n`` and ``N(theta_n)``;
    2. initialize the new-time guess with the predictor step;
    3. iterate ``N(theta_guess)`` through the same implicit solve.
    """

    xp = _xp_from(theta, intensity_old, intensity_picard, diag, xp=xp)
    dtype = theta.dtype

    I_old = intensity_old.astype(dtype, copy=False)
    I_pic = intensity_picard.astype(dtype, copy=False)

    lap = laplacian_dirichletx_periody(theta, dx, dy, xp=xp).astype(dtype, copy=False)

    half_dt_over_m = np.dtype(dtype).type(0.5 * float(dt) / float(mobility))
    N_old = theta_drive(theta, I_old, b=b, bi=bi, xp=xp).astype(dtype, copy=False)

    rhs_base = theta + s * lap + half_dt_over_m * N_old
    bc = theta[0, 0]
    rhs_base[0, :] = bc
    rhs_base[-1, :] = bc

    guess = cn_predictor_step(
        theta,
        I_old,
        b=b,
        bi=bi,
        dt=dt,
        mobility=mobility,
        dx=dx,
        dy=dy,
        s=s,
        off=off,
        diag=diag,
        clamp=clamp,
        tridiag_solver=tridiag_solver,
        xp=xp,
    )

    for _ in range(int(max_iter)):
        N_guess = theta_drive(guess, I_pic, b=b, bi=bi, xp=xp).astype(dtype, copy=False)

        rhs = rhs_base + half_dt_over_m * N_guess
        rhs[0, :] = bc
        rhs[-1, :] = bc

        new = solve_dirichletx_periody(rhs, off=off, diag=diag, theta_bc=float(bc), tridiag_solver=tridiag_solver, xp=xp)

        if clamp is not None:
            new = xp.clip(new, clamp[0], clamp[1])

        update = new - guess
        rms_update = _to_float(xp.sqrt(xp.mean(update * update)), xp)
        guess = new

        if rms_update < float(tol_update):
            break

    return guess


__all__ = ["cn_trapezoid_picard_step"]
