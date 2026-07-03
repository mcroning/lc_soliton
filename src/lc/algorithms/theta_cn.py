"""Crank-Nicolson / implicit x-solve kernels for theta.

This module is a small numerical black box. It consumes arrays and numerical
coefficients only. It knows nothing about liquid crystals as an experiment,
beam geometry, GUI state, files, continuation, or diagnostics.

Boundary convention
-------------------
* x: Dirichlet rows at index 0 and -1.
* y: periodic, diagonalized by FFT along axis 1.

Array convention
----------------
2-D theta arrays are shaped ``(Nx, Ny)``.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from .fft_y import fft_y, lam_y_periodic_second_diff, real_ifft_y
from .thomas import solve_const_offdiag_batched

Array = Any
TridiagSolver = Callable[..., Array]


def _xp_from(*arrays: Array, xp: Any | None = None):
    """Return NumPy/CuPy-like array module, preferring explicit ``xp``."""
    if xp is not None:
        return xp
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy as cp  # type: ignore

            return cp
    return np


def _dtype_type(dtype: Any | None):
    """Return scalar constructor for dtype, or None."""
    if dtype is None:
        return None
    return np.dtype(dtype).type


def _cast(value: Any, dtype: Any | None):
    """Cast scalar value if dtype is provided."""
    dtype_type = _dtype_type(dtype)
    return value if dtype_type is None else dtype_type(value)


def prepare_ie_operator(
    *,
    dt: float,
    mobility: float,
    dx: float,
    dy: float,
    Ny: int,
    xp: Any = np,
    dtype: Any | None = None,
) -> Tuple[Any, Any, Array, Array]:
    """Prepare implicit-Euler operator after diagonalizing periodic y."""

    dtype = np.float32 if dtype is None else dtype
    a = float(dt) / float(mobility)

    lam_y = lam_y_periodic_second_diff(Ny, dy, xp=xp, dtype=dtype)
    inv_dx2 = 1.0 / (float(dx) * float(dx))

    off = _cast(-a * inv_dx2, dtype)
    diag = 1.0 + 2.0 * a * inv_dx2 - a * lam_y
    diag = diag.astype(dtype, copy=False)

    return _cast(a, dtype), off, diag, lam_y


def prepare_cn_operator(
    *,
    dt: float,
    mobility: float,
    dx: float,
    dy: float,
    Ny: int,
    xp: Any = np,
    dtype: Any | None = None,
) -> Tuple[Any, Any, Array, Array]:
    """Prepare Crank-Nicolson operator after diagonalizing periodic y."""

    dtype = np.float32 if dtype is None else dtype
    s = float(dt) / (2.0 * float(mobility))

    lam_y = lam_y_periodic_second_diff(Ny, dy, xp=xp, dtype=dtype)
    inv_dx2 = 1.0 / (float(dx) * float(dx))

    off = _cast(-s * inv_dx2, dtype)
    diag = 1.0 + 2.0 * s * inv_dx2 - s * lam_y
    diag = diag.astype(dtype, copy=False)

    return _cast(s, dtype), off, diag, lam_y


def laplacian_dirichletx_periody(
    theta: Array,
    dx: float,
    dy: float,
    *,
    xp: Any | None = None,
) -> Array:
    """2-D Laplacian with Dirichlet x rows excluded and periodic y."""

    xp = _xp_from(theta, xp=xp)
    lap = xp.zeros_like(theta)

    dx2 = float(dx) * float(dx)
    dy2 = float(dy) * float(dy)

    yp = xp.roll(theta, -1, axis=1)
    ym = xp.roll(theta, +1, axis=1)

    lap[1:-1, :] = (
        (theta[2:, :] - 2.0 * theta[1:-1, :] + theta[:-2, :]) / dx2
        + (yp[1:-1, :] - 2.0 * theta[1:-1, :] + ym[1:-1, :]) / dy2
    )

    return lap


def _call_tridiag_solver(
    solver: TridiagSolver,
    lower: Any,
    diag: Array,
    upper: Any,
    rhs: Array,
    *,
    xp: Any,
) -> Array:
    """Call either the reference or fast tridiagonal solver.

    The reference solver accepts ``xp=``. The fast solver intentionally does
    not require it. This wrapper keeps ``solve_dirichletx_periody`` simple.
    """
    try:
        return solver(lower, diag, upper, rhs, xp=xp)
    except TypeError as e:
        if "unexpected keyword argument 'xp'" not in str(e):
            raise
        return solver(lower, diag, upper, rhs)


def solve_dirichletx_periody(
    rhs: Array,
    *,
    off: Any,
    diag: Array,
    theta_bc: float | None = None,
    tridiag_solver: TridiagSolver = solve_const_offdiag_batched,
    xp: Any | None = None,
) -> Array:
    """Solve Helmholtz-like system with Dirichlet x and periodic y.

    Parameters
    ----------
    rhs
        Right-hand side, shape ``(Nx, Ny)``.
    off
        Scalar lower/upper x off-diagonal coefficient after y FFT.
    diag
        Diagonal coefficient for each y Fourier mode, shape ``(Ny,)``.
    theta_bc
        Boundary value on the x rows. If omitted, ``rhs[0, 0]`` is used.
    tridiag_solver
        Batched tridiagonal solver. Defaults to the readable reference
        ``solve_const_offdiag_batched``. GPU workflows may pass
        ``solve_const_offdiag_batched_fast``.
    """

    xp = _xp_from(rhs, diag, xp=xp)

    if rhs.ndim != 2:
        raise ValueError("rhs must have shape (Nx, Ny)")
    if diag.ndim != 1:
        raise ValueError("diag must have shape (Ny,)")
    if rhs.shape[1] != diag.shape[0]:
        raise ValueError("rhs.shape[1] must equal diag.shape[0]")

    rhs2 = rhs.astype(rhs.dtype, copy=True)

    bc = rhs2[0, 0] if theta_bc is None else xp.asarray(theta_bc, dtype=rhs2.dtype)
    rhs2[0, :] = bc
    rhs2[-1, :] = bc

    rhs_hat = fft_y(rhs2, xp=xp)
    complex_dtype = xp.complex64 if rhs2.dtype == xp.float32 else xp.complex128
    rhs_hat = rhs_hat.astype(complex_dtype, copy=False)

    # Thomas solver expects batched systems shaped (batch=Ny, n=Nx-2).
    rhs_batch = rhs_hat[1:-1, :].T.copy()

    Ny = rhs2.shape[1]
    bc_hat = xp.fft.fft(xp.full((Ny,), bc, dtype=rhs2.dtype)).astype(complex_dtype, copy=False)

    rhs_batch[:, 0] -= off * bc_hat
    rhs_batch[:, -1] -= off * bc_hat

    sol_batch = _call_tridiag_solver(tridiag_solver, off, diag, off, rhs_batch, xp=xp)

    theta_hat = xp.zeros_like(rhs_hat)
    theta_hat[1:-1, :] = sol_batch.T

    out = real_ifft_y(theta_hat, dtype=rhs2.dtype, xp=xp)
    out[0, :] = bc
    out[-1, :] = bc

    return out


def theta_drive(
    theta: Array,
    intensity: Array,
    *,
    b: float,
    bi: float,
    xp: Any | None = None,
) -> Array:
    """Nonlinear director drive ``(b + bi I) sin(2 theta)``."""

    xp = _xp_from(theta, intensity, xp=xp)
    return (float(b) + float(bi) * intensity) * xp.sin(2.0 * theta)


def cn_predictor_step(
    theta: Array,
    intensity: Array,
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
    clamp: tuple[float, float] | None = None,
    tridiag_solver: TridiagSolver = solve_const_offdiag_batched,
    xp: Any | None = None,
) -> Array:
    """One explicit-nonlinearity Crank-Nicolson predictor step."""

    xp = _xp_from(theta, intensity, diag, xp=xp)
    dtype = theta.dtype

    I = intensity.astype(dtype, copy=False)
    lap = laplacian_dirichletx_periody(theta, dx, dy, xp=xp).astype(dtype, copy=False)
    drive = theta_drive(theta, I, b=b, bi=bi, xp=xp).astype(dtype, copy=False)

    dt_over_m = np.dtype(dtype).type(float(dt) / float(mobility))
    rhs = theta + s * lap + dt_over_m * drive

    bc = theta[0, 0]
    rhs[0, :] = bc
    rhs[-1, :] = bc

    out = solve_dirichletx_periody(
        rhs,
        off=off,
        diag=diag,
        theta_bc=float(bc),
        tridiag_solver=tridiag_solver,
        xp=xp,
    )

    if clamp is not None:
        out = xp.clip(out, clamp[0], clamp[1])

    return out


__all__ = [
    "TridiagSolver",
    "prepare_ie_operator",
    "prepare_cn_operator",
    "laplacian_dirichletx_periody",
    "solve_dirichletx_periody",
    "theta_drive",
    "cn_predictor_step",
]
