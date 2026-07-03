"""Batched Thomas solvers for tridiagonal systems.

This module is deliberately small and experiment-free. It knows nothing about
liquid crystals, optics, beams, grids, files, or GUIs. It solves tridiagonal
linear systems supplied as arrays/scalars.

The main function, ``solve_const_offdiag_batched``, matches the convention used
by the trusted LC runner after diagonalizing the periodic y direction by FFT:
one tridiagonal x-system per y-Fourier mode.

Expected shapes
---------------
diag
    Shape ``(batch,)``. One diagonal value for each independent system.
rhs
    Shape ``(batch, n)``. One right-hand side row per independent system.

The lower and upper off-diagonals are scalar constants shared by all systems.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


def solve_const_offdiag_batched(
    lower: Any,
    diag: Array,
    upper: Any,
    rhs: Array,
    *,
    xp: Any | None = None,
) -> Array:
    """Solve batched tridiagonal systems with scalar off-diagonals.

    Parameters
    ----------
    lower, upper
        Scalar lower/upper off-diagonal values.
    diag
        Array with shape ``(batch,)``. ``diag[i]`` is the diagonal value for
        system ``i``.
    rhs
        Array with shape ``(batch, n)``.

    Returns
    -------
    Array
        Solution with the same shape and dtype as ``rhs``.
    """

    xp = _xp_from(diag, rhs, xp=xp)

    if rhs.ndim != 2:
        raise ValueError("rhs must have shape (batch, n)")
    if diag.ndim != 1:
        raise ValueError("diag must have shape (batch,)")
    if rhs.shape[0] != diag.shape[0]:
        raise ValueError("rhs.shape[0] must equal diag.shape[0]")

    batch, n = rhs.shape
    if n == 0:
        return xp.empty_like(rhs)

    dtype = rhs.dtype
    b = diag.astype(dtype, copy=False)
    a = xp.asarray(lower, dtype=dtype)
    c = xp.asarray(upper, dtype=dtype)

    cprime = xp.empty((batch, n), dtype=dtype)
    dprime = xp.empty((batch, n), dtype=dtype)

    denom = b
    cprime[:, 0] = c / denom if n > 1 else xp.zeros((batch,), dtype=dtype)
    dprime[:, 0] = rhs[:, 0] / denom

    for j in range(1, n):
        denom = b - a * cprime[:, j - 1]
        cprime[:, j] = c / denom if j < n - 1 else xp.zeros((batch,), dtype=dtype)
        dprime[:, j] = (rhs[:, j] - a * dprime[:, j - 1]) / denom

    x = xp.empty_like(rhs)
    x[:, -1] = dprime[:, -1]
    for j in range(n - 2, -1, -1):
        x[:, j] = dprime[:, j] - cprime[:, j] * x[:, j + 1]

    return x


__all__ = ["solve_const_offdiag_batched"]
