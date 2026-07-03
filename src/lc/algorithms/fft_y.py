"""FFT helpers for periodic-y / Dirichlet-x solvers.

This module is deliberately small and experiment-free. It provides the
Fourier-y pieces used by theta CN/IE solvers after the periodic y direction
has been diagonalized.

Array convention
----------------
2-D theta arrays are shaped ``(Nx, Ny)`` and y is axis 1.
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


def _float_dtype(dtype: Any | None):
    """Return a NumPy scalar dtype type for scalar construction."""
    return np.dtype(np.float32 if dtype is None else dtype).type


def lam_y_periodic_second_diff(
    Ny: int,
    dy: float,
    *,
    xp: Any = np,
    dtype: Any | None = None,
) -> Array:
    """Eigenvalues of the periodic second-difference operator in y.

    For grid index mode ``m``,

        lambda_m = -4 sin^2(pi m / Ny) / dy^2

    Parameters
    ----------
    Ny
        Number of y grid points.
    dy
        y grid spacing in the same units used by the theta PDE.
    xp
        NumPy/CuPy-like backend.
    dtype
        Real floating dtype for the returned eigenvalues.
    """

    if int(Ny) <= 0:
        raise ValueError("Ny must be positive")
    if float(dy) == 0.0:
        raise ValueError("dy must be nonzero")

    dtype_type = _float_dtype(dtype)
    m = xp.arange(int(Ny), dtype=dtype_type)
    lam = -4.0 * xp.sin(xp.pi * m / int(Ny)) ** 2 / (float(dy) * float(dy))
    return lam.astype(dtype_type, copy=False)


def fft_y(a: Array, *, xp: Any | None = None) -> Array:
    """FFT along y, axis 1, for arrays shaped ``(Nx, Ny)``."""
    xp = _xp_from(a, xp=xp)
    return xp.fft.fft(a, axis=1)


def ifft_y(a_hat: Array, *, xp: Any | None = None) -> Array:
    """Inverse FFT along y, axis 1, for arrays shaped ``(Nx, Ny)``."""
    xp = _xp_from(a_hat, xp=xp)
    return xp.fft.ifft(a_hat, axis=1)


def real_ifft_y(
    a_hat: Array,
    *,
    dtype: Any | None = None,
    xp: Any | None = None,
) -> Array:
    """Inverse FFT along y and return its real part.

    Parameters
    ----------
    a_hat
        Fourier-y representation.
    dtype
        Optional real dtype for the returned array.
    """

    xp = _xp_from(a_hat, xp=xp)
    out = xp.fft.ifft(a_hat, axis=1).real
    return out.astype(dtype, copy=False) if dtype is not None else out


__all__ = [
    "lam_y_periodic_second_diff",
    "fft_y",
    "ifft_y",
    "real_ifft_y",
]
