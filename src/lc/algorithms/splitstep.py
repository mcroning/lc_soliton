"""Multichannel split-step optical propagation kernels.

This module is a numerical black box. It consumes channel-stack fields,
theta slices, kernels, and optical coefficients. It knows nothing about beam
setup, LC-cell experiments, GUI state, files, continuation, or diagnostics.

Field convention
----------------
Optical fields are always channel stacks:

    A.shape == (Nch, Nx, Ny)

A single beam is represented by ``Nch=1``. There is no scalar-field special
case inside this module.

The split-step ordering follows the trusted runner convention for one slice:

    for substep:
        apply nonlinear LC phase from theta
        apply linear Fourier hop

The caller decides how kernels, substeps, wavelengths, and reference indices
are constructed.
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


def as_channel_stack(A: Array, *, xp: Any | None = None) -> Array:
    """Return ``A`` as a channel stack.

    This helper is intended for adapters/tests. Core algorithms should already
    pass channel stacks.
    """

    xp = _xp_from(A, xp=xp)
    if A.ndim == 3:
        return A
    if A.ndim == 2:
        return A[None, :, :]
    raise ValueError("A must have shape (Nch, Nx, Ny) or (Nx, Ny)")


def channel_intensities(A: Array, *, xp: Any | None = None) -> Array:
    """Return per-channel intensities ``|A_c|^2``."""

    xp = _xp_from(A, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    return xp.abs(A) ** 2


def total_intensity(
    A: Array,
    *,
    coherent: bool = False,
    xp: Any | None = None,
) -> Array:
    """Return total intensity from a channel stack.

    Parameters
    ----------
    coherent
        If False, return ``sum_c |A_c|^2``. If True, return
        ``|sum_c A_c|^2``.
    """

    xp = _xp_from(A, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")

    if coherent:
        return xp.abs(xp.sum(A, axis=0)) ** 2
    return xp.sum(channel_intensities(A, xp=xp), axis=0)


def weighted_theta_intensity(
    A: Array,
    weights: Array | None = None,
    *,
    coherent: bool = False,
    xp: Any | None = None,
) -> Array:
    """Return effective plain intensity for theta algorithms.

    The theta algorithms expect plain intensity and multiply by ``bi``
    internally. For equal optical couplings, pass ``weights=None``. For
    channel-dependent coupling, pass weights with shape ``(Nch,)`` such as
    ``bi_c / bi_ref``.
    """

    xp = _xp_from(A, weights, xp=xp)

    if weights is None:
        return total_intensity(A, coherent=coherent, xp=xp)

    if coherent:
        raise ValueError("weighted coherent intensity is ambiguous; combine channels upstream")

    I_c = channel_intensities(A, xp=xp)
    if weights.ndim != 1 or weights.shape[0] != I_c.shape[0]:
        raise ValueError("weights must have shape (Nch,)")
    return xp.sum(weights[:, None, None] * I_c, axis=0)


def neff_from_theta(theta: Array, *, ne: float, no: float, xp: Any | None = None) -> Array:
    """Extraordinary-ray effective index from director angle theta."""

    xp = _xp_from(theta, xp=xp)
    c = xp.cos(theta)
    s = xp.sin(theta)
    return (float(ne) * float(no)) / xp.sqrt((float(ne) * c) ** 2 + (float(no) * s) ** 2)


def linear_kernel(
    fxy2: Array,
    *,
    dz: float,
    wavelength: float,
    n_ref: float,
    xp: Any | None = None,
) -> Array:
    """Return paraxial Fourier-space linear propagator.

    ``fxy2`` has shape ``(Nx, Ny)`` and stores ``fx^2 + fy^2`` in micron units.
    """

    xp = _xp_from(fxy2, xp=xp)
    return xp.exp(-1j * np.pi * float(dz) * float(wavelength) * fxy2 / float(n_ref))


def hop_linear(A: Array, kernel: Array, *, xp: Any | None = None) -> Array:
    """Apply one linear Fourier hop to every channel."""

    xp = _xp_from(A, kernel, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    return xp.fft.ifft2(
        xp.fft.fft2(A, axes=(-2, -1)) * kernel[None, :, :],
        axes=(-2, -1),
    )


def hop_linear_inplace(A: Array, kernel: Array, *, xp: Any | None = None) -> Array:
    """Apply one linear Fourier hop in place and return ``A``."""

    A[...] = hop_linear(A, kernel, xp=xp)
    return A


def nonlinear_phase(
    theta: Array,
    *,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    xp: Any | None = None,
) -> Array:
    """Return LC nonlinear phase factor for one propagation step."""

    xp = _xp_from(theta, xp=xp)
    k0 = 2.0 * np.pi / float(wavelength)
    dn = neff_from_theta(theta, ne=ne, no=no, xp=xp) - float(n_ref)
    return xp.exp(1j * k0 * float(dz) * dn)


def apply_nonlinear_phase_inplace(A: Array, phase: Array, *, xp: Any | None = None) -> Array:
    """Multiply all channels by a common transverse phase."""

    xp = _xp_from(A, phase, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    A[...] = A * phase[None, :, :]
    return A


def advance_slice(
    A: Array,
    theta: Array,
    *,
    kernel: Array,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    Nsub: int = 1,
    xp: Any | None = None,
) -> Array:
    """Advance a channel stack through one z slice.

    The field ``A`` is updated in place and also returned.
    """

    xp = _xp_from(A, theta, kernel, xp=xp)
    if int(Nsub) < 1:
        raise ValueError("Nsub must be >= 1")

    dz_sub = float(dz) / int(Nsub)

    for _ in range(int(Nsub)):
        phase = nonlinear_phase(
            theta,
            dz=dz_sub,
            wavelength=wavelength,
            n_ref=n_ref,
            ne=ne,
            no=no,
            xp=xp,
        )
        apply_nonlinear_phase_inplace(A, phase, xp=xp)
        hop_linear_inplace(A, kernel, xp=xp)

    return A


def advance_slice_with_midintensity(
    A: Array,
    theta: Array,
    *,
    kernel: Array,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    Nsub: int = 1,
    coherent: bool = False,
    theta_weights: Array | None = None,
    xp: Any | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Advance one z slice and return midpoint intensity.

    Returns
    -------
    A
        Updated channel stack.
    I_before
        Effective plain theta intensity before propagation.
    I_after
        Effective plain theta intensity after propagation.
    I_mid
        ``0.5 * (I_before + I_after)``.
    """

    xp = _xp_from(A, theta, kernel, theta_weights, xp=xp)

    I_before = weighted_theta_intensity(
        A,
        theta_weights,
        coherent=coherent,
        xp=xp,
    )

    advance_slice(
        A,
        theta,
        kernel=kernel,
        dz=dz,
        wavelength=wavelength,
        n_ref=n_ref,
        ne=ne,
        no=no,
        Nsub=Nsub,
        xp=xp,
    )

    I_after = weighted_theta_intensity(
        A,
        theta_weights,
        coherent=coherent,
        xp=xp,
    )

    I_mid = 0.5 * (I_before + I_after)
    return A, I_before, I_after, I_mid


__all__ = [
    "as_channel_stack",
    "channel_intensities",
    "total_intensity",
    "weighted_theta_intensity",
    "neff_from_theta",
    "linear_kernel",
    "hop_linear",
    "hop_linear_inplace",
    "nonlinear_phase",
    "apply_nonlinear_phase_inplace",
    "advance_slice",
    "advance_slice_with_midintensity",
]
