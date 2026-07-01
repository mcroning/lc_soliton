"""
lc_soliton.optics_engine.operators

Low-level optical operators.

This module contains no workflow logic.  It only builds refractive-index
fields, Fourier grids, and propagation kernels.
"""

from __future__ import annotations


def neff_from_theta(theta, *, ne, no, xp):
    """
    Extraordinary effective index for director angle theta.

        n_eff = ne*no / sqrt((ne cos theta)^2 + (no sin theta)^2)
    """

    return (ne * no) / xp.sqrt((ne * xp.cos(theta)) ** 2 + (no * xp.sin(theta)) ** 2)


def delta_n_from_theta(theta, *, ne, no, n_ref=None, xp):
    """
    Refractive-index perturbation from theta.

    If n_ref is None, subtract the median n_eff so the phase screen is
    centered.  This matches the historical propagation convention.
    """

    neff = neff_from_theta(theta, ne=ne, no=no, xp=xp)

    if n_ref is None:
        n_ref = xp.median(neff)

    return neff - n_ref


def transverse_k_grids(Nx, Ny, *, dx, dy, xp):
    """
    Angular spatial-frequency grids kx, ky for FFT propagation.

    dx and dy are physical spacings in the same length units as wavelength.
    """

    fx = xp.fft.fftfreq(int(Nx), d=dx)
    fy = xp.fft.fftfreq(int(Ny), d=dy)

    kx = 2 * xp.pi * fx[:, None]
    ky = 2 * xp.pi * fy[None, :]

    return kx, ky


def angular_spectrum_kernel(
    Nx,
    Ny,
    *,
    dx,
    dy,
    wavelength,
    dz,
    n0,
    xp,
    evanescent="clip",
):
    """
    Nonparaxial angular-spectrum propagation kernel.

    Coordinates dx, dy, dz, wavelength must be in the same physical units.

    H = exp(i kz dz), kz = sqrt((n0 k0)^2 - kx^2 - ky^2)

    evanescent:
        "clip"  : evanescent kz is clipped to zero phase advance
        "decay" : evanescent modes get imaginary kz decay
    """

    k0 = 2 * xp.pi / wavelength
    k = n0 * k0

    kx, ky = transverse_k_grids(Nx, Ny, dx=dx, dy=dy, xp=xp)
    q2 = kx * kx + ky * ky
    kz2 = k * k - q2

    if evanescent == "decay":
        kz = xp.sqrt(kz2.astype(complex))
    else:
        kz = xp.sqrt(xp.maximum(kz2, 0))

    return xp.exp(1j * kz * dz)
