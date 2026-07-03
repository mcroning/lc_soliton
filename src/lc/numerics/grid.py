"""Numerical grid specifications and builders.

This module converts human-facing numerical choices into computational grid
arrays. It does not know about LC materials, beams, workflows, products, or
algorithms.

Human-facing units
------------------
* transverse and propagation lengths: microns
* theta PDE x-coordinate uses the trusted dimensionless coordinate u = 2x/d

Array convention
----------------
2-D fields are shaped ``(Nx, Ny)``.
Optical channel stacks are shaped ``(Nch, Nx, Ny)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

Array = Any


@dataclass(frozen=True)
class GridSpec:
    """Human-facing transverse/z discretization choices."""

    Nx: int = 256
    Ny: int = 256
    dz_um: float = 5.0
    x_aperture_um: float = 75.0
    y_aperture_um: float = 100.0
    z_length_um: float = 3000.0

    def validate(self) -> None:
        if self.Nx <= 1 or self.Ny <= 1:
            raise ValueError("Nx and Ny must be > 1")
        if self.dz_um <= 0.0:
            raise ValueError("dz_um must be positive")
        if self.x_aperture_um <= 0.0:
            raise ValueError("x_aperture_um must be positive")
        if self.y_aperture_um <= 0.0:
            raise ValueError("y_aperture_um must be positive")
        if self.z_length_um <= 0.0:
            raise ValueError("z_length_um must be positive")


@dataclass(frozen=True)
class TimeSpec:
    """Physical/pseudo-time discretization choices."""

    dt: float = 7.5e-4
    Nt: int = 20

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.Nt < 0:
            raise ValueError("Nt must be nonnegative")


@dataclass(frozen=True)
class RuntimeGrid:
    """Backend arrays and derived spacings for algorithms/builders."""

    spec: GridSpec
    xp: Any
    real_dtype: Any

    Nx: int
    Ny: int
    Nz: int

    dx_um: float
    dy_um: float
    dz_um: float

    du: float
    dv: float

    x_um: Array
    y_um: Array
    fx_um: Array
    fy_um: Array
    fxy2_um: Array

    def summary(self) -> dict[str, float | int | str]:
        return {
            "Nx": self.Nx,
            "Ny": self.Ny,
            "Nz": self.Nz,
            "dx_um": float(self.dx_um),
            "dy_um": float(self.dy_um),
            "dz_um": float(self.dz_um),
            "du": float(self.du),
            "dv": float(self.dv),
            "x_aperture_um": float(self.spec.x_aperture_um),
            "y_aperture_um": float(self.spec.y_aperture_um),
            "z_length_um": float(self.spec.z_length_um),
            "real_dtype": str(np.dtype(self.real_dtype)),
        }


def _round_nz(z_length_um: float, dz_um: float) -> int:
    """Return Nz from z length and dz, requiring at least one slice."""
    return max(1, int(round(float(z_length_um) / float(dz_um))))


def make_grid(
    spec: GridSpec,
    *,
    xp: Any = np,
    real_dtype: Any = np.float32,
) -> RuntimeGrid:
    """Build a runtime grid from ``GridSpec``.

    ``x_um`` and ``y_um`` are cell-centered coordinates. ``fxy2_um`` stores
    ``fx^2 + fy^2`` for optical Fourier propagation.
    """

    spec.validate()

    Nx = int(spec.Nx)
    Ny = int(spec.Ny)
    Nz = _round_nz(spec.z_length_um, spec.dz_um)

    dx_um = float(spec.x_aperture_um) / Nx
    dy_um = float(spec.y_aperture_um) / Ny
    dz_um = float(spec.dz_um)

    x_um = ((xp.arange(Nx, dtype=real_dtype) - Nx / 2) * dx_um + 0.5 * dx_um).astype(real_dtype, copy=False)
    y_um = ((xp.arange(Ny, dtype=real_dtype) - Ny / 2) * dy_um + 0.5 * dy_um).astype(real_dtype, copy=False)

    fx_um = xp.fft.fftfreq(Nx, d=dx_um).astype(real_dtype, copy=False)
    fy_um = xp.fft.fftfreq(Ny, d=dy_um).astype(real_dtype, copy=False)
    fxy2_um = (fx_um[:, None] ** 2 + fy_um[None, :] ** 2).astype(real_dtype, copy=False)

    # Trusted LC theta coordinates: u = 2x/d, with d = x aperture.
    du = 2.0 / (Nx - 1)
    dv = du * (dy_um / dx_um)

    return RuntimeGrid(
        spec=spec,
        xp=xp,
        real_dtype=real_dtype,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx_um=dx_um,
        dy_um=dy_um,
        dz_um=dz_um,
        du=du,
        dv=dv,
        x_um=x_um,
        y_um=y_um,
        fx_um=fx_um,
        fy_um=fy_um,
        fxy2_um=fxy2_um,
    )


def from_cell(
    *,
    Nx: int,
    Ny: int,
    dz_um: float,
    thickness_um: float,
    y_aperture_um: float,
    interaction_length_um: float,
) -> GridSpec:
    """Convenience constructor from LC-cell geometry."""
    return GridSpec(
        Nx=Nx,
        Ny=Ny,
        dz_um=dz_um,
        x_aperture_um=thickness_um,
        y_aperture_um=y_aperture_um,
        z_length_um=interaction_length_um,
    )


__all__ = [
    "Array",
    "GridSpec",
    "TimeSpec",
    "RuntimeGrid",
    "make_grid",
    "from_cell",
]
