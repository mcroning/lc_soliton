"""Build multichannel launch fields from beam specifications.

This module converts human-facing ``BeamExperiment`` objects into optical
channel stacks consumed by the algorithms.

It does not propagate fields, build FFT kernels, solve theta, or manage
products.

Array convention
----------------
Launch fields have shape ``(Nch, Nx, Ny)``.
A single beam is still a one-channel stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .beam import BeamExperiment
from ..numerics.grid import RuntimeGrid

Array = Any


@dataclass(frozen=True)
class LaunchResult:
    """Prepared multichannel optical launch."""

    A0: Array
    theta_weights: Array
    wavelengths_um: Array
    coherence: str

    def summary(self) -> dict:
        return {
            "Nch": int(self.A0.shape[0]),
            "coherence": self.coherence,
            "wavelengths_um": [float(x) for x in np.asarray(_to_numpy(self.wavelengths_um)).ravel()],
            "theta_weights": [float(x) for x in np.asarray(_to_numpy(self.theta_weights)).ravel()],
        }


def _to_numpy(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def gaussian_channel(ch, grid: RuntimeGrid, *, complex_dtype: Any) -> Array:
    """Return one normalized Gaussian channel on ``grid``."""

    xp = grid.xp
    X = grid.x_um[:, None]
    Y = grid.y_um[None, :]

    amp = xp.exp(
        -(((X - float(ch.x0_um)) / float(ch.waist_x_um)) ** 2)
        -(((Y - float(ch.y0_um)) / float(ch.waist_y_um)) ** 2)
    )

    phase = float(ch.phase_rad)
    if ch.tilt_x_rad_per_um or ch.tilt_y_rad_per_um or phase:
        amp = amp * xp.exp(
            1j
            * (
                float(ch.tilt_x_rad_per_um) * X
                + float(ch.tilt_y_rad_per_um) * Y
                + phase
            )
        )

    amp = amp.astype(complex_dtype, copy=False)

    # Normalize so integral |A|^2 dx dy equals channel power.
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    p0 = xp.sum(xp.abs(amp) ** 2) * dxdy
    if float(ch.power) == 0.0:
        amp = xp.zeros_like(amp)
    else:
        amp = amp * xp.sqrt(float(ch.power) / p0)

    return amp.astype(complex_dtype, copy=False)


def build_launch(
    beams: BeamExperiment,
    grid: RuntimeGrid,
    *,
    complex_dtype: Any = np.complex64,
) -> LaunchResult:
    """Build ``A0`` channel stack from a ``BeamExperiment``."""

    beams.validate()
    xp = grid.xp

    fields = [
        gaussian_channel(ch, grid, complex_dtype=complex_dtype)
        for ch in beams.channels
    ]

    A0 = xp.stack(fields, axis=0).astype(complex_dtype, copy=False)

    theta_weights = xp.asarray(
        [float(ch.theta_weight) for ch in beams.channels],
        dtype=grid.real_dtype,
    )
    wavelengths_um = xp.asarray(
        [float(ch.wavelength_um) for ch in beams.channels],
        dtype=grid.real_dtype,
    )

    return LaunchResult(
        A0=A0,
        theta_weights=theta_weights,
        wavelengths_um=wavelengths_um,
        coherence=beams.coherence,
    )


def total_power(A0: Array, grid: RuntimeGrid) -> float:
    """Return integral sum_c |A_c|^2 dx dy."""
    xp = grid.xp
    p = xp.sum(xp.abs(A0) ** 2) * float(grid.dx_um) * float(grid.dy_um)
    return float(_to_numpy(p))


__all__ = [
    "Array",
    "LaunchResult",
    "gaussian_channel",
    "build_launch",
    "total_power",
]
