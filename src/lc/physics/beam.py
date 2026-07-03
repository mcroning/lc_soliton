"""Multichannel optical beam specifications.

This module contains human-facing beam parameters only. It does not build
arrays, grids, FFT kernels, propagation operators, or products.

A single beam is represented as one channel. Multiple beams are represented as
a channel list. Launch builders turn these specs into an A0 channel stack with
shape ``(Nch, Nx, Ny)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CoherenceMode = Literal["incoherent", "coherent"]


@dataclass(frozen=True)
class BeamChannel:
    """One optical input channel focused at the LC entrance."""

    name: str = "beam"
    wavelength_um: float = 0.633
    power: float = 1.0

    waist_x_um: float = 3.0
    waist_y_um: float = 3.0

    x0_um: float = 0.0
    y0_um: float = 0.0

    tilt_x_rad_per_um: float = 0.0
    tilt_y_rad_per_um: float = 0.0

    phase_rad: float = 0.0

    # Relative contribution to theta source, useful for future wavelengths or
    # polarizations. Builders may normalize these against a chosen reference.
    theta_weight: float = 1.0

    def validate(self) -> None:
        if self.wavelength_um <= 0.0:
            raise ValueError("wavelength_um must be positive")
        if self.power < 0.0:
            raise ValueError("power must be nonnegative")
        if self.waist_x_um <= 0.0 or self.waist_y_um <= 0.0:
            raise ValueError("waists must be positive")
        if self.theta_weight < 0.0:
            raise ValueError("theta_weight must be nonnegative")


@dataclass(frozen=True)
class BeamExperiment:
    """Collection of input optical channels."""

    channels: tuple[BeamChannel, ...] = field(default_factory=lambda: (BeamChannel(),))
    coherence: CoherenceMode = "incoherent"

    def validate(self) -> None:
        if len(self.channels) < 1:
            raise ValueError("at least one beam channel is required")
        if self.coherence not in ("incoherent", "coherent"):
            raise ValueError("coherence must be 'incoherent' or 'coherent'")
        for ch in self.channels:
            ch.validate()

    @property
    def Nch(self) -> int:
        return len(self.channels)


def single_gaussian(
    *,
    wavelength_um: float = 0.633,
    power: float = 1.0,
    waist_um: float = 3.0,
    name: str = "beam",
) -> BeamExperiment:
    """Convenience constructor for one centered circular Gaussian channel."""
    return BeamExperiment(
        channels=(
            BeamChannel(
                name=name,
                wavelength_um=wavelength_um,
                power=power,
                waist_x_um=waist_um,
                waist_y_um=waist_um,
            ),
        ),
        coherence="incoherent",
    )


def symmetric_pair(
    *,
    wavelength_um: float = 0.633,
    total_power: float = 1.0,
    waist_um: float = 3.0,
    separation_um: float = 10.0,
    angle_x_rad_per_um: float = 0.0,
    phase_difference_rad: float = 0.0,
    coherence: CoherenceMode = "coherent",
) -> BeamExperiment:
    """Convenience constructor for two equal Gaussian channels.

    The beams are placed at y = +/- separation/2 and launched with opposite
    x tilts if ``angle_x_rad_per_um`` is nonzero.
    """
    p = 0.5 * float(total_power)
    sep = 0.5 * float(separation_um)
    return BeamExperiment(
        channels=(
            BeamChannel(
                name="beam_minus",
                wavelength_um=wavelength_um,
                power=p,
                waist_x_um=waist_um,
                waist_y_um=waist_um,
                y0_um=-sep,
                tilt_x_rad_per_um=-float(angle_x_rad_per_um),
                phase_rad=0.0,
            ),
            BeamChannel(
                name="beam_plus",
                wavelength_um=wavelength_um,
                power=p,
                waist_x_um=waist_um,
                waist_y_um=waist_um,
                y0_um=sep,
                tilt_x_rad_per_um=float(angle_x_rad_per_um),
                phase_rad=float(phase_difference_rad),
            ),
        ),
        coherence=coherence,
    )


def summary(exp: BeamExperiment) -> dict:
    """Return a serializable summary of the beam experiment."""
    exp.validate()
    return {
        "Nch": exp.Nch,
        "coherence": exp.coherence,
        "channels": [
            {
                "name": ch.name,
                "wavelength_um": float(ch.wavelength_um),
                "power": float(ch.power),
                "waist_x_um": float(ch.waist_x_um),
                "waist_y_um": float(ch.waist_y_um),
                "x0_um": float(ch.x0_um),
                "y0_um": float(ch.y0_um),
                "tilt_x_rad_per_um": float(ch.tilt_x_rad_per_um),
                "tilt_y_rad_per_um": float(ch.tilt_y_rad_per_um),
                "phase_rad": float(ch.phase_rad),
                "theta_weight": float(ch.theta_weight),
            }
            for ch in exp.channels
        ],
    }


__all__ = [
    "CoherenceMode",
    "BeamChannel",
    "BeamExperiment",
    "single_gaussian",
    "symmetric_pair",
    "summary",
]
