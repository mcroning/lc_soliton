"""Liquid-crystal material, cell, and bias specifications.

This module describes the physical LC apparatus. It does not build grids,
theta arrays, optical fields, products, or workflows.

Human-facing units
------------------
* lengths: microns
* voltage: volts
* angles: radians
* elastic constant K: newtons
* dielectric anisotropy delta_eps: relative / dimensionless
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

Array = Any
EPS0 = 8.8541878128e-12


@dataclass(frozen=True)
class LCMaterial:
    """Intrinsic LC material constants."""

    name: str = "generic"
    ne: float = 1.70
    no: float = 1.50
    K_N: float = 7.0e-12
    delta_eps: float = 13.0

    def validate(self) -> None:
        if self.ne <= 0.0 or self.no <= 0.0:
            raise ValueError("ne and no must be positive")
        if self.K_N <= 0.0:
            raise ValueError("K_N must be positive")
        if self.delta_eps <= 0.0:
            raise ValueError("delta_eps must be positive")


@dataclass(frozen=True)
class LCCell:
    """LC cell geometry and director boundary limits."""

    thickness_um: float = 75.0
    y_aperture_um: float = 100.0
    interaction_length_um: float = 3000.0
    theta_bc: float = 0.0
    theta_min: float = 0.0
    theta_max: float = math.pi / 2
    theta_center: float | None = None

    def validate(self) -> None:
        if self.thickness_um <= 0.0:
            raise ValueError("thickness_um must be positive")
        if self.y_aperture_um <= 0.0:
            raise ValueError("y_aperture_um must be positive")
        if self.interaction_length_um <= 0.0:
            raise ValueError("interaction_length_um must be positive")
        if self.theta_min > self.theta_max:
            raise ValueError("theta_min must be <= theta_max")
        if not (self.theta_min <= self.theta_bc <= self.theta_max):
            raise ValueError("theta_bc must lie within [theta_min, theta_max]")


@dataclass(frozen=True)
class LCBias:
    """Applied electrical bias."""

    Vapp: float = 0.9144
    b_override: float | None = None

    def validate(self) -> None:
        if self.Vapp < 0.0:
            raise ValueError("Vapp must be nonnegative")
        if self.b_override is not None and self.b_override < 0.0:
            raise ValueError("b_override must be nonnegative")


@dataclass(frozen=True)
class LCSpec:
    """Complete LC apparatus description."""

    material: LCMaterial = LCMaterial()
    cell: LCCell = LCCell()
    bias: LCBias = LCBias()
    mobility: float = 1.0
    theta_z_gamma: float = 0.0

    def validate(self) -> None:
        self.material.validate()
        self.cell.validate()
        self.bias.validate()
        if self.mobility <= 0.0:
            raise ValueError("mobility must be positive")
        if self.theta_z_gamma < 0.0:
            raise ValueError("theta_z_gamma must be nonnegative")


def compute_b_from_voltage(*, Vapp: float, delta_eps: float, K_N: float) -> float:
    """Return dimensionless electrical bias b = delta_eps eps0 V^2/(8K)."""
    return float(delta_eps) * EPS0 * float(Vapp) ** 2 / (8.0 * float(K_N))


def freedericksz_voltage(*, K_N: float, delta_eps: float) -> float:
    """Return the one-constant Freedericksz voltage scale."""
    return math.pi * math.sqrt(float(K_N) / (EPS0 * float(delta_eps)))


def resolved_b(spec: LCSpec) -> float:
    """Return b_override if present, otherwise compute b from voltage."""
    spec.validate()
    if spec.bias.b_override is not None:
        return float(spec.bias.b_override)
    return compute_b_from_voltage(
        Vapp=spec.bias.Vapp,
        delta_eps=spec.material.delta_eps,
        K_N=spec.material.K_N,
    )


def theta_center(spec: LCSpec) -> float:
    """Return requested center theta seed, defaulting to pi/4."""
    return math.pi / 4 if spec.cell.theta_center is None else float(spec.cell.theta_center)


def neff_from_theta(theta: Array, *, ne: float, no: float, xp: Any = np) -> Array:
    """Extraordinary-ray effective index."""
    c = xp.cos(theta)
    s = xp.sin(theta)
    return (float(ne) * float(no)) / xp.sqrt((float(ne) * c) ** 2 + (float(no) * s) ** 2)


def summary(spec: LCSpec) -> dict[str, float | str]:
    """Return a serializable summary."""
    spec.validate()
    return {
        "material": spec.material.name,
        "ne": float(spec.material.ne),
        "no": float(spec.material.no),
        "K_N": float(spec.material.K_N),
        "delta_eps": float(spec.material.delta_eps),
        "thickness_um": float(spec.cell.thickness_um),
        "y_aperture_um": float(spec.cell.y_aperture_um),
        "interaction_length_um": float(spec.cell.interaction_length_um),
        "theta_bc": float(spec.cell.theta_bc),
        "theta_min": float(spec.cell.theta_min),
        "theta_max": float(spec.cell.theta_max),
        "theta_center": float(theta_center(spec)),
        "Vapp": float(spec.bias.Vapp),
        "b": float(resolved_b(spec)),
        "V_F": float(freedericksz_voltage(K_N=spec.material.K_N, delta_eps=spec.material.delta_eps)),
        "mobility": float(spec.mobility),
        "theta_z_gamma": float(spec.theta_z_gamma),
    }


__all__ = [
    "Array",
    "EPS0",
    "LCMaterial",
    "LCCell",
    "LCBias",
    "LCSpec",
    "compute_b_from_voltage",
    "freedericksz_voltage",
    "resolved_b",
    "theta_center",
    "neff_from_theta",
    "summary",
]
