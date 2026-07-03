"""Build initial LC director bias fields.

This module converts an ``LCSpec`` plus a runtime grid into theta arrays for
the algorithms. It does not solve the nonlinear static bias equation yet; v001
provides the same smooth bias seed used in the clean validation runs.

Array convention
----------------
2-D theta fields are shaped ``(Nx, Ny)``.
z-stacks are shaped ``(Nz, Nx, Ny)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .liquid_crystal import LCSpec, theta_center
from ..numerics.grid import RuntimeGrid

Array = Any


@dataclass(frozen=True)
class BiasResult:
    """Prepared director bias fields."""

    theta_2d: Array
    theta_stack: Array
    theta_bc: float
    theta_clamp: tuple[float, float]

    def summary(self) -> dict[str, float | tuple[float, float]]:
        return {
            "theta_bc": float(self.theta_bc),
            "theta_clamp": self.theta_clamp,
            "theta_min": float(_to_numpy(self.theta_2d).min()),
            "theta_max": float(_to_numpy(self.theta_2d).max()),
        }


def _to_numpy(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def build_cosine_bias_2d(spec: LCSpec, grid: RuntimeGrid) -> Array:
    """Return a smooth x-dependent theta seed with Dirichlet x rows.

    The center value is ``spec.cell.theta_center`` if supplied, otherwise
    pi/4.  This is a seed/preparation field, not a nonlinear bias solver.
    """

    spec.validate()
    xp = grid.xp
    dtype = grid.real_dtype

    center = theta_center(spec)
    bc = float(spec.cell.theta_bc)

    xnorm = grid.x_um / max(1e-300, float(spec.cell.thickness_um) / 2.0)
    profile = bc + (center - bc) * xp.cos(0.5 * xp.pi * xnorm)
    profile = xp.maximum(profile, bc)

    theta = xp.repeat(profile[:, None], int(grid.Ny), axis=1)
    theta = xp.clip(
        theta,
        float(spec.cell.theta_min),
        float(spec.cell.theta_max),
    ).astype(dtype, copy=False)

    theta[0, :] = bc
    theta[-1, :] = bc
    return theta


def stack_theta(theta_2d: Array, grid: RuntimeGrid) -> Array:
    """Repeat a 2-D theta field into a z-stack."""
    xp = grid.xp
    return xp.repeat(theta_2d[None, :, :], int(grid.Nz), axis=0).astype(grid.real_dtype, copy=False)


def build_bias(spec: LCSpec, grid: RuntimeGrid) -> BiasResult:
    """Build the default 2-D and z-stack theta bias fields."""
    theta_2d = build_cosine_bias_2d(spec, grid)
    theta_stack = stack_theta(theta_2d, grid)
    return BiasResult(
        theta_2d=theta_2d,
        theta_stack=theta_stack,
        theta_bc=float(spec.cell.theta_bc),
        theta_clamp=(float(spec.cell.theta_min), float(spec.cell.theta_max)),
    )


__all__ = [
    "Array",
    "BiasResult",
    "build_cosine_bias_2d",
    "stack_theta",
    "build_bias",
]
