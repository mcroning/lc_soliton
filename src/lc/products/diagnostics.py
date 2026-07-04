"""Shared diagnostics for LC workflows and products.

These functions operate on already-prepared arrays and runtime grids. They do
not know how to build experiments, run algorithms, save files, or plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..numerics.backend import asnumpy

Array = Any


def scalar_float(x: Any) -> float:
    """Convert backend scalar/array scalar to Python float."""
    return float(asnumpy(x))


def total_power(I: Array, grid: Any) -> float:
    """Return integral I dx dy for a 2-D intensity."""
    xp = grid.xp
    p = xp.sum(I) * float(grid.dx_um) * float(grid.dy_um)
    return scalar_float(p)


def centroid(I: Array, grid: Any) -> tuple[float, float]:
    """Return intensity centroid (xc_um, yc_um)."""
    xp = grid.xp
    raw = xp.sum(I) + xp.asarray(1e-300, dtype=I.dtype)
    xc = xp.sum(I * grid.x_um[:, None]) / raw
    yc = xp.sum(I * grid.y_um[None, :]) / raw
    return scalar_float(xc), scalar_float(yc)


def rms_widths(I: Array, grid: Any) -> tuple[float, float]:
    """Return intensity RMS widths (sx_um, sy_um)."""
    xp = grid.xp
    raw = xp.sum(I) + xp.asarray(1e-300, dtype=I.dtype)
    xc = xp.sum(I * grid.x_um[:, None]) / raw
    yc = xp.sum(I * grid.y_um[None, :]) / raw
    sx = xp.sqrt(xp.sum(I * (grid.x_um[:, None] - xc) ** 2) / raw)
    sy = xp.sqrt(xp.sum(I * (grid.y_um[None, :] - yc) ** 2) / raw)
    return scalar_float(sx), scalar_float(sy)


def intensity_metrics(I: Array, grid: Any) -> dict[str, float]:
    """Return standard scalar metrics for one 2-D intensity frame."""
    xp = grid.xp
    xc, yc = centroid(I, grid)
    sx, sy = rms_widths(I, grid)
    return {
        "power": total_power(I, grid),
        "Imax": scalar_float(xp.max(I)),
        "sx_um": sx,
        "sy_um": sy,
        "xc_um": xc,
        "yc_um": yc,
    }


def theta_metrics(theta: Array) -> dict[str, float]:
    """Return min/max/rms metrics for a theta array."""
    try:
        xp = theta.__array_namespace__()  # type: ignore[attr-defined]
    except Exception:
        xp = None

    arr = asnumpy(theta)
    return {
        "theta_min": float(np.min(arr)),
        "theta_max": float(np.max(arr)),
        "theta_rms": float(np.sqrt(np.mean(arr * arr))),
    }


def theta_update_metrics(theta: Array, theta_prev: Array) -> dict[str, float]:
    """Return RMS and max update between two theta arrays."""
    d = asnumpy(theta - theta_prev)
    return {
        "dtheta_rms": float(np.sqrt(np.mean(d * d))),
        "dtheta_max": float(np.max(np.abs(d))),
    }


def residual_theta_static(
    theta: Array,
    intensity: Array,
    *,
    b: float,
    bi: float,
    dx: float,
    dy: float,
    theta_bc: float,
    xp: Any | None = None,
) -> dict[str, float]:
    """Return residual metrics for static theta equation.

    Residual is evaluated on interior x rows:

        lap(theta) + (b + bi I) sin(2 theta)

    This is a diagnostic only; it is not used by the algorithms.
    """
    if xp is None:
        xp = getattr(grid := None, "xp", np)  # harmless fallback
        if type(theta).__module__.split(".")[0] == "cupy":
            import cupy as cp  # type: ignore

            xp = cp

    lap = xp.zeros_like(theta)
    yp = xp.roll(theta, -1, axis=1)
    ym = xp.roll(theta, +1, axis=1)

    lap[1:-1, :] = (
        (theta[2:, :] - 2.0 * theta[1:-1, :] + theta[:-2, :]) / (float(dx) * float(dx))
        + (yp[1:-1, :] - 2.0 * theta[1:-1, :] + ym[1:-1, :]) / (float(dy) * float(dy))
    )

    R = lap + (float(b) + float(bi) * intensity) * xp.sin(2.0 * theta)
    Ri = R[1:-1, :]
    return {
        "residual_rms": scalar_float(xp.sqrt(xp.mean(Ri * Ri))),
        "residual_max": scalar_float(xp.max(xp.abs(Ri))),
    }


__all__ = [
    "Array",
    "scalar_float",
    "total_power",
    "centroid",
    "rms_widths",
    "intensity_metrics",
    "theta_metrics",
    "theta_update_metrics",
    "residual_theta_static",
]
