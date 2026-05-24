"""
Validation helpers for structured simulation requests.
"""

from __future__ import annotations

from .request import SimulationRequest


def validate_request(req: SimulationRequest) -> None:
    if req.grid.Nx <= 0:
        raise ValueError("Nx must be positive")

    if req.grid.Ny <= 0:
        raise ValueError("Ny must be positive")

    if req.grid.Nz <= 0:
        raise ValueError("Nz must be positive")

    if req.geometry.dz_um <= 0:
        raise ValueError("dz_um must be positive")

    if req.launch.power_mW <= 0:
        raise ValueError("power_mW must be positive")
