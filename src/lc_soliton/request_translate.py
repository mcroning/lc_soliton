"""
Translation from structured SimulationRequest objects to legacy LCParams kwargs.
"""

from __future__ import annotations

from typing import Any

from .request import SimulationRequest


def request_to_lcparams_kwargs(request: SimulationRequest) -> dict[str, Any]:
    """
    Convert a structured SimulationRequest into LCParams keyword arguments.

    The legacy params dictionary is applied last so existing callers can still
    override fields during migration.
    """
    data = {
        # grid
        "Nx": request.grid.Nx,
        "Ny": request.grid.Ny,
        "Nz": request.grid.Nz,

        # geometry
        "xaper_um": request.geometry.xaper_um,
        "yaper_um": request.geometry.yaper_um,
        "dz_um": request.geometry.dz_um,

        # material fields currently accepted by LCParams
        "ne": request.material.ne,
        "no": request.material.no,

        # launch fields currently accepted by LCParams
        "power_norm": request.launch.power_mW,
        "waist_x_um": request.launch.waist_um,
        "waist_y_um": request.launch.waist_um,
        "y_sep_um": request.launch.separation_um,

        # solver
        "static_max_steps": request.solver.static_max_steps,
    }

    data.update(request.params)
    return data


__all__ = ["request_to_lcparams_kwargs"]
