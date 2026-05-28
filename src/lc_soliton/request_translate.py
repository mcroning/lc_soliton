"""
Translation from structured SimulationRequest objects to legacy LCParams kwargs.
"""

from __future__ import annotations

from typing import Any

from .request import SimulationRequest

REQUEST_TRANSLATION_VERSION = "1"

def request_to_lcparams_kwargs(request):
    waist_x_um = (
        request.launch.waist_x_um
        if request.launch.waist_x_um is not None
        else request.launch.waist_um
    )
    
    waist_y_um = (
        request.launch.waist_y_um
        if request.launch.waist_y_um is not None
        else request.launch.waist_um
    )
    return {
        "Nx": request.grid.Nx,
        "Ny": request.grid.Ny,
        "Nz": request.grid.Nz,

        "xaper_um": request.geometry.xaper_um,
        "yaper_um": request.geometry.yaper_um,
        "dz_um": request.geometry.dz_um,

        "wavelength_um": request.geometry.wavelength_um,

        "ne": request.material.ne,
        "no": request.material.no,
        "b": request.material.b,
        "bi": request.material.bi,
        "mobility": request.material.mobility,
        "theta_bc": request.material.theta_bc,
        "theta_z_gamma": request.material.theta_z_gamma,

        "waist_x_um": waist_x_um,
        "waist_y_um": waist_y_um,
        "y_sep_um": request.launch.separation_um,
        "coherent": request.launch.coherent,

        
        "theta_bias_amp": request.material.theta_bias_amp,
        "theta_clamp_min": request.material.theta_clamp_min,
        "theta_clamp_max": request.material.theta_clamp_max,
        "power_norm": request.launch.power_mW,
        "dz_opt_max_phi": request.solver.dz_opt_max_phi,
        "dn_max_est": request.solver.dn_max_est,
        "max_substeps": request.solver.max_substeps,
        "dtau_static": request.solver.dtau_static,
        "static_max_steps": request.solver.static_max_steps,
        "static_tol_rms": request.solver.static_tol_rms,
        "static_tol_max": request.solver.static_tol_max,
        "static_selfcons_passes": request.solver.static_selfcons_passes,
        "static_mix": request.solver.static_mix,
        "backend": request.runtime.backend,
    }
## temp commented out of list
#       "use_sponge": request.boundary.use_sponge,
#       "windowedge": request.boundary.windowedge,

__all__ = [
    "REQUEST_TRANSLATION_VERSION",
    "request_to_lcparams_kwargs",
]
