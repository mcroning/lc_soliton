"""
Canonical public execution entry points.
"""

from __future__ import annotations
from pathlib import Path
from .static import run_static
from .timedependent import run_td
from .dualgrid import run_dg_td
from .legacy_validated.lc_engine_validated import LCParams
from .request import SimulationRequest


ENGINE_MODES = {
    "strict_static": run_static,
    "td_predictor_only": run_td,
    "dg_td_predictor": run_dg_td,
}


def available_engine_modes():
    """
    Return sorted list of supported engine modes.
    """
    return sorted(ENGINE_MODES)


def run_engine(mode_or_request, *args, **kwargs):
    """
    Canonical public execution dispatcher.

    Accepts either:

        run_engine("strict_static", params, run_dir=...)

    or:

        run_engine(SimulationRequest(...))
    """
    if isinstance(mode_or_request, SimulationRequest):
        request = mode_or_request
        mode = request.mode

        param_data = {
            "Nx": request.grid.Nx,
            "Ny": request.grid.Ny,
            "Nz": request.grid.Nz,
            "xaper_um": request.geometry.xaper_um,
            "yaper_um": request.geometry.yaper_um,
            "dz_um": request.geometry.dz_um,
            "static_max_steps": request.solver.static_max_steps,
        }
        param_data.update(request.params)
        
        params = LCParams(**param_data)
        from .request import save_request

        common_kwargs = dict(
            run_dir=request.output.run_dir,
            save_slices=request.output.save_slices,
            save_full=request.output.save_full,
            progress=print if request.runtime.progress else None,
        )

        save_request(
            request,
            Path(request.output.run_dir) / "request.json",
        )

        if mode == "strict_static":
            return run_static(params, **common_kwargs)

        if mode in ("td_predictor_only", "dg_td_predictor"):
            return ENGINE_MODES[mode](
                params,
                Nt=request.solver.Nt,
                dt=request.solver.dt,
                t_stride=request.solver.t_stride,
                **common_kwargs,
            )

        raise ValueError(
            f"Unknown engine mode: {mode}. "
            f"Available modes: {sorted(ENGINE_MODES)}"
        )

    mode = mode_or_request

    if mode not in ENGINE_MODES:
        raise ValueError(
            f"Unknown engine mode: {mode}. "
            f"Available modes: {sorted(ENGINE_MODES)}"
        )

    return ENGINE_MODES[mode](*args, **kwargs)
