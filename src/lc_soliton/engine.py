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

def _add_request_metadata(run_dir: str | Path) -> None:
    import json

    from .version import __version__
    from .request_translate import REQUEST_TRANSLATION_VERSION

    path = Path(run_dir) / "metadata.json"

    if not path.exists():
        return

    metadata = json.loads(path.read_text())

    metadata["request_info"] = {
        "package_version": __version__,
        "translation_version": REQUEST_TRANSLATION_VERSION,
    }

    path.write_text(json.dumps(metadata, indent=2))

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

        from .request_validation import validate_request
        validate_request(request)

        mode = request.mode

        from .request_translate import request_to_lcparams_kwargs
        
        param_data = request_to_lcparams_kwargs(request)
        
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
            result = run_static(params, **common_kwargs)
            _add_request_metadata(request.output.run_dir)
            return result

        if mode in ("td_predictor_only", "dg_td_predictor"):
            result = ENGINE_MODES[mode](
                params,
                Nt=request.solver.Nt,
                dt=request.solver.dt,
                t_stride=request.solver.t_stride,
                **common_kwargs,
            )
            _add_request_metadata(request.output.run_dir)
            return result

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
