"""
Canonical public execution entry points.
"""

from __future__ import annotations
from pathlib import Path
from .static import run_static
from .timedependent import run_td
from .legacy_validated.lc_engine_validated import LCParams
from .request import SimulationRequest


ENGINE_MODES = {
    "static": run_static,
    "time_dependent": run_td,
}

LEGACY_ENGINE_MODE_ALIASES = {
    "strict_static": "static",
    "td_predictor_only": "time_dependent",
}

ENGINE_MODE_DESCRIPTIONS = {
    "static": "Static self-consistent propagation",
    "time_dependent": "Time-dependent evolution",
}


def describe_engine_modes() -> dict[str, str]:
    """
    Return public engine mode descriptions.
    """
    return dict(ENGINE_MODE_DESCRIPTIONS)

def canonical_engine_mode(mode: str) -> str:
    return LEGACY_ENGINE_MODE_ALIASES.get(mode, mode)

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

def available_engine_modes(include_experimental: bool = False):
    modes = ["static", "time_dependent"]
    return modes


def run_engine(mode_or_request, *args, progress_callback=None, **kwargs):
    """
    Canonical public execution dispatcher.

    Accepts either:

        run_engine("strict_static", params, run_dir=...)

    or:

        run_engine(SimulationRequest(...))
    """

    should_stop = kwargs.pop("should_stop", None)

    if isinstance(mode_or_request, SimulationRequest):
        request = mode_or_request

        if getattr(request, "params", None):
            raise ValueError(
                "SimulationRequest.params is deprecated. "
                "Use grid/geometry/material/launch/boundary/solver sections."
            )

        from .request_validation import validate_request
        validate_request(request)

        mode = canonical_engine_mode(request.mode)

        from .request_translate import request_to_lcparams_kwargs

        param_data = request_to_lcparams_kwargs(request)

        params = LCParams(**param_data)
        if getattr(request, "launch_profile_path", None):
            setattr(params, "launch_profile_path", request.launch_profile_path)

        if getattr(request, "launch_profile_theta_source", None):
            setattr(
                params,
                "launch_profile_theta_source",
                request.launch_profile_theta_source,
            )
        from .request import save_request

        common_kwargs = dict(
            run_dir=request.output.run_dir,
            save_slices=request.output.save_slices,
            save_full=request.output.save_full,
            progress=(
                progress_callback
                if progress_callback is not None
                else print if request.runtime.progress else None
            ),
            should_stop=should_stop
        )

        save_request(
            request,
            Path(request.output.run_dir) / "request.json",
        )

        if mode == "static":
            print(
                "[engine] strict_max_outer_passes =",
                request.solver.strict_max_outer_passes,
            )

            result = run_static(
                params,
                strict_max_outer_passes=int(
                    request.solver.strict_max_outer_passes
                ),
                **common_kwargs,
            )

            _add_request_metadata(request.output.run_dir)
            return result

        if mode == "time_dependent":
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

    mode = canonical_engine_mode(mode_or_request)

    if mode not in ENGINE_MODES:
        raise ValueError(
            f"Unknown engine mode: {mode}. "
            f"Available modes: {sorted(ENGINE_MODES)}"
        )

    return ENGINE_MODES[mode](*args, **kwargs)
