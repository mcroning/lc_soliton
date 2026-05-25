"""
lc_soliton

Liquid-crystal optical soliton simulation tools.
"""

from .version import __version__, __version_name__

__all__ = [
    "__version__",
    "__version_name__",

    # configuration
    "RunConfig",
    "LCParams",

    # backend / execution
    "get_backend",
    "run_lc_validated",
    "run_static",
    "run_td",
    "run_dg_td",
    "run_engine",
    "available_engine_modes",

    # requests
    "SimulationRequest",
    "LaunchRequest",
    "GridRequest",
    "GeometryRequest",
    "SolverRequest",
    "MaterialRequest",
    "OutputRequest",
    "RuntimeRequest",
    "save_request",
    "load_request",
    "simulation_request_schema",

    # run loading
    "load_run",

    # reference cases
    "load_reference_case",
    "summarize_reference_case",
    "run_reference_case",
    "available_reference_cases",
    "load_eigensoliton_profile",
    "list_eigensoliton_profiles",

]


def __getattr__(name):
    if name == "RunConfig":
        from .config import RunConfig
        return RunConfig

    if name in {"LCParams", "get_backend", "run_lc_validated"}:
        from . import runners
        return getattr(runners, name)

    if name == "run_static":
        from .static import run_static
        return run_static

    if name == "run_td":
        from .timedependent import run_td
        return run_td

    if name == "run_dg_td":
        from .dualgrid import run_dg_td
        return run_dg_td

    if name == "load_run":
        from .runs import load_run
        return load_run

    if name == "load_reference_case":
        from .reference_cases import load_reference_case
        return load_reference_case

    if name == "run_engine":
        from .engine import run_engine
        return run_engine

    if name == "available_engine_modes":
        from .engine import available_engine_modes
        return available_engine_modes

    if name == "run_reference_case":
        from .reference_runner import run_reference_case
        return run_reference_case

    if name == "available_reference_cases":
        from .reference_runner import available_reference_cases
        return available_reference_cases

    if name == "summarize_reference_case":
        from .reference_cases import summarize_reference_case
        return summarize_reference_case

    if name in {
        "SimulationRequest",
        "LaunchRequest",
        "GridRequest",
        "GeometryRequest",
        "MaterialRequest",
        "SolverRequest",
        "OutputRequest",
        "RuntimeRequest",
        "save_request",
        "load_request",
    }:
        from . import request
        return getattr(request, name)


    if name == "simulation_request_schema":
        from .request_schema import simulation_request_schema
        return simulation_request_schema

    if name in {"list_eigensoliton_profiles", "load_eigensoliton_profile"}:
        from . import eigensoliton
        return getattr(eigensoliton, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
