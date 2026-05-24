"""
lc_soliton

Liquid-crystal optical soliton simulation tools.
"""

from .version import __version__, __version_name__

__all__ = [
    "__version__",
    "__version_name__",
    "RunConfig",
    "LCParams",
    "get_backend",
    "run_lc_validated",
    "run_static",
    "run_td",
    "run_dg_td",
    "load_run",
    "load_reference_case",
    "run_engine",
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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
