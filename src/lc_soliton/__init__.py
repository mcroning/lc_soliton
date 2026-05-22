"""
lc_soliton

Liquid-crystal optical soliton simulation tools.
"""

from .version import __version__, __version_name__
from .config import RunConfig
from .runners import LCParams, get_backend, run_lc_validated
from .static import run_static
from .timedependent import run_td
from .dualgrid import run_dg_td
from .runs import load_run

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
]
