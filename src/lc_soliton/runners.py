"""
Legacy compatibility runners.

Prefer:
- lc_soliton.static.run_static
- lc_soliton.timedependent.run_td
- lc_soliton.dualgrid.run_dg_td
"""

from .legacy_validated.lc_engine_validated import (
    LCParams,
    get_backend,
    run_lc_validated,
)

__all__ = [
    "LCParams",
    "get_backend",
    "run_lc_validated",
]
