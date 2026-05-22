"""
Canonical high-level simulation runners.

These names are intended to become the stable public entry points.
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
