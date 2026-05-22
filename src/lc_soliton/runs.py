"""
Public run-loading API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .legacy_validated.lc_offload_tools import rebuild_ctx_from_run


def load_run(run_dir: str | Path) -> tuple[Any, dict]:
    """
    Load a saved LC run directory.

    Parameters
    ----------
    run_dir:
        Path to a saved run directory containing reproducibility metadata.

    Returns
    -------
    ctx, prdata:
        Reconstructed simulation context and legacy parameter dictionary.
    """
    return rebuild_ctx_from_run(Path(run_dir))


__all__ = [
    "load_run",
]
