"""
Public dual-grid propagation API.

Dual-grid mode keeps the optical field on the full grid while solving the LC
director response on a coarser grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .core.context import LCParams


def run_dg_td(
    params: LCParams,
    *,
    run_dir: str | Path,
    Nt: int,
    dt: float,
    t_stride: int = 1,
    save_slices: bool = True,
    save_full: bool = False,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """
    Run dual-grid time-dependent LC propagation.

    This workflow is intentionally disabled while it is rebuilt on top of the
    cleaned TD runner.
    """
    raise NotImplementedError(
        "Dual-grid TD is being rebuilt on top of the cleaned TD runner. "
        "The old dg_td_predictor path was experimental and is intentionally "
        "not exposed as the production dual-grid runner."
    )


__all__ = [
    "LCParams",
    "run_dg_td",
]