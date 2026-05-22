"""
Public time-dependent propagation API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .legacy_validated.lc_engine_validated import LCParams, run_lc_validated


def run_td(
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
    Run a validated time-dependent LC propagation calculation.

    This uses the validated predictor-only TD branch.
    """
    return run_lc_validated(
        params,
        run_dir=run_dir,
        mode="td_predictor_only",
        Nt=Nt,
        dt=dt,
        t_stride=t_stride,
        save_slices=save_slices,
        save_full=save_full,
        progress=progress,
    )


__all__ = [
    "LCParams",
    "run_td",
]
