"""
Public static-propagation API.

This module defines the stable user-facing names for static LC propagation.
The current implementation delegates to validated legacy code while the
internals are being cleaned up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .legacy_validated.lc_engine_validated import LCParams, run_lc_validated
from .environment import write_environment_json


def run_static(
    params: LCParams,
    *,
    run_dir: str | Path,
    save_slices: bool = True,
    save_full: bool = False,
    progress: Callable[[str], None] | None = print,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Run a validated strict-static LC propagation calculation.

    Parameters
    ----------
    params:
        Static simulation parameters.
    run_dir:
        Directory where output files will be written.
    save_slices:
        Whether to save lightweight xz/yz diagnostic slices.
    save_full:
        Whether to save full intermediate arrays when supported.
    progress:
        Optional progress callback. Use ``None`` to suppress progress output.

    Returns
    -------
    dict
        Run metadata and lightweight result information.
    """
    result = run_lc_validated(
        params,
        run_dir=run_dir,
        mode="strict_static",
        Nt=1,
        save_slices=save_slices,
        save_full=save_full,
        progress=progress,
        should_stop=should_stop,
    )

    write_environment_json(run_dir)

    return result


__all__ = [
    "LCParams",
    "run_static",
]
