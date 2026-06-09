"""
Public stability-runner API for saved eigensoliton profiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .validated_core.stability_core import (
    rebuild_ctx_from_run,
    launch_stability_vs_z,
)


def run_eigenmode_launch_stability(
    run_dir: str | Path,
    profile_path: str | Path,
    *,
    Nz: int = 200,
    dz_um: float = 5.0,
    perturbations: list[dict[str, Any]] | None = None,
    save_dir: str | Path | None = None,
    runner_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Launch a saved eigensoliton profile through z propagation and measure stability.
    """
    run_dir = Path(run_dir)
    profile_path = Path(profile_path)

    ctx, prdata = rebuild_ctx_from_run(run_dir)

    ctx.Nz = int(Nz)
    ctx.dz = float(dz_um)

    if save_dir is None:
        save_dir = profile_path.parent / "launch_stability"

    df, df_pair = launch_stability_vs_z(
        ctx,
        profile_path,
        perturbations=perturbations,
        runner_kwargs=runner_kwargs,
        save_dir=save_dir,
    )

    return {
        "run_dir": str(run_dir),
        "profile_path": str(profile_path),
        "save_dir": str(save_dir),
        "Nz": int(Nz),
        "dz_um": float(dz_um),
        "metrics_csv": str(Path(save_dir) / f"{profile_path.stem}__launch_metrics.csv"),
        "paired_csv": str(Path(save_dir) / f"{profile_path.stem}__launch_paired.csv"),
        "n_metrics": int(len(df)),
        "n_paired": int(len(df_pair)),
        "metrics": df,
        "paired": df_pair,
    }


__all__ = ["run_eigenmode_launch_stability"]
