"""
Reference-case validation helpers.
"""

from __future__ import annotations

from pathlib import Path
import json


def validate_strict_static_centroid_drift(
    repo_root: str | Path | None = None,
) -> dict:
    """
    Validate the stored strict-static centroid-drift reference case.

    Returns
    -------
    dict
        The stored summary metrics.
    """
    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    case_dir = (
        repo_root
        / "validation"
        / "reference_cases"
        / "strict_static_centroid_drift"
    )

    required = [
        "README.md",
        "trusted_metrics.json",
        "xz_reference.png",
        "centroid_reference.png",
        "config.json",
        "static_z_summary.json",
        "static_z_reports.json",
    ]

    missing = [name for name in required if not (case_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing reference files: {missing}")

    summary = json.loads((case_dir / "static_z_summary.json").read_text())
    trusted = json.loads((case_dir / "trusted_metrics.json").read_text())

    assert summary["converged_count"] == 50
    assert summary["failed_count"] == 0
    assert abs(summary["max_residual_rms"] - trusted["max_residual_rms"]) < 1e-12
    assert abs(summary["max_residual_max"] - trusted["max_residual_max"]) < 1e-12
    assert abs(summary["P_final"] - trusted["P_final"]) < 1e-12

    return summary


__all__ = ["validate_strict_static_centroid_drift"]
