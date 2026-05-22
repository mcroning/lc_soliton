"""
Validate the trusted strict-static centroid-drift reference artifacts.
"""

from pathlib import Path
import json


CASE_DIR = Path("validation/reference_cases/strict_static_centroid_drift")


def main():
    required = [
        "README.md",
        "trusted_metrics.json",
        "xz_reference.png",
        "centroid_reference.png",
        "config.json",
        "static_z_summary.json",
        "static_z_reports.json",
    ]

    missing = [name for name in required if not (CASE_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Missing reference files: {missing}")

    summary = json.loads((CASE_DIR / "static_z_summary.json").read_text())
    trusted = json.loads((CASE_DIR / "trusted_metrics.json").read_text())

    assert summary["converged_count"] == 50
    assert summary["failed_count"] == 0

    assert abs(summary["max_residual_rms"] - trusted["max_residual_rms"]) < 1e-12
    assert abs(summary["max_residual_max"] - trusted["max_residual_max"]) < 1e-12
    assert abs(summary["P_final"] - trusted["P_final"]) < 1e-12

    print("strict static reference validation passed")


if __name__ == "__main__":
    main()
