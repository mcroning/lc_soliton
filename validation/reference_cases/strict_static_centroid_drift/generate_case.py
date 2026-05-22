"""
Reference-case regeneration notes for strict_static_centroid_drift.

This case was generated from the validated strict static z-march bridge.

Current status:
- trusted artifacts are stored in this directory
- validation/validate_strict_static_reference.py checks the stored metrics
- full regeneration requires rebuilding the legacy ctx and helper-function bundle

TODO:
    Add a public high-level wrapper that rebuilds ctx from config.json and calls:

        run_static_strict_z_march_bridge(
            ctx,
            choose_optics_substeps=...,
            get_h_for_dz_legacy=...,
            prepare_cn_ky_operator=...,
            strict_static_relax_slice_selfconsistent=...,
        )

Until that wrapper exists, this file intentionally does not attempt to rerun
the full case.
"""

from pathlib import Path
import json


CASE_DIR = Path(__file__).resolve().parent


def main():
    summary = json.loads((CASE_DIR / "static_z_summary.json").read_text())
    trusted = json.loads((CASE_DIR / "trusted_metrics.json").read_text())

    print("Reference case: strict_static_centroid_drift")
    print("Stored summary:")
    print(json.dumps(summary, indent=2))

    print("\nTrusted metrics:")
    print(json.dumps(trusted, indent=2))

    print("\nFull regeneration requires a public wrapper around the legacy z-march bridge.")


if __name__ == "__main__":
    main()
