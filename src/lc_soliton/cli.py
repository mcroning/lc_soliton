"""
Command-line interface for lc_soliton.
"""

from __future__ import annotations

import argparse

from .config import RunConfig, derive_lc_constants, print_config_summary, validate_config

from .version import __version__

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="lc-soliton",
        description="Liquid-crystal optical soliton simulation tools.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a default derived RunConfig summary.",
    )
    parser.add_argument(
    "--validate-reference",
    action="store_true",
    help="Validate the stored strict-static reference case.",
    )

    args = parser.parse_args(argv)

    if args.validate_reference:
        from .validation import validate_strict_static_centroid_drift
    
        summary = validate_strict_static_centroid_drift()
        print("strict static reference validation passed")
        print("max_residual_rms:", summary["max_residual_rms"])
        return 0

    if args.summary:
        print("lc_soliton version:", __version__)
        cfg = derive_lc_constants(RunConfig())
        validate_config(cfg)
        print_config_summary(cfg)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
