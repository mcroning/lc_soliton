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
        "--env",
        action="store_true",
        help="Print package and Python environment information.",
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


    if args.env:
        import sys
        from pathlib import Path
        import lc_soliton

        print("lc_soliton version:", __version__)
        print("python:", sys.version.replace("\n", " "))
        print("executable:", sys.executable)
        print("package:", Path(lc_soliton.__file__).resolve())

        try:
            import numpy as np
            print("numpy:", np.__version__)
        except Exception as e:
            print("numpy: unavailable", e)

        try:
            import scipy
            print("scipy:", scipy.__version__)
        except Exception as e:
            print("scipy: unavailable", e)

        try:
            import cupy as cp
            print("cupy:", cp.__version__)
            print("cuda devices:", cp.cuda.runtime.getDeviceCount())
        except Exception as e:
            print("cupy: unavailable", e)

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
