"""
Command-line interface for lc_soliton.
"""

from __future__ import annotations

import argparse

from .engine import available_engine_modes

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
        "--list-references",
        action="store_true",
        help="List available reference cases.",
    )

    parser.add_argument(
        "--run-reference",
        metavar="NAME",
        default=None,
        help="Run/validate a public reference case.",
    )

    parser.add_argument(
        "--show-reference",
        metavar="NAME",
        default=None,
        help="Show stored reference-case metadata.",
    )

    parser.add_argument(
    "--validate-reference",
    action="store_true",
    help="Validate the stored strict-static reference case.",
    )

    parser.add_argument(
    "--list-modes",
    action="store_true",
    help="List available engine modes.",
    )

    args = parser.parse_args(argv)


    if args.list_references:
        from .reference_runner import available_reference_cases

        print("available reference cases:")
        for name in available_reference_cases():
            print(f"  {name}")

        return 0

    if args.run_reference:
        from .reference_runner import run_reference_case

        summary = run_reference_case(args.run_reference)
        print("reference case passed:", args.run_reference)

        if isinstance(summary, dict) and "max_residual_rms" in summary:
            print("max_residual_rms:", summary["max_residual_rms"])

        return 0

    if args.show_reference:
        from .reference_cases import load_reference_case

        data = load_reference_case(args.show_reference)

        print("reference case:", data["name"])
        print("case_dir:", data["case_dir"])

        if "trusted_metrics" in data:
            print("\ntrusted metrics:")
            for k, v in data["trusted_metrics"].items():
                if k != "notes":
                    print(f"  {k}: {v}")

        if "summary" in data:
            print("\nsummary:")
            for k, v in data["summary"].items():
                print(f"  {k}: {v}")

        if "figures" in data:
            print("\nfigures:")
            for k, v in data["figures"].items():
                print(f"  {k}: {v}")

        return 0

    if args.validate_reference:
        from .reference_runner import run_reference_case

        summary = run_reference_case("strict_static_centroid_drift")
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

        print("available engine modes:", ", ".join(available_engine_modes()))
        
        cfg = derive_lc_constants(RunConfig())
        validate_config(cfg)
        print_config_summary(cfg)
        return 0

    if args.list_modes:
        print("available engine modes:")
        for mode in available_engine_modes():
            print(f"  {mode}")
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
