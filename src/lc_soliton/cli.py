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

    args = parser.parse_args(argv)

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
