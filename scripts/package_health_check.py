"""
Basic package health check.

Run from the repository root:

    python scripts/package_health_check.py
"""

import lc_soliton
from lc_soliton.config import RunConfig, derive_lc_constants, validate_config


def main():
    print("lc_soliton version:", lc_soliton.__version__)

    cfg = derive_lc_constants(RunConfig())
    validate_config(cfg)

    print("default config OK")
    print("Nx, Ny, Nz:", cfg.grid.Nx, cfg.grid.Ny, cfg.grid.Nz)
    print("b, bi:", cfg.material.b, cfg.material.bi)


if __name__ == "__main__":
    main()
