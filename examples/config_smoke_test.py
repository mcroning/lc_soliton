"""
Minimal lc_soliton configuration smoke test.
"""

from lc_soliton.config import (
    RunConfig,
    derive_lc_constants,
    print_config_summary,
    validate_config,
)


def main():
    cfg = RunConfig()

    cfg.grid.Nx = 128
    cfg.grid.Ny = 256
    cfg.grid.rlen_um = 1000.0
    cfg.grid.dz_um = 10.0

    cfg.material.P_mW = 2.0
    cfg.material.bias_voltage = 3.5

    cfg = derive_lc_constants(cfg)

    validate_config(cfg)

    print_config_summary(cfg)


if __name__ == "__main__":
    main()
