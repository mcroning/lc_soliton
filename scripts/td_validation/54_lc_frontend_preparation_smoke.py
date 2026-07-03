#!/usr/bin/env python
"""54: smoke test for the clean lc front-end preparation layer.

This tests the human-facing preparation stack:

    LCSpec
    BeamExperiment
    BackendSpec
    GridSpec
        ↓
    RuntimeGrid
    BiasResult
    LaunchResult

It intentionally does not run propagation or theta solving.  Those are covered
by algorithm tests 50/52/53.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_src_on_path()

    from lc.numerics.backend import BackendSpec, get_backend, asnumpy
    from lc.numerics.grid import GridSpec, TimeSpec, make_grid
    from lc.physics.liquid_crystal import (
        LCBias,
        LCCell,
        LCMaterial,
        LCSpec,
        resolved_b,
        summary as lc_summary,
    )
    from lc.physics.beam import single_gaussian, symmetric_pair, summary as beam_summary
    from lc.physics.bias import build_bias
    from lc.physics.launch import build_launch, total_power

    backend = get_backend(BackendSpec(backend="numpy", precision="float64", verbose=False))

    lc = LCSpec(
        material=LCMaterial(name="generic-test", ne=1.7, no=1.5, K_N=7.0e-12, delta_eps=13.0),
        cell=LCCell(thickness_um=75.0, y_aperture_um=100.0, interaction_length_um=500.0, theta_bc=0.0),
        bias=LCBias(Vapp=0.9144),
        mobility=1.0,
    )
    lc.validate()

    grid = make_grid(
        GridSpec(
            Nx=128,
            Ny=128,
            dz_um=5.0,
            x_aperture_um=lc.cell.thickness_um,
            y_aperture_um=lc.cell.y_aperture_um,
            z_length_um=lc.cell.interaction_length_um,
        ),
        xp=backend.xp,
        real_dtype=backend.real_dtype,
    )

    time = TimeSpec(dt=7.5e-4, Nt=3)
    time.validate()

    bias = build_bias(lc, grid)

    one_beam = single_gaussian(wavelength_um=0.633, power=1.0, waist_um=3.0)
    launch1 = build_launch(one_beam, grid, complex_dtype=backend.complex_dtype)
    p1 = total_power(launch1.A0, grid)

    pair = symmetric_pair(
        wavelength_um=0.633,
        total_power=1.0,
        waist_um=3.0,
        separation_um=10.0,
        angle_x_rad_per_um=0.01,
        phase_difference_rad=0.0,
        coherence="coherent",
    )
    launch2 = build_launch(pair, grid, complex_dtype=backend.complex_dtype)
    p2 = total_power(launch2.A0, grid)

    assert grid.Nx == 128
    assert grid.Ny == 128
    assert grid.Nz == 100
    assert bias.theta_2d.shape == (grid.Nx, grid.Ny)
    assert bias.theta_stack.shape == (grid.Nz, grid.Nx, grid.Ny)
    assert launch1.A0.shape == (1, grid.Nx, grid.Ny)
    assert launch2.A0.shape == (2, grid.Nx, grid.Ny)
    assert abs(p1 - 1.0) < 1e-12
    assert abs(p2 - 1.0) < 1e-12

    theta2 = asnumpy(bias.theta_2d)
    assert np.allclose(theta2[0, :], lc.cell.theta_bc)
    assert np.allclose(theta2[-1, :], lc.cell.theta_bc)
    assert theta2.max() <= lc.cell.theta_max + 1e-14
    assert theta2.min() >= lc.cell.theta_min - 1e-14

    print("54_lc_frontend_preparation_smoke")
    print(f"  backend                 : {backend.name}")
    print(f"  precision               : {backend.real_dtype}")
    print(f"  grid                    : {grid.Nx} x {grid.Ny} x {grid.Nz}")
    print(f"  dx_um, dy_um, dz_um     : {grid.dx_um:.12g}, {grid.dy_um:.12g}, {grid.dz_um:.12g}")
    print(f"  du, dv                  : {grid.du:.12g}, {grid.dv:.12g}")
    print(f"  b                       : {resolved_b(lc):.12g}")
    print(f"  theta_min/max           : {theta2.min():.12g}, {theta2.max():.12g}")
    print(f"  one_beam_power          : {p1:.16e}")
    print(f"  pair_power              : {p2:.16e}")
    print(f"  one_beam_Nch            : {launch1.A0.shape[0]}")
    print(f"  pair_Nch                : {launch2.A0.shape[0]}")
    print(f"  lc_summary_keys         : {len(lc_summary(lc))}")
    print(f"  beam_summary_Nch        : {beam_summary(pair)['Nch']}")
    print("54_lc_frontend_preparation_smoke: PASS")


if __name__ == "__main__":
    main()
