#!/usr/bin/env python
"""56: smoke test for the clean static workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_src_on_path()

    from lc import run
    from lc.request import StaticRequest
    from lc.numerics.backend import BackendSpec
    from lc.numerics.grid import GridSpec
    from lc.physics.liquid_crystal import (
        LCSpec,
        LCMaterial,
        LCCell,
        LCBias,
    )
    from lc.physics.beam import single_gaussian

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32")
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--z-um", type=float, default=250.0)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--outer", type=int, default=3)
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    args = ap.parse_args()

    lc = LCSpec(
        material=LCMaterial(
            name="generic",
            ne=1.7,
            no=1.5,
            K_N=7.0e-12,
            delta_eps=13.0,
        ),
        cell=LCCell(
            thickness_um=75.0,
            y_aperture_um=100.0,
            interaction_length_um=args.z_um,
            theta_bc=0.0,
        ),
        bias=LCBias(Vapp=0.9144),
        mobility=1.0,
    )

    request = StaticRequest(
        lc=lc,
        beams=single_gaussian(
            wavelength_um=0.633,
            power=1.0,
            waist_um=3.0,
        ),
        backend=BackendSpec(
            backend=args.backend,
            precision=args.precision,
            verbose=True,
        ),
        grid=GridSpec(
            Nx=args.Nx,
            Ny=args.Ny,
            dz_um=args.dz_um,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=args.z_um,
        ),
        max_outer=args.outer,
        tridiag=args.tridiag,
    )

    result = run(request)

    print("56_static_workflow_smoke")
    result.show()

    m = result.metrics

    assert abs(m["power"] - 1.0) < 5e-3
    assert m["theta_max"] > 0.75
    assert m["sx_um"] < 10.0
    assert m["sy_um"] < 10.0
    assert m["outer_steps"] <= args.outer

    print("56_static_workflow_smoke: PASS")


if __name__ == "__main__":
    main()