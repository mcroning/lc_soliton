#!/usr/bin/env python
"""62: smoke test for the clean soliton workflow."""

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

    from lc.request import StaticRequest
    from lc.numerics.backend import BackendSpec
    from lc.numerics.grid import GridSpec
    from lc.physics.beam import single_gaussian
    from lc.physics.liquid_crystal import LCSpec, LCCell, LCMaterial, LCBias
    from lc.workflows.soliton import SolitonRequest, run_soliton

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="cupy")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float64")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--z-um", type=float, default=250.0)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--max-outer", type=int, default=8)
    ap.add_argument("--theta-steps", type=int, default=20)
    ap.add_argument("--field-mix", type=float, default=0.5)
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

    base = StaticRequest(
        lc=lc,
        beams=single_gaussian(
            wavelength_um=0.633,
            power=args.power,
            waist_um=3.0,
        ),
        backend=BackendSpec(args.backend, args.precision, verbose=True),
        grid=GridSpec(
            Nx=args.Nx,
            Ny=args.Ny,
            dz_um=args.dz_um,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=args.z_um,
        ),
        tridiag=args.tridiag,
    )

    req = SolitonRequest(
        base=base,
        max_outer=args.max_outer,
        theta_steps_per_outer=args.theta_steps,
        field_mix=args.field_mix,
        tol_field=1e-5,
        tol_theta=1e-4,
        tol_residual_rms=1e-1,
        tol_residual_max=1.0,
    )

    res = run_soliton(req)

    print("62_soliton_workflow_smoke")
    res.show()

    assert res.kind == "SolitonResult"
    assert res.A is not None
    assert res.theta is not None
    assert res.intensity is not None
    assert tuple(res.A.shape[-2:]) == (args.Nx, args.Ny)
    assert tuple(res.theta.shape) == (args.Nx, args.Ny)
    assert abs(res.metrics["power"] - args.power) < 5e-3
    assert res.metrics["theta_max"] > 0.75
    assert "beta" in res.metrics
    assert "field_rel" in res.metrics
    assert "residual_rms" in res.metrics
    assert res.metrics["outer_steps"] <= args.max_outer

    print("62_soliton_workflow_smoke: PASS")


if __name__ == "__main__":
    main()
