#!/usr/bin/env python
"""60: smoke test for the static z-stack workflow."""

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
    from lc.physics.liquid_crystal import LCSpec, LCCell
    from lc.workflows.static_zstack import StaticZStackControls, run_static_zstack

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="cupy")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--z-um", type=float, default=250.0)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--max-outer", type=int, default=3)
    ap.add_argument("--static-inner-steps", type=int, default=3)
    ap.add_argument("--static-max-steps", type=int, default=50)
    ap.add_argument("--static-resid-every", type=int, default=10)
    ap.add_argument("--static-relax-omega", type=float, default=0.3)
    ap.add_argument("--tol-residual-rms", type=float, default=5e-3)
    ap.add_argument("--tol-residual-max", type=float, default=2e-1)
    args = ap.parse_args()

    lc = LCSpec(
        cell=LCCell(
            thickness_um=75.0,
            y_aperture_um=100.0,
            interaction_length_um=args.z_um,
            theta_bc=0.0,
        )
    )

    req = StaticRequest(
        lc=lc,
        beams=single_gaussian(wavelength_um=0.633, power=1.0, waist_um=3.0),
        backend=BackendSpec(args.backend, args.precision, verbose=True),
        grid=GridSpec(
            Nx=args.Nx,
            Ny=args.Ny,
            dz_um=args.dz_um,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=args.z_um,
        ),
        max_outer=args.max_outer,
        static_inner_steps=args.static_inner_steps,
        static_max_steps=args.static_max_steps,
        static_resid_every=args.static_resid_every,
        static_relax_omega=args.static_relax_omega,
        tol_residual_rms=args.tol_residual_rms,
        tol_residual_max=args.tol_residual_max,
        tridiag=args.tridiag,
    )

    res = run_static_zstack(req, controls=StaticZStackControls(observer_stride=1))

    print("60_static_zstack_workflow_smoke")
    res.show()

    expected_Nz = int(round(args.z_um / args.dz_um))
    assert res.kind == "StaticZStackResult"
    assert res.theta is not None
    assert tuple(res.theta.shape) == (expected_Nz, args.Nx, args.Ny)
    assert res.intensity is not None
    assert tuple(res.intensity.shape) == (expected_Nz, args.Nx, args.Ny)
    assert abs(res.metrics["power"] - 1.0) < 5e-3
    assert res.metrics["theta_max"] > 0.75
    assert res.metrics["outer_steps"] <= args.max_outer
    assert res.metrics["static_max_steps"] == args.static_max_steps
    assert res.metrics["static_resid_every"] == args.static_resid_every
    assert "residual_rms" in res.metrics
    assert "residual_max" in res.metrics
    print("60_static_zstack_workflow_smoke: PASS")


if __name__ == "__main__":
    main()
