#!/usr/bin/env python
"""65: smoke test for the clean time-dependent propagation workflow."""

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

    from lc.request import TDRequest
    from lc.numerics.backend import BackendSpec
    from lc.numerics.grid import GridSpec, TimeSpec
    from lc.physics.beam import single_gaussian
    from lc.physics.liquid_crystal import LCSpec, LCCell, LCMaterial, LCBias
    from lc.workflows.timedependent import TDControls, run_timedependent

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="cupy")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float64")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--z-um", type=float, default=250.0)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--Nt", type=int, default=3)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--theta-picard-iters", type=int, default=4)
    ap.add_argument("--theta-picard-tol", type=float, default=1e-6)
    ap.add_argument("--observer-stride-z", type=int, default=None)
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

    req = TDRequest(
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
        time=TimeSpec(
            Nt=args.Nt,
            dt=args.dt,
        ),
        tridiag=args.tridiag,
    )

    controls = TDControls(
        theta_picard_iters=args.theta_picard_iters,
        theta_picard_tol=args.theta_picard_tol,
        observer_stride_z=args.observer_stride_z,
    )

    res = run_timedependent(req, controls=controls)

    print("65_timedependent_workflow_smoke")
    res.show()

    expected_Nz = int(round(args.z_um / args.dz_um))

    assert res.kind == "TDResult"
    assert res.theta is not None
    assert res.A_last is not None
    assert res.final_intensity is not None
    assert tuple(res.theta.shape) == (expected_Nz, args.Nx, args.Ny)
    assert tuple(res.A_last.shape[-2:]) == (args.Nx, args.Ny)
    assert tuple(res.final_intensity.shape) == (args.Nx, args.Ny)
    assert abs(res.metrics["power"] - args.power) < 5e-3
    assert res.metrics["Imax"] > 0.0
    assert res.metrics["theta_max"] > 0.75
    assert res.metrics["Nt"] == args.Nt
    assert res.metrics["bi"] > 0.0
    assert len(res.samples) > 0

    print("65_timedependent_workflow_smoke: PASS")


if __name__ == "__main__":
    main()
