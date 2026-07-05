#!/usr/bin/env python
"""58: smoke test for the existence-curve workflow."""

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
    from lc.physics.liquid_crystal import LCSpec, LCCell
    from lc.physics.beam import single_gaussian
    from lc.workflows.existence_curve import (
        ExistenceCurveRequest,
        run_existence_curve,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="cupy")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    args = ap.parse_args()

    lc = LCSpec(cell=LCCell(interaction_length_um=250.0))

    base = StaticRequest(
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
            Nx=128,
            Ny=128,
            dz_um=5.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=250.0,
        ),
        max_outer=3,
        tridiag=args.tridiag,
    )

    req = ExistenceCurveRequest(
        base=base,
        powers=(0.2, 0.5, 1.0),
    )

    result = run_existence_curve(req)

    print("58_existence_curve_workflow_smoke")
    print(f"  num_points             : {result.metrics['num_points']}")
    print(f"  converged_count        : {result.metrics['converged_count']}")
    print(f"  power_min             : {result.metrics['power_min']}")
    print(f"  power_max             : {result.metrics['power_max']}")

    assert result.kind == "ExistenceCurveResult"
    assert len(result.results) == len(req.powers)
    assert len(result.samples) == len(req.powers)
    assert result.metrics["num_points"] == len(req.powers)
    assert result.metrics["power_min"] == min(req.powers)
    assert result.metrics["power_max"] == max(req.powers)

    for p, r in zip(req.powers, result.results):
        assert r.converged
        assert r.metrics["theta_max"] > 0.75
        assert abs(r.metrics["power"] - p) < 5e-3

    print("58_existence_curve_workflow_smoke: PASS")


if __name__ == "__main__":
    main()
