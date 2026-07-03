#!/usr/bin/env python
"""Tiny outside-in app for the clean LC package.

Not Streamlit. This is a minimal command-line app showing the intended public
API:

    request = TDRequest(...)
    result = run(request)
    result.show()
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_src_on_path()

    from lc.request import TDRequest
    from lc.run import run
    from lc.numerics.backend import BackendSpec
    from lc.numerics.grid import GridSpec, TimeSpec
    from lc.physics.beam import single_gaussian
    from lc.physics.liquid_crystal import LCSpec, LCCell

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--z-um", type=float, default=500.0)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--Nt", type=int, default=3)
    args = ap.parse_args()

    lc = LCSpec(cell=LCCell(interaction_length_um=args.z_um))
    request = TDRequest(
        lc=lc,
        beams=single_gaussian(wavelength_um=0.633, power=1.0, waist_um=3.0),
        backend=BackendSpec(args.backend, args.precision, verbose=True),
        grid=GridSpec(
            Nx=args.Nx,
            Ny=args.Ny,
            dz_um=args.dz_um,
            x_aperture_um=lc.cell.thickness_um,
            y_aperture_um=lc.cell.y_aperture_um,
            z_length_um=lc.cell.interaction_length_um,
        ),
        time=TimeSpec(dt=7.5e-4, Nt=args.Nt),
        tridiag=args.tridiag,
    )

    result = run(request)
    result.show()


if __name__ == "__main__":
    main()
