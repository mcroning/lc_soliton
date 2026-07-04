#!/usr/bin/env python
"""57: smoke test for LC products persistence helpers."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_src_on_path()

    from lc import run
    from lc.request import TDRequest
    from lc.numerics.backend import BackendSpec
    from lc.numerics.grid import GridSpec, TimeSpec
    from lc.physics.beam import single_gaussian
    from lc.physics.liquid_crystal import LCSpec, LCCell
    from lc.products.persistence import (
        load_json,
        save_arrays_npz,
        save_json,
        save_run_summary,
        save_samples_csv,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="numpy")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    lc = LCSpec(cell=LCCell(interaction_length_um=100.0))
    req = TDRequest(
        lc=lc,
        beams=single_gaussian(wavelength_um=0.633, power=1.0, waist_um=3.0),
        backend=BackendSpec(args.backend, args.precision, verbose=True),
        grid=GridSpec(
            Nx=64,
            Ny=64,
            dz_um=5.0,
            x_aperture_um=lc.cell.thickness_um,
            y_aperture_um=lc.cell.y_aperture_um,
            z_length_um=lc.cell.interaction_length_um,
        ),
        time=TimeSpec(dt=7.5e-4, Nt=1),
        tridiag="fast",
    )

    result = run(req)

    if args.run_dir is None:
        tmp = tempfile.TemporaryDirectory()
        run_dir = Path(tmp.name) / "lc57_run"
    else:
        tmp = None
        run_dir = Path(args.run_dir)

    files = save_run_summary(
        run_dir,
        result,
        metadata={
            "test": "57_static_products_persistence",
            "backend": args.backend,
            "precision": args.precision,
        },
        arrays={
            "tiny_array": np.arange(6, dtype=np.float64).reshape(2, 3),
        },
    )

    # Exercise individual helpers too.
    save_json(run_dir / "extra.json", {"ok": True, "value": 3})
    save_samples_csv(run_dir / "extra_samples.csv", [{"a": 1, "b": 2.5}, {"a": 2, "c": "x"}])
    save_arrays_npz(run_dir / "extra_arrays.npz", x=np.arange(4))

    summary = load_json(run_dir / "summary.json")
    metadata = load_json(run_dir / "metadata.json")

    assert summary["kind"] == result.kind
    assert "metrics" in summary
    assert metadata["test"] == "57_static_products_persistence"
    assert Path(files["summary_json"]).exists()
    assert Path(files["samples_csv"]).exists()
    assert Path(files["metadata_json"]).exists()
    assert Path(files["arrays_npz"]).exists()
    assert (run_dir / "extra.json").exists()
    assert (run_dir / "extra_samples.csv").exists()
    assert (run_dir / "extra_arrays.npz").exists()

    arr = np.load(run_dir / "arrays.npz")
    assert "tiny_array" in arr
    assert arr["tiny_array"].shape == (2, 3)

    print("57_products_persistence_smoke")
    print(f"  run_dir                 : {run_dir}")
    print(f"  summary_json            : {files['summary_json']}")
    print(f"  samples_csv             : {files['samples_csv']}")
    print(f"  metadata_json           : {files['metadata_json']}")
    print(f"  arrays_npz              : {files['arrays_npz']}")
    print(f"  result_kind             : {result.kind}")
    print(f"  metrics_count           : {len(result.metrics)}")
    print(f"  samples_count           : {len(result.samples)}")
    print("57_products_persistence_smoke: PASS")

    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
