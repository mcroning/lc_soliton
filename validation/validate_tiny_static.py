"""
Infrastructure validation for the tiny static example.

This checks that the canonical runner completes and writes expected files.
It is not a physics validation.
"""

from pathlib import Path
import shutil

from lc_soliton import LCParams, run_lc_validated


def main():
    run_dir = Path("runs/validation_tiny_static")

    if run_dir.exists():
        shutil.rmtree(run_dir)

    params = LCParams(
        Nx=64,
        Ny=64,
        Nz=8,
        xaper_um=75.0,
        yaper_um=150.0,
        dz_um=5.0,
        waist_x_um=8.0,
        waist_y_um=8.0,
        b=1.0,
        bi=0.2,
        static_max_steps=20,
    )

    result = run_lc_validated(
        params,
        run_dir=run_dir,
        mode="strict_static",
        Nt=1,
        save_slices=True,
        save_full=False,
        progress=None,
    )

    required = [
        "metadata.json",
        "scalar_log.csv",
        "final_summary.npz",
        "Ixz.dat",
        "Iyz.dat",
    ]

    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing expected output files: {missing}")

    if result["run_dir"] != str(run_dir):
        raise RuntimeError("Unexpected run_dir in result")

    print("tiny static infrastructure validation passed")


if __name__ == "__main__":
    main()
