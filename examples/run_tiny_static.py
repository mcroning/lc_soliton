"""
Tiny end-to-end static LC run.

This is intentionally small. It verifies that the canonical runner can create
a run directory and write metadata/output files.
"""

from pathlib import Path

from lc_soliton import LCParams, run_lc_validated


def main():
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

    run_dir = Path("runs/tiny_static_example")

    result = run_lc_validated(
        params,
        run_dir=run_dir,
        mode="strict_static",
        Nt=1,
        save_slices=True,
        save_full=False,
        progress=print,
    )

    print("Run complete.")
    print("Run directory:", run_dir)
    print("Result keys:", sorted(result.keys()))


if __name__ == "__main__":
    main()
