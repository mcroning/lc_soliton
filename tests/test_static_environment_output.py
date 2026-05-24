from pathlib import Path
import shutil


def test_run_static_writes_environment_json():
    from lc_soliton import LCParams, run_static

    run_dir = Path("runs/test_static_environment_output")

    if run_dir.exists():
        shutil.rmtree(run_dir)

    params = LCParams(
        Nx=32,
        Ny=32,
        Nz=2,
        static_max_steps=2,
    )

    run_static(
        params,
        run_dir=run_dir,
        save_slices=False,
        save_full=False,
        progress=None,
    )

    assert (run_dir / "environment.json").exists()
