from pathlib import Path
import shutil


def test_run_engine_executes_static_request():
    from lc_soliton import (
        SimulationRequest,
        GridRequest,
        OutputRequest,
        RuntimeRequest,
        run_engine,
    )

    run_dir = Path("runs/test_engine_request_execution")

    if run_dir.exists():
        shutil.rmtree(run_dir)

    req = SimulationRequest(
        mode="static",
        grid=GridRequest(Nx=32, Ny=32, Nz=2),
        params={"static_max_steps": 2},
        output=OutputRequest(
            run_dir=str(run_dir),
            save_slices=False,
            save_full=False,
        ),
        runtime=RuntimeRequest(
            progress=False,
        ),
    )

    result = run_engine(req)

    assert result is not None
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "request.json").exists()

    import json

    metadata = json.loads((run_dir / "metadata.json").read_text())

    assert metadata["request_info"]["translation_version"]
    assert metadata["request_info"]["package_version"]
