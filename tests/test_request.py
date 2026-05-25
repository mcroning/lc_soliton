from pathlib import Path


def test_simulation_request_roundtrip(tmp_path):
    from lc_soliton.request import (
        SimulationRequest,
        GridRequest,
        OutputRequest,
        RuntimeRequest,
        save_request,
        load_request,
    )

    req = SimulationRequest(
        mode="static",
        grid=GridRequest(Nx=64, Ny=64, Nz=8),
        params={},
        output=OutputRequest(run_dir="runs/test_request"),
        runtime=RuntimeRequest(backend="auto", progress=False),
    )

    path = tmp_path / "request.json"
    save_request(req, path)

    loaded = load_request(path)

    assert loaded.mode == "static"
    assert loaded.grid.Nx == 64
    assert loaded.output.run_dir == "runs/test_request"
    assert loaded.runtime.progress is False
