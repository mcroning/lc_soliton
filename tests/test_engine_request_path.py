import json
from pathlib import Path

from lc_soliton import (
    SimulationRequest,
    GridRequest,
    GeometryRequest,
    MaterialRequest,
    LaunchRequest,
    BoundaryRequest,
    SolverRequest,
    OutputRequest,
    RuntimeRequest,
    run_engine,
)


def test_run_engine_accepts_typed_static_request(tmp_path):
    run_dir = tmp_path / "engine_typed_static"

    req = SimulationRequest(
        mode="static",
        grid=GridRequest(Nx=64, Ny=64, Nz=8),
        geometry=GeometryRequest(
            xaper_um=75.0,
            yaper_um=100.0,
            dz_um=5.0,
            wavelength_um=0.633,
        ),
        material=MaterialRequest(
            b=2.49,
            bi=214.29,
            ne=1.7,
            no=1.5,
        ),
        launch=LaunchRequest(
            power_mW=1.0,
            waist_um=4.0,
            separation_um=0.0,
            coherent=False,
        ),
        boundary=BoundaryRequest(
            use_sponge=True,
            windowedge=0.1,
        ),
        solver=SolverRequest(
            static_max_steps=3,
            Nt=1,
            dt=0.02,
            t_stride=1,
        ),
        output=OutputRequest(
            run_dir=str(run_dir),
            save_slices=False,
            save_full=False,
        ),
        runtime=RuntimeRequest(
            backend="auto",
            progress=False,
        ),
    )

    result = run_engine(req)

    assert result is not None
    assert run_dir.exists()
    assert (run_dir / "request.json").exists()
    assert (run_dir / "metadata.json").exists()

    request_json = json.loads((run_dir / "request.json").read_text())

    assert request_json["mode"] == "static"
    assert request_json["grid"]["Nx"] == 64
    assert request_json["geometry"]["yaper_um"] == 100.0
    assert request_json["launch"]["separation_um"] == 0.0
    assert request_json.get("params") in (None, {})
