def test_run_engine_accepts_request_object_imports():
    from lc_soliton import SimulationRequest, run_engine

    req = SimulationRequest(
        mode="static",
        params={"Nx": 32, "Ny": 32, "Nz": 2},
    )

    assert req.mode == "static"
    assert callable(run_engine)

