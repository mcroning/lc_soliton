def test_run_engine_accepts_request_object_imports(tmp_path):
    from lc_soliton import (
        SimulationRequest,
        GridRequest,
        GeometryRequest,
        MaterialRequest,
        LaunchRequest,
        SolverRequest,
        OutputRequest,
        RuntimeRequest,
        run_engine,
    )

    req = SimulationRequest(
        mode="static",
        params={"Nx": 32, "Ny": 32, "Nz": 2},
    )

    req = SimulationRequest(
        mode="static",
    
        grid=GridRequest(
            Nx=64,
            Ny=64,
            Nz=8,
        ),
    
        geometry=GeometryRequest(
            xaper_um=75,
            yaper_um=100,
            dz_um=5,
        ),
    
        material=MaterialRequest(
            b=2.49,
            bi=214.29,
        ),
    
        launch=LaunchRequest(
            power_mW=1,
            waist_um=4,
            separation_um=0,
        ),
    
        solver=SolverRequest(
            static_max_steps=5,
        ),
    
        output=OutputRequest(
            run_dir=str(tmp_path / "engine_test"),
            save_slices=False,
            save_full=False,
        ),
    
        runtime=RuntimeRequest(
            backend="cpu",
            progress=False,
        ),
    )
    
    assert req.mode == "static"
    assert callable(run_engine)

