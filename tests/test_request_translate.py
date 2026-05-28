def test_request_to_lcparams_kwargs_contains_structured_fields():
    from lc_soliton import (
        SimulationRequest,
        GridRequest,
        GeometryRequest,
        MaterialRequest,
        LaunchRequest,
        SolverRequest,
    )
    from lc_soliton.request_translate import request_to_lcparams_kwargs

    req = SimulationRequest(
        mode="static",
        grid=GridRequest(Nx=10, Ny=11, Nz=12),
        geometry=GeometryRequest(xaper_um=70, yaper_um=800, dz_um=5),
        material=MaterialRequest(ne=1.71, no=1.52),
        launch=LaunchRequest(power_mW=2, waist_um=6, separation_um=15),
        solver=SolverRequest(static_max_steps=3),
    )

    data = request_to_lcparams_kwargs(req)

    assert data["Nx"] == 10
    assert data["Ny"] == 11
    assert data["Nz"] == 12
    assert data["xaper_um"] == 70
    assert data["ne"] == 1.71
    assert data["waist_x_um"] == 6
    assert data["static_max_steps"] == 3
