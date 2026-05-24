import pytest


def test_run_engine_validates_request_before_execution():
    from lc_soliton import SimulationRequest, GridRequest, run_engine

    req = SimulationRequest(
        grid=GridRequest(Nx=0, Ny=32, Nz=2),
    )

    with pytest.raises(ValueError, match="Nx must be positive"):
        run_engine(req)
