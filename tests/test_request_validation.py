import pytest


def test_request_validation_rejects_negative_grid():
    from lc_soliton import SimulationRequest, GridRequest
    from lc_soliton.request_validation import validate_request

    req = SimulationRequest(
        grid=GridRequest(Nx=-1, Ny=32, Nz=2),
    )

    with pytest.raises(ValueError):
        validate_request(req)
