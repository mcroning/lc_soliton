import pytest


@pytest.mark.parametrize(
    "grid_kwargs, message",
    [
        ({"Nx": 0, "Ny": 32, "Nz": 2}, "Nx must be positive"),
        ({"Nx": 32, "Ny": 0, "Nz": 2}, "Ny must be positive"),
        ({"Nx": 32, "Ny": 32, "Nz": 0}, "Nz must be positive"),
    ],
)
def test_request_validation_rejects_nonpositive_grid(grid_kwargs, message):
    from lc_soliton import SimulationRequest, GridRequest
    from lc_soliton.request_validation import validate_request

    req = SimulationRequest(grid=GridRequest(**grid_kwargs))

    with pytest.raises(ValueError, match=message):
        validate_request(req)
