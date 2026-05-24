def test_material_request_fields_reach_engine():
    from lc_soliton import (
        SimulationRequest,
        MaterialRequest,
    )

    req = SimulationRequest(
        material=MaterialRequest(
            ne=1.71,
            no=1.52,
        )
    )

    assert req.material.ne == 1.71
    assert req.material.no == 1.52
