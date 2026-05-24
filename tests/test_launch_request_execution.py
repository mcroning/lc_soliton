def test_launch_request_fields_exist():
    from lc_soliton import LaunchRequest, SimulationRequest

    req = SimulationRequest(
        launch=LaunchRequest(
            power_mW=2.0,
            waist_um=5.0,
            separation_um=12.0,
        )
    )

    assert req.launch.power_mW == 2.0
    assert req.launch.waist_um == 5.0
    assert req.launch.separation_um == 12.0
