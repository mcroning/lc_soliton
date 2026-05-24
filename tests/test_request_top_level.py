def test_request_objects_top_level():
    import lc_soliton

    assert callable(lc_soliton.SimulationRequest)
    assert callable(lc_soliton.OutputRequest)
    assert callable(lc_soliton.RuntimeRequest)
