def test_available_engine_modes():
    from lc_soliton import available_engine_modes

    modes = available_engine_modes()

    assert "strict_static" in modes
    assert "td_predictor_only" in modes
    assert "dg_td_predictor" in modes
