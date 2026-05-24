def test_engine_modes_exist():
    from lc_soliton.engine import ENGINE_MODES

    assert "strict_static" in ENGINE_MODES
    assert "td_predictor_only" in ENGINE_MODES
    assert "dg_td_predictor" in ENGINE_MODES
