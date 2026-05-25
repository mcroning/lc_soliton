def test_engine_modes_exist():
    from lc_soliton.engine import ENGINE_MODES

    assert "static" in ENGINE_MODES
    assert "time_dependent" in ENGINE_MODES
    assert "time_dependent_dual_grid" in ENGINE_MODES
