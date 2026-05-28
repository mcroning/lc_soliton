def test_available_engine_modes():
    from lc_soliton import available_engine_modes

    modes = available_engine_modes()

    assert "static" in modes
    assert "time_dependent" in modes
    assert "time_dependent_dual_grid" not in modes

    experimental = available_engine_modes(include_experimental=True)

    assert "time_dependent_dual_grid" in experimental
