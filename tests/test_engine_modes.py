from lc_soliton import available_engine_modes, describe_engine_modes


def test_app_facing_modes_hide_unvalidated_dual_grid():
    modes = available_engine_modes()

    assert "static" in modes
    assert "time_dependent" in modes
    assert "time_dependent_dual_grid" not in modes


def test_mode_descriptions_cover_app_facing_modes():
    descriptions = describe_engine_modes()

    for mode in available_engine_modes():
        assert mode in descriptions
        assert descriptions[mode]

def test_experimental_modes_can_be_requested_explicitly():
    modes = available_engine_modes(include_experimental=True)
    assert "time_dependent_dual_grid" in modes