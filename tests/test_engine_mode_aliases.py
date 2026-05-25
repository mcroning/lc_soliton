def test_legacy_engine_mode_aliases():
    from lc_soliton.engine import canonical_engine_mode

    assert canonical_engine_mode("strict_static") == "static"
    assert canonical_engine_mode("td_predictor_only") == "time_dependent"
    assert canonical_engine_mode("dg_td_predictor") == "time_dependent_dual_grid"
    assert canonical_engine_mode("static") == "static"
