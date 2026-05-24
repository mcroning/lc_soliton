def test_run_engine_top_level_import():
    import lc_soliton

    assert callable(lc_soliton.run_engine)
