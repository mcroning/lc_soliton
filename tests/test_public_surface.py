def test_public_surface():
    import lc_soliton

    required = [
        "run_static",
        "run_td",
        "run_dg_td",
        "run_engine",
        "available_engine_modes",
        "load_reference_case",
        "summarize_reference_case",
        "run_reference_case",
        "available_reference_cases",
    ]

    for name in required:
        assert hasattr(lc_soliton, name), name
