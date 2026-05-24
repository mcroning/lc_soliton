def test_public_api_surface():
    import lc_soliton

    expected = [
        "RunConfig",
        "LCParams",
        "run_static",
        "run_td",
        "run_dg_td",
    ]

    missing = [name for name in expected if not hasattr(lc_soliton, name)]

    assert not missing, f"Missing public API symbols: {missing}"
