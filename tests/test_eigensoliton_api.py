def test_eigensoliton_api_exports():
    import lc_soliton

    assert callable(lc_soliton.list_eigensoliton_profiles)
    assert callable(lc_soliton.load_eigensoliton_profile)
