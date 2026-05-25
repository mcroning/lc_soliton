def test_list_eigensoliton_profiles_returns_list_for_empty_dir(tmp_path):
    from lc_soliton import list_eigensoliton_profiles

    profiles = list_eigensoliton_profiles(tmp_path)

    assert isinstance(profiles, list)
