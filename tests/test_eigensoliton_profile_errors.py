import pytest


def test_list_eigensoliton_profiles_missing_dir():
    from lc_soliton import list_eigensoliton_profiles

    profiles = list_eigensoliton_profiles("does/not/exist")

    assert profiles == []
