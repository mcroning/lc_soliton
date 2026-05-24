def test_collect_environment_has_version():
    from lc_soliton.environment import collect_environment

    env = collect_environment()

    assert "lc_soliton_version" in env
    assert "python_version" in env
    assert "hostname" in env
