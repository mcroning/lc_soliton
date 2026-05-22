def test_version_exists():
    import lc_soliton

    assert hasattr(lc_soliton, "__version__")
    assert isinstance(lc_soliton.__version__, str)
