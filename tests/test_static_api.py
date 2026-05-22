def test_run_static_imports():
    from lc_soliton import LCParams, run_static

    assert LCParams is not None
    assert callable(run_static)
