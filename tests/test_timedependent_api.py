def test_run_td_imports():
    from lc_soliton import LCParams, run_td

    assert LCParams is not None
    assert callable(run_td)
