def test_run_td_imports():
    from lc_soliton import LCParams
    from lc_soliton.timedependent import run_td

    assert LCParams is not None
    assert callable(run_td)