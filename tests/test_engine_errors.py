import pytest


def test_run_engine_rejects_unknown_mode():
    from lc_soliton.engine import run_engine

    with pytest.raises(ValueError, match="Unknown engine mode"):
        run_engine("not_a_mode")
