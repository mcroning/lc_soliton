import pytest


def test_run_reference_case_rejects_unknown_case():
    from lc_soliton import run_reference_case

    with pytest.raises(ValueError, match="Unknown reference case"):
        run_reference_case("not_a_reference_case")
