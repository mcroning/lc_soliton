def test_available_reference_cases():
    from lc_soliton import available_reference_cases

    cases = available_reference_cases()

    assert "strict_static_centroid_drift" in cases


def test_run_reference_case_exists():
    import lc_soliton

    assert callable(lc_soliton.run_reference_case)
