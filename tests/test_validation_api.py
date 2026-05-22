def test_validation_api_imports():
    from lc_soliton.validation import validate_strict_static_centroid_drift

    assert callable(validate_strict_static_centroid_drift)
