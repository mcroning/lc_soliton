def test_load_reference_case():
    from lc_soliton import load_reference_case

    data = load_reference_case("strict_static_centroid_drift")

    assert data["name"] == "strict_static_centroid_drift"
    assert "trusted_metrics" in data
    assert "summary" in data
    assert "figures" in data
    assert "reports" in data
    assert isinstance(data["reports"], list)
