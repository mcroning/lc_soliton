def test_summarize_reference_case():
    from lc_soliton.reference_cases import summarize_reference_case

    summary = summarize_reference_case("strict_static_centroid_drift")

    assert summary["name"] == "strict_static_centroid_drift"
    assert summary["has_trusted_metrics"]
    assert summary["has_reports"]
    assert "max_residual_rms" in summary
