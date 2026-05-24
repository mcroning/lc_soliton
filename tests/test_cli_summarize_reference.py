import subprocess


def test_cli_summarize_reference_runs():
    result = subprocess.run(
        ["lc-soliton", "--summarize-reference", "strict_static_centroid_drift"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "reference summary:" in result.stdout
    assert "strict_static_centroid_drift" in result.stdout
    assert "max_residual_rms" in result.stdout
