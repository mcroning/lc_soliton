import subprocess


def test_cli_run_reference_runs():
    result = subprocess.run(
        ["lc-soliton", "--run-reference", "strict_static_centroid_drift"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "reference case passed:" in result.stdout
    assert "strict_static_centroid_drift" in result.stdout
    assert "max_residual_rms:" in result.stdout
