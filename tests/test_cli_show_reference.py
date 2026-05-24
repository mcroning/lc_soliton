import subprocess


def test_cli_show_reference_runs():
    result = subprocess.run(
        ["lc-soliton", "--show-reference", "strict_static_centroid_drift"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "reference case:" in result.stdout
    assert "strict_static_centroid_drift" in result.stdout
    assert "trusted metrics:" in result.stdout
    assert "summary:" in result.stdout
