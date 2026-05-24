import subprocess


def test_cli_list_references_runs():
    result = subprocess.run(
        ["lc-soliton", "--list-references"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "available reference cases:" in result.stdout
    assert "strict_static_centroid_drift" in result.stdout
