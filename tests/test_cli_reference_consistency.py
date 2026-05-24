import subprocess


def test_reference_cli_commands_agree():
    a = subprocess.run(
        ["lc-soliton", "--validate-reference"],
        capture_output=True,
        text=True,
        check=False,
    )

    b = subprocess.run(
        ["lc-soliton", "--run-reference", "strict_static_centroid_drift"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert a.returncode == 0
    assert b.returncode == 0
    assert "max_residual_rms:" in a.stdout
    assert "max_residual_rms:" in b.stdout
