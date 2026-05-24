import subprocess


def test_cli_list_modes_runs():
    result = subprocess.run(
        ["lc-soliton", "--list-modes"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "available engine modes:" in result.stdout
    assert "strict_static" in result.stdout
    assert "td_predictor_only" in result.stdout
    assert "dg_td_predictor" in result.stdout
