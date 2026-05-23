import subprocess
import sys


def test_cli_env_runs():
    result = subprocess.run(
        ["lc-soliton", "--env"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "lc_soliton version:" in result.stdout
    assert "python:" in result.stdout
