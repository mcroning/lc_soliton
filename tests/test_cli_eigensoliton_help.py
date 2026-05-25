import subprocess


def test_cli_help_mentions_eigensoliton_profiles():
    result = subprocess.run(
        ["lc-soliton", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--list-eigensoliton-profiles" in result.stdout
