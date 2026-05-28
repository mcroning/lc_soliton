import subprocess


from lc_soliton import available_engine_modes

def test_cli_list_modes_runs():
    result = subprocess.run(
        ["lc-soliton", "--list-modes"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "available engine modes:" in result.stdout
    assert "static" in result.stdout
    assert "time_dependent" in result.stdout

    # DG hidden from normal CLI listing
    assert "time_dependent_dual_grid" not in result.stdout

    # but still available internally
    assert "time_dependent_dual_grid" in available_engine_modes(
        include_experimental=True
    )
