import json
import subprocess


def test_cli_run_request_rejects_invalid_request(tmp_path):
    request_path = tmp_path / "bad_request.json"
    request_path.write_text(
        json.dumps(
            {
                "mode": "strict_static",
                "grid": {
                    "Nx": 0,
                    "Ny": 32,
                    "Nz": 2
                },
                "output": {
                    "run_dir": "runs/bad_cli_request",
                    "save_slices": False,
                    "save_full": False
                },
                "runtime": {
                    "progress": False
                }
            }
        )
    )

    result = subprocess.run(
        ["lc-soliton", "--run-request", str(request_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Nx must be positive" in result.stderr
