import json
import subprocess
from pathlib import Path
import shutil


def test_cli_run_request_executes_static(tmp_path):
    run_dir = Path("runs/test_cli_run_request")

    if run_dir.exists():
        shutil.rmtree(run_dir)

    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "mode": "static",
                "params": {
                    "Nx": 32,
                    "Ny": 32,
                    "Nz": 2,
                    "static_max_steps": 2,
                },
                "output": {
                    "run_dir": str(run_dir),
                    "save_slices": False,
                    "save_full": False,
                },
                "runtime": {
                    "backend": "auto",
                    "progress": False,
                },
            }
        )
    )

    result = subprocess.run(
        ["lc-soliton", "--run-request", str(request_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "request completed:" in result.stdout
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "environment.json").exists()
