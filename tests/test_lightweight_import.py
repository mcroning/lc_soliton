import subprocess
import sys


def test_import_lc_soliton_does_not_import_cupy_in_fresh_process():
    code = (
        "import sys; "
        "import lc_soliton; "
        "raise SystemExit(1 if 'cupy' in sys.modules else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
    )

    assert result.returncode == 0
