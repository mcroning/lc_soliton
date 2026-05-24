"""
Environment and provenance helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import socket
import sys

from .version import __version__, __version_name__


def collect_environment() -> dict:
    """
    Collect JSON-safe package, Python, dependency, and hardware information.
    """
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "lc_soliton_version": __version__,
        "lc_soliton_version_name": __version_name__,
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }

    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except Exception as e:
        env["numpy_error"] = str(e)

    try:
        import scipy
        env["scipy_version"] = scipy.__version__
    except Exception as e:
        env["scipy_error"] = str(e)

    try:
        import cupy as cp
        env["cupy_version"] = cp.__version__
        env["cuda_device_count"] = int(cp.cuda.runtime.getDeviceCount())
        if env["cuda_device_count"] > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            env["cuda_device_0_name"] = props["name"].decode()
    except Exception as e:
        env["cupy_error"] = str(e)

    return env


def write_environment_json(run_dir: str | Path) -> Path:
    """
    Write environment metadata to run_dir/environment.json.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    path = run_dir / "environment.json"
    path.write_text(json.dumps(collect_environment(), indent=2))

    return path
