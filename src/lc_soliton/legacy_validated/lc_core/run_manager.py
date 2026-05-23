# lc_core/run_manager.py

from __future__ import annotations

import json
import time
import socket
from pathlib import Path
import os
DEFAULT_RUN_ROOT = Path(os.environ.get("LC_SOLITON_RUN_ROOT", "runs"))

def make_run_dir(
    *,
    root=None,
    tag="run",
):
    if root is None:
        root = DEFAULT_RUN_ROOT

    stamp = time.strftime("%Y%m%d_%H%M%S")

    run_dir = Path(root) / f"{tag}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


from .config import config_to_dict

def save_config_json(cfg, run_dir):
    path = Path(run_dir) / "config.json"

    path.write_text(
        json.dumps(
            config_to_dict(cfg),
            indent=2,
        )
    )

    return path


def save_environment_json(run_dir, *, extra=None):
    info = dict(
        hostname=socket.gethostname(),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    if extra is not None:
        info.update(extra)

    path = Path(run_dir) / "environment.json"

    path.write_text(json.dumps(info, indent=2))

    return path