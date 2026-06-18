"""
Public eigensoliton profile utilities.

These wrap transitional validated profile-loading helpers.
"""

from __future__ import annotations

from pathlib import Path


def list_eigensoliton_profiles(run_dir: str | Path) -> list[dict]:
    """
    List saved eigensoliton profiles in a run directory.
    """
    from .validated_core.eigenmode_core import list_run_profiles

    return list_run_profiles(Path(run_dir))


def load_eigensoliton_profile(profile_path: str | Path) -> dict:
    """
    Load a saved eigensoliton profile dictionary.
    """
    from .validated_core.eigenmode_core import load_saved_mode_profile

    return load_saved_mode_profile(Path(profile_path))

__all__ = [
    "list_eigensoliton_profiles",
    "load_eigensoliton_profile",
]
