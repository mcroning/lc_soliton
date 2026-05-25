# Eigensoliton profiles

The package currently includes public utilities for discovering and loading
saved eigensoliton profiles.

## Python API

    from lc_soliton import list_eigensoliton_profiles
    from lc_soliton import load_eigensoliton_profile

    profiles = list_eigensoliton_profiles(run_dir)

    profile = load_eigensoliton_profile(profile_path)

These wrap transitional validated profile-loading utilities while the full
eigensoliton/existence-curve machinery is migrated into the package.

## CLI

    lc-soliton --list-eigensoliton-profiles RUN_DIR

## Migration status

Current support:

- list saved profiles
- load saved profile dictionaries

Future support:

- request-native eigensoliton runs
- existence-curve continuation
- packaged benchmark branches
- GUI profile selection
