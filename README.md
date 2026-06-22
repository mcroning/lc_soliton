# lc-soliton

Liquid-crystal optical soliton simulation tools.

The package supports:

- static self-consistent optical propagation
- time-dependent liquid-crystal evolution
- optional dual-grid acceleration
- eigensoliton and existence-curve workflows
- stability analysis
- reproducible run storage
- Streamlit GUI operation

## Quick Start

New users should begin with:

    docs/Quick_Start.md

Documentation index:

    docs/index.md

## Current Status

The package is suitable for internal research use and active development.

Current strengths include:

- installable package
- Streamlit GUI
- static and time-dependent solvers
- dual-grid acceleration
- reference-case validation
- reproducible run storage

Active development continues on:

- existence-curve workflow polish
- stability-analysis workflow polish
- high-power static convergence
- documentation and usability improvements

## Installation

    pip install -e ".[dev]"

## Quick Test

    lc-soliton --summary

## Reference Validation

Check that the installed package reproduces the trusted strict-static benchmark:

    lc-soliton --validate-reference

## Example

    python examples/config_smoke_test.py

## Public API

The current recommended public entry points are:

    from lc_soliton import (
        LCParams,
        run_static,
        run_td,
        load_run,
    )

Example:

    from pathlib import Path
    from lc_soliton import LCParams, run_static

    params = LCParams(Nx=64, Ny=64, Nz=8)
    result = run_static(params, run_dir=Path("runs/example"))

New user-facing scripts should prefer the public API.

## Run Output Directory

By default, generated outputs go under:

    runs/

You can override this with:

    export LC_SOLITON_RUN_ROOT=/path/to/run/storage

## Run Provenance Metadata

Public runners automatically save:

    environment.json

inside each run directory.

This records:

- lc_soliton version
- Python version
- NumPy/SciPy versions
- CuPy/CUDA information (if available)
- hostname
- platform
- timestamp

This helps with reproducibility and debugging.

## Documentation

    docs/Quick_Start.md
    docs/gui.md
    docs/public_api.md
    docs/eigensoliton_profiles.md
    docs/theory.md
