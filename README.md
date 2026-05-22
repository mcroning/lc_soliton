# lc-soliton

Liquid-crystal optical soliton simulation tools.

This package preserves and organizes validated research code for:
- static LC propagation
- time-dependent LC relaxation
- eigensoliton / existence-curve calculations
- stability diagnostics
- run loading and reproducibility

The initial package intentionally imports validated code with minimal refactoring.

## Current status

This repository currently wraps a validated legacy LC simulation engine
with minimal numerical changes.

The immediate goals are:
- preservation
- reproducibility
- packaging
- documentation
- regression testing

before deeper refactoring.

## Installation

    pip install -e ".[dev]"

## Quick test

    lc-soliton --summary

## Example

    python examples/config_smoke_test.py

## Public API

The current recommended public entry points are:

    from lc_soliton import (
        LCParams,
        run_static,
        run_td,
        run_dg_td,
        load_run,
    )

Example:

    from pathlib import Path
    from lc_soliton import LCParams, run_static

    params = LCParams(Nx=64, Ny=64, Nz=8)
    result = run_static(params, run_dir=Path("runs/example"))

Legacy validated internals remain available during the transition, but new
user-facing scripts should prefer the public API above.

## Reference validation

A stored strict-static reference case can be checked with:

    lc-soliton --validate-reference

This verifies trusted convergence and residual metrics for the first packaged
physics regression case.
