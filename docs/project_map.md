# Project Map

## Main package

    src/lc_soliton/

### User-facing entry points

    static.py
    timedependent.py
    eigensoliton_runner.py
    engine.py
    request.py

These provide the primary interfaces used by the GUI, notebooks, and scripts.

## Core infrastructure

    src/lc_soliton/core/

Shared utilities including:

- context construction
- launch generation
- storage helpers
- backend selection
- bias calculations

## Validated computational kernels

    src/lc_soliton/validated_core/

Contains validated implementations used by the production runners.

Examples include:

- runner_core.py
- launch_core.py
- eigenmode_core.py
- stability_core.py

## Legacy validated code

    src/lc_soliton/legacy_validated/

Preserved for reproducibility and comparison with trusted historical results.

This code is not intended to be the primary development target.

## Physics modules

    src/lc_soliton/physics/

Physics-specific helper routines.

## Validation

    src/lc_soliton/validation/
    validation/

Reference cases and validation assets.

## GUI

    app/app.py

Primary Streamlit application.

## Examples

    examples/

Small demonstration scripts and workflows.

## Tests

    tests/

Automated tests and smoke tests.

## Documentation

    docs/

User, developer, and theory documentation.

## Generated Outputs

    runs/
    small_runs/

Run directories, diagnostics, movies, metadata, and reproducibility artifacts.

Generated outputs are ignored by git.