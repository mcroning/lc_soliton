# Public API

These are the intended user-facing imports.

## Configuration

    from lc_soliton import LCParams

## Static propagation

    from lc_soliton import run_static

## Time-dependent propagation

    from lc_soliton import run_td

## Dual-grid time-dependent propagation

    from lc_soliton import run_dg_td

## Run loading

    from lc_soliton import load_run

## Validation

    from lc_soliton.validation import validate_static_centroid_drift

## Transitional internals

Code under:

    lc_soliton.legacy_validated

is preserved for reproducibility but should not be used in new user-facing
scripts unless no public API exists yet.

## Stability expectation

The following top-level imports are considered stable:

    from lc_soliton import RunConfig
    from lc_soliton import LCParams
    from lc_soliton import run_static
    from lc_soliton import run_td
    from lc_soliton import run_dg_td
    from lc_soliton import load_run

Tests protect these names so future refactors do not accidentally remove them.

## Engine dispatcher

A canonical dispatcher is available for higher-level tools:

    from lc_soliton import run_engine

Supported modes currently include:

    static
    time_dependent
    time_dependent_dual_grid

This is intended for CLI, GUI, notebooks, and future job schedulers that need a
single execution entry point.

The available engine modes can be queried programmatically:

    from lc_soliton import available_engine_modes

    print(available_engine_modes())

This is useful for GUIs, CLIs, and future job schedulers.

## CLI engine modes

Available execution modes can be listed from the command line:

    lc-soliton --list-modes

## Reference-case CLI commands

Reference cases can be inspected or validated from the command line.

Show stored metadata:

    lc-soliton --show-reference static_centroid_drift

Run/validate through the public reference-case dispatcher:

    lc-soliton --run-reference static_centroid_drift

Legacy convenience validation command:

    lc-soliton --validate-reference

## Reference-case loading API

Stored reference cases can be loaded from Python:

    from lc_soliton import load_reference_case

    data = load_reference_case("static_centroid_drift")

The returned dictionary may include:

    data["trusted_metrics"]
    data["summary"]
    data["reports"]
    data["config"]
    data["figures"]

GUIs and notebooks should use this API instead of hardcoding reference-case
filenames or directory layouts.

## Protected public surface

The following names are protected by regression tests:

    run_static
    run_td
    run_dg_td
    run_engine
    available_engine_modes
    load_reference_case
    summarize_reference_case
    run_reference_case
    available_reference_cases

If these names change, tests should fail and documentation should be updated.
