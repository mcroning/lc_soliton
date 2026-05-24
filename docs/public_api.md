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

    from lc_soliton.validation import validate_strict_static_centroid_drift

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
