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
