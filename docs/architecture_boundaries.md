# Architecture boundaries

## Public stable layer

Stable user-facing APIs live under:

    src/lc_soliton/

Examples:
- run_static
- run_td
- run_dg_td
- validation APIs
- CLI commands

GUI and examples should prefer these interfaces.

## Transitional validated layer

Legacy validated implementation code lives under:

    src/lc_soliton/legacy_validated/

This layer exists to preserve:
- validated physics behavior
- reproducibility
- historical implementations

Direct usage from notebooks is discouraged for new development.

## Migration philosophy

Features should migrate:

    notebook experiment
        ->
    validated legacy implementation
        ->
    public package wrapper
        ->
    stable API
        ->
    GUI exposure

## Long-term goal

Eventually:
- GUI
- CLI
- tests
- examples
- notebooks

should all call the same stable public APIs.

The legacy_validated layer should become progressively thinner over time.
