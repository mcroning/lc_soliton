# Architecture roadmap

## Current state

The package currently wraps validated research code under:

    lc_soliton.legacy_validated

Public APIs are being stabilized before major internal refactoring.

## Planned long-term structure

    lc_soliton/
        api/
        physics/
        numerics/
        propagation/
        validation/
        gui/
        io/

## Design philosophy

1. Preserve validated physics first
2. Add regression tests before refactoring
3. Separate public API from implementation details
4. Keep GPU/CPU portability
5. Keep reproducibility as a first-class feature

## Near-term goals

- stable public APIs
- trusted regression cases
- GUI based on stable APIs
- documented parameter system

## Long-term goals

- removal of notebook-era bridges
- replacement of legacy helper injection
- modular propagation engine
- cleaner backend abstraction
- publication-quality documentation
