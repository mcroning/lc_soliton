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
