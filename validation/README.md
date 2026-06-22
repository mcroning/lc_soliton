# Validation

This directory contains trusted validation assets used to verify that package changes do not alter established numerical behavior.

The goal of validation is not to prove the physics is correct, but to detect unintended changes to trusted solver behavior.

## Trusted Reference Case

Current packaged reference case:

    strict_static_centroid_drift

This case is used throughout the package for regression testing and validation.

Run:

    lc-soliton --validate-reference

to verify that the installed package reproduces the stored reference metrics.

## Smoke Suite

A lightweight smoke test is provided for cluster and development environments:

    python scripts/run_cluster_smoke_suite.py

The smoke suite exercises:

- static solver
- time-dependent solver
- dual-grid time-dependent solver

The smoke suite is intended to catch installation, API, and execution failures.

It is not a physics validation benchmark.

## Expected Validation Behavior

Successful validation should reproduce the trusted reference case within established tolerances.

Typical diagnostics include:

- residual RMS
- residual maximum
- centroid drift
- convergence status

Small floating-point differences between platforms are expected.

## Reference Assets

This directory may contain:

- trusted reference metrics
- validation scripts
- small reference datasets
- regeneration instructions

Large simulation outputs should not be committed to git.

## Philosophy

Validation answers the question:

    "Does the current package behave like the trusted package?"

It does not answer:

    "Is the underlying physical model correct?"

Scientific validation of the physical model remains a separate research task.

## Future Validation Targets

Planned additions include:

- existence-curve regression cases
- eigensoliton profile validation
- time-dependent stability benchmarks
- cross-platform CPU/GPU consistency checks
- long-propagation regression tests
