# Changelog

## 0.0.1 - initial packaging work

Initial preservation and packaging release.

Added:
- source rescue snapshot outside the repository
- installable `lc-soliton` package skeleton
- validated legacy engine imported under `legacy_validated`
- public wrapper modules for configuration, IO, solvers, diagnostics, plotting, and runners
- canonical runner API exposing `LCParams` and `run_lc_validated`
- command-line entry point: `lc-soliton --summary`
- import, CPU, GPU, and configuration smoke tests
- tiny infrastructure example and validation script
- initial documentation stubs
