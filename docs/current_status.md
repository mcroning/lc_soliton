# Current project status

## Branch

Active development branch:

    rescue-package-20260522

## Milestones

- v0.0.1-rescue: initial rescue/package skeleton
- v0.0.2-public-api: public API and validation milestone

## Current strengths

- installable package
- public APIs for static, TD, dual-grid TD, and run loading
- CLI commands for summary, environment reporting, and validation
- trusted strict-static reference case
- reference validation command
- Streamlit GUI prototype
- lightweight package import
- run provenance via environment.json
- portability roadmap
- development workflow documentation

## Known limitations

- trusted reference execution still depends on legacy bridge machinery
- GUI currently displays reference artifacts but should not call legacy internals directly
- eigensoliton / existence-curve machinery still needs clean package integration
- Streamlit cluster access may require OnDemand/HPC Desktop workflows
- legacy_validated remains transitional

## Next likely development target

Create a stable package API for trusted reference execution:

    run_reference_case("strict_static_centroid_drift")

or:

    run_static_reference_case(config_path, run_dir)

This should become the shared entry point for:
- CLI
- validation
- GUI
- notebooks
