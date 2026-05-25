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

## Recent additions

- Public reference-case loading API:
      load_reference_case("strict_static_centroid_drift")

- CLI reference-case display:
      lc-soliton --show-reference strict_static_centroid_drift

- CLI engine mode listing:
      lc-soliton --list-modes

- Engine dispatcher:
      run_engine(mode, ...)

- Engine modes can be queried with:
      available_engine_modes()

- Streamlit GUI now uses the public reference-case API.

- Public runners write environment.json into run directories.

## Next recommended target

Create true reference-case execution API:

    run_reference_case("strict_static_centroid_drift", run_dir=...)

This should regenerate the trusted case through a stable package entry point,
rather than requiring direct access to transitional legacy bridge machinery.

## Reference-case API convergence

The reference-case system now has a public dispatcher:

    run_reference_case("strict_static_centroid_drift")

The following user-facing paths should use this shared API:

- CLI:
      lc-soliton --run-reference strict_static_centroid_drift
      lc-soliton --validate-reference

- GUI:
      Validate trusted reference case

- tests:
      reference CLI consistency tests

This reduces duplicated validation logic and keeps CLI/GUI behavior aligned.

## Structured request milestone

The package now includes a canonical request layer:

    SimulationRequest

with structured sections:

    GridRequest
    GeometryRequest
    MaterialRequest
    LaunchRequest
    SolverRequest
    OutputRequest
    RuntimeRequest

Request-driven execution is available through:

    run_engine(SimulationRequest(...))
    lc-soliton --run-request request.json

Request-driven runs persist:

    request.json
    metadata.json
    environment.json

Requests are validated before execution. Current validation checks include
positive grid sizes, positive dz, and positive launch power.

The legacy flat params dictionary remains available as a bridge while structured
request sections are gradually wired into execution.

## Eigensoliton profile API milestone

The package now exposes initial public eigensoliton profile utilities:

    list_eigensoliton_profiles(run_dir)
    load_eigensoliton_profile(profile_path)

and CLI access:

    lc-soliton --list-eigensoliton-profiles RUN_DIR

This is a discovery/loading layer only. The full eigensoliton solver and
existence-curve machinery still need to be migrated into package-native
public APIs.

## GUI engine mode milestone

The Streamlit GUI now exposes all public engine modes:

    static
    time_dependent
    time_dependent_dual_grid

All three modes can be launched through the request-driven GUI path. The next
GUI work should focus on mode-specific result visualization, especially avoiding
misleading flattened z/time plots for time-dependent runs.
