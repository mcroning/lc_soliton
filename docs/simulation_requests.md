# Simulation requests

Simulation requests are JSON-friendly descriptions of a run.

They are represented in Python by:

    SimulationRequest
    OutputRequest
    RuntimeRequest

## Example request

    {
      "mode": "static",
      "params": {
        "Nx": 32,
        "Ny": 32,
        "Nz": 2,
        "static_max_steps": 2
      },
      "output": {
        "run_dir": "runs/example_request",
        "save_slices": false,
        "save_full": false
      },
      "runtime": {
        "backend": "auto",
        "progress": false
      }
    }

Run it with:

    lc-soliton --run-request request.json

## Purpose

Simulation requests provide a portable execution format for:

- CLI runs
- GUI runs
- Slurm jobs
- future container execution
- reference-case regeneration

## Request provenance

When a simulation is executed through:

    run_engine(SimulationRequest(...))

or:

    lc-soliton --run-request request.json

the request is saved into the output directory as:

    request.json

A request-driven run therefore contains:

    request.json
    metadata.json
    environment.json

This makes the run reproducible from the original execution request.

## Structured request sections

Simulation requests are gradually moving away from one flat parameter dictionary
toward structured sections.

Current structure:

    SimulationRequest
        grid: GridRequest
        solver: SolverRequest
        output: OutputRequest
        runtime: RuntimeRequest
        params: dict

`params` remains as a legacy bridge for options that have not yet been promoted
to structured request fields.

### GridRequest

    Nx
    Ny
    Nz

### SolverRequest

    static_max_steps
    Nt
    dt
    t_stride

Future request sections may include:
- geometry
- material
- optical launch
- boundary conditions

### GeometryRequest

    xaper_um
    yaper_um
    dz_um

`GeometryRequest` is now consumed by the engine request path and mapped into
the legacy execution parameters while the migration is ongoing.

### MaterialRequest

    ne
    no
    K
    De

`MaterialRequest` describes optical and elastic material constants. It is part
of the structured request schema but is not yet fully consumed by all execution
paths.

### LaunchRequest

    power_mW
    waist_um
    separation_um

`LaunchRequest` describes the optical launch/excitation configuration. It is
part of the structured request schema but is not yet fully consumed by all
execution paths.

## Migration status

Some request sections are already consumed by the engine:

    GridRequest
    GeometryRequest
    SolverRequest
    OutputRequest
    RuntimeRequest

Other sections currently serve as structured semantic metadata while the legacy
parameter bridge remains available:

    MaterialRequest
    LaunchRequest

## Schema introspection

The current high-level request schema can be inspected with:

    lc-soliton --request-schema

or from Python:

    from lc_soliton import simulation_request_schema

    schema = simulation_request_schema()

This is useful for GUIs, job launchers, documentation generation, and future
container or remote execution frontends.

## Request validation

Structured simulation requests are validated before request-driven execution.

Current validation checks include:

- positive grid sizes
- positive propagation step size
- positive launch power

Validation occurs when running through:

    run_engine(SimulationRequest(...))

or:

    lc-soliton --run-request request.json

This helps catch malformed requests before expensive simulations begin.
