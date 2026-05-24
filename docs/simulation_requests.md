# Simulation requests

Simulation requests are JSON-friendly descriptions of a run.

They are represented in Python by:

    SimulationRequest
    OutputRequest
    RuntimeRequest

## Example request

    {
      "mode": "strict_static",
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
