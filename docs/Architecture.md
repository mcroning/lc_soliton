# LC Soliton Numerical Architecture

**Status:** Living design document

This document defines the architectural principles for the next-generation
LC soliton package. It describes responsibilities of each subsystem,
their public interfaces, and—equally importantly—what they must *not*
know about.

---

# Design Principles

1. Separate **numerical engines** from **scientific workflows**.
2. Each package has a single responsibility.
3. Numerical engines contain algorithms—not applications.
4. Workflows orchestrate engines—they do not implement numerical methods.
5. Every architectural layer has its own validation suite.
6. Float64 correctness is established first; optimization comes afterwards.
7. Architecture should follow the physics rather than historical code.

---

# Overall Architecture

```text
                Request
                   │
                   ▼
              Context Builder
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   Theta Engine         Optics Engine
        │                     │
        └──────────┬──────────┘
                   ▼
             Diagnostics Layer
                   │
                   ▼
             Scientific Workflows
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
        CLI              Streamlit GUI
```

The arrows represent **dependencies**.

Dependencies should always point downward.

No package should depend on another package at the same architectural level.

---

# Context Layer

## Purpose

Construct validated simulation contexts from user requests.

Examples

- geometry
- material parameters
- grid
- derived constants
- backend selection

## Public API

- request
- context

## Must not know about

- CN
- Picard
- FFT
- optics
- eigensolitons
- stability

---

# Theta Engine

## Purpose

Solve the liquid-crystal director equation.

The theta engine owns **all numerical algorithms** for the LC PDE.

## Public API

```python
advance_theta_cn(...)
relax_theta_steady(...)
```

These are the only routines workflows should call.

## Internal modules

```text
operators.py
residuals.py
cn.py
nonlinear.py
engine.py
```

## Must not know about

- optics
- eigensolitons
- stability
- GUI
- CLI
- dual grid

---

# Optics Engine

## Purpose

Propagate an optical field through a refractive-index distribution.

## Planned Public API

```python
advance_optics(...)
```

## Internal modules

```text
operators.py
propagator.py
launch.py
diagnostics.py
engine.py
```

## Must not know about

- theta solver internals
- eigensoliton continuation
- stability analysis
- GUI
- CLI

---

# Diagnostics Layer

## Purpose

Provide standardized measurements shared by every workflow.

Examples

- PDE residual
- CN residual
- update norm
- power conservation
- centroid
- overlap
- propagation statistics

Diagnostics never modify the solution.

They only measure it.

---

# Dual Grid

## Purpose

Manage communication between coarse theta grids and fine optical grids.

Responsibilities

```text
restrict()

prolong()

allocate()

synchronize()
```

The dual-grid package should know nothing about

- PDEs
- optics
- eigensolitons

It only transfers arrays.

---

# Scientific Workflows

Workflows assemble numerical engines into scientific algorithms.

Current workflows

- Static
- Time-dependent

Planned workflows

- Eigensoliton continuation
- Stability analysis

A workflow should read almost like the Methods section of a paper.

Example

```python
theta = theta_seed

while not converged:

    theta = relax_theta_steady(...)

return theta
```

or

```python
for step in time:

    field = advance_optics(field, theta)

    theta = advance_theta_cn(theta, intensity)

    diagnostics(...)
```

No workflow should contain FFTs, Thomas solvers, CN assembly,
or nonlinear iteration.

---

# Validation Philosophy

Every architectural layer has acceptance tests.

Current validation ladder

| Test | Purpose |
|------:|---------|
| 15 | Residual identity |
| 16 | CN linear identity |
| 17 | Pure float64 Picard |
| 18 | One CN timestep |
| 19 | Frozen-I relaxation |
| 20 | Public theta API |
| 21 | Static workflow |
| 22 | Time-dependent workflow |
| 23 | Diagnostics |
| 24 | Workflow diagnostics refactor |

Future engines should follow the same pattern:

```
engine
    ↓
diagnostics
    ↓
workflow
    ↓
validation
```

---

# Coding Philosophy

Prefer

- simple modules
- explicit data flow
- readable orchestration
- validated numerical kernels

Avoid

- hidden coupling
- duplicated algorithms
- workflow-specific numerical code
- optimization before validation

---

# Long-Term Goal

The package should mirror the scientific concepts rather than the implementation.

A new researcher should be able to understand the workflows without reading the numerical methods.

Numerical methods belong in the engines.

Scientific intent belongs in the workflows.
