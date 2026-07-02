# lc_soliton_next

Clean architecture port.

Rules:
- Algorithms consume only arrays, coefficients, controls, and buffers.
- Algorithms do not know about experiments, beams, GUI, files, or continuation.
- Physics modules describe the virtual laboratory and optical experiment.
- Workflows cook inputs for algorithms and collect products.
- `src/lc_soliton/validated_core/runner_core.py` is the oracle until replaced.
