"""
Future public eigensoliton execution API.

This module will eventually host:

- packaged eigensoliton solves
- continuation/existence curves
- branch following
- profile polishing
- benchmark nonlinear eigenproblem cases

Current status:
    placeholder scaffold only
"""

from __future__ import annotations


def run_eigensoliton_case(*args, **kwargs):
    """
    Placeholder future public eigensoliton runner.
    """
    raise NotImplementedError(
        "Public eigensoliton execution API not yet migrated."
    )


__all__ = [
    "run_eigensoliton_case",
]
