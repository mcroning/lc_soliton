"""
Public reference-case execution helpers.
"""

from __future__ import annotations

from .validation import validate_strict_static_centroid_drift


REFERENCE_CASES = {
    "strict_static_centroid_drift": validate_strict_static_centroid_drift,
}


def available_reference_cases():
    """
    Return sorted list of available public reference cases.
    """
    return sorted(REFERENCE_CASES)


def run_reference_case(name: str, *args, **kwargs):
    """
    Execute or validate a public reference case.
    """
    if name not in REFERENCE_CASES:
        raise ValueError(
            f"Unknown reference case: {name}. "
            f"Available cases: {sorted(REFERENCE_CASES)}"
        )

    return REFERENCE_CASES[name](*args, **kwargs)
