"""
Scoring utilities for handling both single-float and dual-metric eval_fn returns.

Some benchmarks (trip, meeting) return a dict:
    {"partial_score": float, "exact_score": float}
Others (calendar, olympiad, jssp) return a plain float.

This module provides helpers to normalize these into a consistent interface.
"""

from typing import Any, Dict, Union

ScoreResult = Union[float, Dict[str, float]]


def normalize_score(result: ScoreResult, key: str = "partial_score") -> float:
    """Extract a single float score from an eval_fn result.

    Args:
        result: either a float or a dict with "partial_score"/"exact_score".
        key: which key to extract when result is a dict.

    Returns:
        A float score in [0, 1].
    """
    if isinstance(result, dict):
        return float(result.get(key, 0.0))
    return float(result)


def is_dual_score(result: ScoreResult) -> bool:
    """Check whether an eval_fn result contains dual metrics."""
    return isinstance(result, dict)
