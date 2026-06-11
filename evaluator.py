"""
Retrieval confidence scoring via concentration inequalities.

Ported from cautious-rag/cautious_rag/{bounds,decision}. Given a list of
cosine-similarity scores for retrieved documents, returns a scalar confidence
value in [0, 1] suitable for the graph's routing threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Concentration bounds (inline, no dependency on cautious-rag package)
# ---------------------------------------------------------------------------

def _hoeffding_lower(scores: np.ndarray, delta: float) -> float:
    """Hoeffding lower bound: mean - sqrt(-ln(δ/2) / (2n))."""
    n = len(scores)
    if n == 0:
        return 0.0
    eps = math.sqrt(-math.log(delta / 2) / (2 * n))
    return float(np.mean(scores) - eps)


def _bernstein_lower(scores: np.ndarray, delta: float) -> float:
    """Empirical Bernstein lower bound (tighter when variance is low)."""
    n = len(scores)
    if n < 2:
        return _hoeffding_lower(scores, delta)
    variance = float(np.var(scores, ddof=1))
    if variance == 0:
        return _hoeffding_lower(scores, delta)
    log_term = math.log(2 / delta)
    eps = math.sqrt(2 * variance * log_term / n) + log_term / (3 * n)
    return float(np.mean(scores) - eps)


def _adaptive_lower(scores: np.ndarray, delta: float) -> float:
    """Pick the tightest (highest) lower bound across Hoeffding and Bernstein."""
    h = _hoeffding_lower(scores, delta)
    b = _bernstein_lower(scores, delta)
    return max(h, b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

BoundType = Literal["hoeffding", "bernstein", "adaptive"]


@dataclass
class ConfidenceResult:
    score: float          # lower bound used as confidence (clipped to [0, 1])
    mean_similarity: float
    lower_bound: float
    bound_used: str
    n_docs: int


def compute_confidence(
    similarity_scores: list[float],
    *,
    confidence_level: float = 0.95,
    bound: BoundType = "adaptive",
) -> ConfidenceResult:
    """
    Compute a confidence score from cosine-similarity scores of retrieved docs.

    Args:
        similarity_scores: Per-document cosine similarity in [0, 1].
        confidence_level:  Statistical confidence (1 - δ).
        bound:             Which concentration bound to apply.

    Returns:
        ConfidenceResult with .score in [0, 1].
    """
    scores = np.array(similarity_scores, dtype=float)
    delta = 1.0 - confidence_level

    if len(scores) == 0:
        return ConfidenceResult(
            score=0.0,
            mean_similarity=0.0,
            lower_bound=0.0,
            bound_used="none",
            n_docs=0,
        )

    if bound == "hoeffding":
        lb = _hoeffding_lower(scores, delta)
        bound_used = "hoeffding"
    elif bound == "bernstein":
        lb = _bernstein_lower(scores, delta)
        bound_used = "bernstein"
    else:
        h = _hoeffding_lower(scores, delta)
        b = _bernstein_lower(scores, delta)
        if b > h:
            lb, bound_used = b, "bernstein"
        else:
            lb, bound_used = h, "hoeffding"

    return ConfidenceResult(
        score=float(np.clip(lb, 0.0, 1.0)),
        mean_similarity=float(np.mean(scores)),
        lower_bound=lb,
        bound_used=bound_used,
        n_docs=len(scores),
    )
