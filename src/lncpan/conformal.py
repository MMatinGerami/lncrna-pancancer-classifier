"""Split conformal prediction sets for a multiclass classifier.

Given class probabilities from a model that never saw the calibration samples, a
non-conformity score per calibration sample and a target error rate `alpha`, the sets
built here contain the true class with probability at least 1 - alpha on exchangeable
test data (Vovk et al., 2005; Angelopoulos & Bates, 2021 for the two scores used).

Two scores are implemented:

- "lac": 1 - p(true class). Smallest sets on average, but coverage can be uneven across
  classes.
- "aps": cumulative probability of the classes at least as likely as the true class
  (adaptive prediction sets, without the randomised tie-break, so slightly conservative).

`class_conditional_quantiles` gives the Mondrian variant: one threshold per class, so the
guarantee holds for each class separately, which is what matters for rare cancer types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORES = ("lac", "aps")


def nonconformity(proba: np.ndarray, y: np.ndarray, score: str) -> np.ndarray:
    """One score per sample for its true class (higher = less conforming)."""
    return nonconformity_all(proba, score)[np.arange(len(y)), y]


def nonconformity_all(proba: np.ndarray, score: str) -> np.ndarray:
    """Score of every class for every sample, shape (n, K)."""
    if score == "lac":
        return 1.0 - proba
    if score == "aps":
        order = np.argsort(-proba, axis=1)
        sorted_p = np.take_along_axis(proba, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        out = np.empty_like(proba)
        np.put_along_axis(out, order, cum, axis=1)
        return out
    raise ValueError(f"unknown score {score!r}; choose from {SCORES}")


def quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of the calibration scores."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def class_conditional_quantiles(
    scores: np.ndarray, y: np.ndarray, alpha: float, n_classes: int
) -> np.ndarray:
    """One quantile per class (Mondrian conformal). Classes with no calibration sample get inf."""
    q = np.full(n_classes, np.inf)
    for c in range(n_classes):
        m = y == c
        if m.any():
            q[c] = quantile(scores[m], alpha)
    return q


def prediction_sets(proba: np.ndarray, q: float | np.ndarray, score: str) -> np.ndarray:
    """Boolean (n, K) matrix: class k is in the set when its score is <= q (or <= q[k])."""
    return nonconformity_all(proba, score) <= np.asarray(q)


def summarize(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Coverage and set-size statistics. An empty set is an abstention, so coverage is also
    reported on the non-empty sets alone."""
    size = sets.sum(1)
    covered = sets[np.arange(len(y)), y]
    nonempty = size > 0
    return {
        "coverage": float(covered.mean()),
        "coverage_nonempty": float(covered[nonempty].mean()) if nonempty.any() else float("nan"),
        "mean_size": float(size.mean()),
        "median_size": float(np.median(size)),
        "singleton_frac": float((size == 1).mean()),
        "empty_frac": float((size == 0).mean()),
    }


def per_class_coverage(sets: np.ndarray, y: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    covered = sets[np.arange(len(y)), y]
    rows = []
    for c, name in enumerate(classes):
        m = y == c
        if m.any():
            rows.append({"cancer_type": name, "n": int(m.sum()), "coverage": covered[m].mean()})
    return pd.DataFrame(rows)
