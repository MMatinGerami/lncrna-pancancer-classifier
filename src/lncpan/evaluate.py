"""Metrics with bootstrap confidence intervals, and calibration."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: weighted mean |accuracy - confidence| over confidence bins."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _metrics(n_classes: int) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    labels = np.arange(n_classes)
    return {
        "accuracy": lambda y, p: accuracy_score(y, p.argmax(1)),
        "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, p.argmax(1)),
        "macro_f1": lambda y, p: f1_score(
            y, p.argmax(1), average="macro", labels=labels, zero_division=0
        ),
        "top3_accuracy": lambda y, p: top_k_accuracy_score(y, p, k=3, labels=labels),
        "log_loss": lambda y, p: log_loss(y, np.clip(p, 1e-7, 1), labels=labels),
        "ece": expected_calibration_error,
    }


def evaluate(y: np.ndarray, proba: np.ndarray, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Point estimates and percentile bootstrap 95% CIs (resampling test samples)."""
    y = np.asarray(y)
    fns = _metrics(proba.shape[1])
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in fns}
    # rare classes can be absent from a bootstrap resample; sklearn warns, the metric is still valid
    warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for k, f in fns.items():
            boots[k].append(f(y[idx], proba[idx]))
    rows = []
    for k, f in fns.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        rows.append({"metric": k, "value": f(y, proba), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Macro-F1(a) - macro-F1(b) on the same test samples, with a paired bootstrap 95% CI.

    `p_two_sided` is the bootstrap proportion of resamples on the other side of zero, doubled.
    """
    y = np.asarray(y)
    labels = np.arange(proba_a.shape[1])

    def f1(idx, p):
        return f1_score(y[idx], p[idx].argmax(1), average="macro", labels=labels, zero_division=0)

    rng = np.random.default_rng(seed)
    full = np.arange(len(y))
    observed = f1(full, proba_a) - f1(full, proba_b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = f1(idx, proba_a) - f1(idx, proba_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": observed, "ci_low": lo, "ci_high": hi, "p_two_sided": min(p, 1.0)}
