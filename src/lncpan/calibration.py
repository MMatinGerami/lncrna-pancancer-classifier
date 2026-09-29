"""Post-hoc probability calibration."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(
    proba: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 20.0)
) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T dividing the logits, chosen to
    minimise the negative log-likelihood on held-out samples. T > 1 softens over-confident
    probabilities, T < 1 sharpens under-confident ones."""
    logits = np.log(np.clip(proba, 1e-12, None))
    idx = np.arange(len(y))

    def nll(t: float) -> float:
        return float(-log_softmax(logits / t, axis=1)[idx, y].mean())

    return float(minimize_scalar(nll, bounds=bounds, method="bounded").x)


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-12, None))
    return softmax(logits / t, axis=1)


def reliability_curve(
    correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin accuracy, mean confidence and count over equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    acc, conf, count = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            acc[i], conf[i], count[i] = correct[m].mean(), confidence[m].mean(), m.sum()
    return acc, conf, count
