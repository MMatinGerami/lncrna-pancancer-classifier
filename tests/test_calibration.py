import numpy as np
import pytest
from scipy.special import softmax

from lncpan.calibration import apply_temperature, fit_temperature, reliability_curve
from lncpan.evaluate import expected_calibration_error


def _data(scale: float, n=5000, k=8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 2.0
    return softmax(logits * scale, axis=1), y


def test_overconfident_probabilities_get_a_temperature_above_one():
    p, y = _data(scale=3.0)
    t = fit_temperature(p, y)
    assert t > 1.5
    assert expected_calibration_error(y, apply_temperature(p, t)) < expected_calibration_error(y, p)


def test_underconfident_probabilities_get_a_temperature_below_one():
    p, y = _data(scale=0.4)
    assert fit_temperature(p, y) < 0.8


def test_temperature_one_is_identity_and_argmax_is_preserved():
    p, _ = _data(scale=1.0, n=50)
    assert np.allclose(apply_temperature(p, 1.0), p)
    assert (apply_temperature(p, 4.0).argmax(1) == p.argmax(1)).all()


def test_reliability_curve_bins():
    correct = np.array([1, 0, 1, 1])
    conf = np.array([0.95, 0.92, 0.55, 0.51])
    acc, mean_conf, n = reliability_curve(correct, conf, n_bins=10)
    assert n[9] == 2 and n[5] == 2 and n.sum() == 4
    assert acc[9] == pytest.approx(0.5)
    assert mean_conf[5] == pytest.approx(0.53)
