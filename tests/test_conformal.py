import numpy as np
import pytest

from lncpan.conformal import (
    class_conditional_quantiles,
    nonconformity,
    nonconformity_all,
    per_class_coverage,
    prediction_sets,
    quantile,
    summarize,
    tune_raps,
)


def _softmax_data(n=4000, k=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 2.0  # informative but imperfect probabilities
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True), y


def test_aps_score_is_cumulative_probability_down_to_true_class():
    p = np.array([[0.5, 0.3, 0.2]])
    assert nonconformity(p, np.array([1]), "aps") == pytest.approx(0.8)
    assert nonconformity(p, np.array([0]), "lac") == pytest.approx(0.5)


def test_aps_scores_of_all_classes_are_ordered_like_probabilities():
    p = np.array([[0.1, 0.6, 0.3]])
    s = nonconformity_all(p, "aps")[0]
    assert s[1] < s[2] < s[0]
    assert s[0] == pytest.approx(1.0)


def test_quantile_is_finite_sample_corrected():
    scores = np.arange(1, 11) / 10  # n = 10
    # ceil(11 * 0.9) = 10 -> the largest score
    assert quantile(scores, 0.1) == pytest.approx(1.0)
    # ceil(11 * 0.5) = 6 -> the 6th smallest
    assert quantile(scores, 0.5) == pytest.approx(0.6)
    assert quantile(scores, 0.0) == float("inf")


@pytest.mark.parametrize("score", ["lac", "aps"])
def test_marginal_coverage_holds_on_exchangeable_data(score):
    p, y = _softmax_data()
    half = len(y) // 2
    q = quantile(nonconformity(p[:half], y[:half], score), alpha=0.1)
    out = summarize(prediction_sets(p[half:], q, score), y[half:])
    assert out["coverage"] >= 0.88  # 0.90 nominal, sampling noise on 2,000 test points
    assert 1.0 <= out["mean_size"] < p.shape[1]


def test_class_conditional_quantiles_cover_every_class():
    p, y = _softmax_data(seed=1)
    half = len(y) // 2
    scores = nonconformity(p[:half], y[:half], "lac")
    q = class_conditional_quantiles(scores, y[:half], 0.1, p.shape[1])
    sets = prediction_sets(p[half:], q, "lac")
    cov = per_class_coverage(sets, y[half:], np.arange(p.shape[1]))
    assert (cov["coverage"] >= 0.85).all()


def test_summarize_counts_empty_sets_as_abstentions():
    sets = np.array([[True, False], [False, False], [False, True]])
    out = summarize(sets, np.array([0, 0, 0]))
    assert out["empty_frac"] == pytest.approx(1 / 3)
    assert out["coverage"] == pytest.approx(1 / 3)
    assert out["coverage_nonempty"] == pytest.approx(1 / 2)


def test_class_with_no_calibration_samples_is_always_included():
    q = class_conditional_quantiles(np.array([0.2, 0.3]), np.array([0, 0]), 0.1, n_classes=2)
    assert q[1] == float("inf")
    sets = prediction_sets(np.array([[0.9, 0.1]]), q, "lac")
    assert sets[0, 1]


def test_raps_with_zero_penalty_is_aps():
    p, _ = _softmax_data(n=50)
    assert np.allclose(nonconformity_all(p, "raps", k_reg=2, lam=0.0), nonconformity_all(p, "aps"))


def test_raps_penalises_only_classes_ranked_below_k_reg():
    p = np.array([[0.5, 0.3, 0.15, 0.05]])
    aps = nonconformity_all(p, "aps")[0]
    raps = nonconformity_all(p, "raps", k_reg=2, lam=0.1)[0]
    assert raps[0] == pytest.approx(aps[0])
    assert raps[1] == pytest.approx(aps[1])
    assert raps[2] == pytest.approx(aps[2] + 0.1)
    assert raps[3] == pytest.approx(aps[3] + 0.2)


def test_raps_sets_are_no_larger_than_aps_sets_on_a_long_tail():
    p, y = _softmax_data(n=6000, k=30, seed=3)
    tune, cal, test = slice(0, 1000), slice(1000, 3000), slice(3000, None)
    k_reg, lam = tune_raps(p[tune], y[tune], alpha=0.1)
    assert lam > 0
    q_aps = quantile(nonconformity(p[cal], y[cal], "aps"), 0.1)
    q_raps = quantile(nonconformity(p[cal], y[cal], "raps", k_reg, lam), 0.1)
    aps = summarize(prediction_sets(p[test], q_aps, "aps"), y[test])
    raps = summarize(prediction_sets(p[test], q_raps, "raps", k_reg, lam), y[test])
    assert raps["coverage"] >= 0.88
    assert raps["mean_size"] <= aps["mean_size"]
