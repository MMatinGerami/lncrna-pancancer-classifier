import numpy as np
import pytest
from sklearn.datasets import make_classification

from lncpan.evaluate import evaluate, expected_calibration_error
from lncpan.features import TopVarianceSelector
from lncpan.isolation import run_isolated
from lncpan.models import MODEL_NAMES, build_model


@pytest.fixture(scope="module")
def toy():
    X, y = make_classification(300, 40, n_informative=10, n_classes=4, random_state=0)
    return X.astype(np.float32), y


def test_top_variance_uses_training_data_only():
    X = np.zeros((10, 3), dtype=np.float32)
    X[:, 2] = np.arange(10)
    sel = TopVarianceSelector(k=1).fit(X)
    assert sel.support_.tolist() == [2]
    X_new = np.column_stack([np.arange(5) * 100, np.zeros(5), np.zeros(5)]).astype(np.float32)
    assert np.array_equal(sel.transform(X_new), X_new[:, [2]])


def _fit_predict(name, X, y):
    model = build_model(name, k=20, seed=0)
    if name == "mlp":
        model.set_params(clf__max_epochs=5, clf__device="cpu")
    if name == "xgboost":
        model.set_params(clf__n_estimators=10)
    return model.fit(X, y).predict_proba(X)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_fit_and_predict_proba(toy, name):
    X, y = toy
    proba = run_isolated(_fit_predict, name, X, y)
    assert proba.shape == (len(X), 4)
    np.testing.assert_allclose(proba.sum(1), 1, rtol=1e-4)


def test_ece_perfectly_calibrated_is_zero():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, proba) == pytest.approx(0.0)


def test_evaluate_returns_ci(toy):
    X, y = toy
    proba = np.full((len(y), 4), 0.25)
    out = evaluate(y, proba, n_boot=50).set_index("metric")
    assert (out["ci_low"] <= out["value"] + 1e-9).all()
    assert (out["value"] <= out["ci_high"] + 1e-9).all()
