import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from lncpan.cli import align_genes, predict_frame


@pytest.fixture
def toy_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"G{i}" for i in range(5)])
    y = (X["G0"] + X["G1"] > 0).astype(int).to_numpy()
    model = Pipeline([("clf", LogisticRegression())]).fit(X, y)
    thresholds = {
        "classes": ["A", "B"],
        "lac": {"0.1": {"marginal": 0.5, "class_conditional": [0.5, 0.5]}},
    }
    return model, thresholds, X


def test_align_genes_reorders_and_fills_missing(toy_model):
    model, _, X = toy_model
    shuffled = X[["G3", "G1", "G0"]].copy()
    aligned, missing = align_genes(shuffled, model)
    assert list(aligned.columns) == [f"G{i}" for i in range(5)]
    assert missing == ["G2", "G4"]
    assert (aligned["G2"] == 0).all()


def test_predict_frame_builds_sets(toy_model):
    model, thresholds, X = toy_model
    out = predict_frame(X.head(10), model, thresholds, alpha=0.1, class_conditional=False)
    assert list(out.columns) == ["sample", "predicted", "probability", "set_size", "prediction_set"]
    assert set(out["predicted"]) <= {"A", "B"}
    # threshold 0.5 on 1 - p means every class with p >= 0.5 is in the set: at least the top one
    assert (out["set_size"] >= 1).all()


def test_predict_frame_rejects_uncalibrated_alpha(toy_model):
    model, thresholds, X = toy_model
    with pytest.raises(SystemExit):
        predict_frame(X.head(2), model, thresholds, alpha=0.3, class_conditional=False)
