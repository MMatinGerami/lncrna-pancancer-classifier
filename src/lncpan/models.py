"""Model zoo: every model is a Pipeline that starts with leakage-safe feature selection.

XGBoost and PyTorch are imported lazily (see `lncpan.isolation` for why).
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lncpan.features import TopVarianceSelector

MODEL_NAMES = ("logreg", "xgboost", "mlp")


def build_model(name: str, k: int, seed: int) -> Pipeline:
    """Return an unfitted pipeline. Hyperparameters are tuned via `param_grid`."""
    select = ("select", TopVarianceSelector(k=k))
    if name == "logreg":
        clf = LogisticRegression(C=0.1, max_iter=3000, random_state=seed)
        return Pipeline([select, ("scale", StandardScaler()), ("clf", clf)])
    if name == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            tree_method="hist", eval_metric="mlogloss", n_jobs=-1, random_state=seed
        )
        return Pipeline([select, ("clf", clf)])
    if name == "mlp":
        from lncpan.mlp import TorchMLPClassifier

        clf = TorchMLPClassifier(random_state=seed)
        return Pipeline([select, ("scale", StandardScaler()), ("clf", clf)])
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def param_grid(name: str, model_cfg: dict[str, Any]) -> dict[str, list]:
    """Prefix the config grid with the pipeline step name expected by GridSearchCV."""
    grid = model_cfg[name]
    if name == "mlp":
        grid = {**grid, "hidden_dims": [tuple(h) for h in grid["hidden_dims"]]}
    return {f"clf__{k}": list(v) for k, v in grid.items()}
