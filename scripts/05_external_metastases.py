"""External test: trace TCGA metastases (never seen in training) back to their tissue of origin."""

from __future__ import annotations

import joblib
import pandas as pd

from lncpan.config import load_config
from lncpan.evaluate import evaluate
from lncpan.io import load_dataset
from lncpan.isolation import run_isolated
from lncpan.models import MODEL_NAMES


def _predict(path, X):
    return joblib.load(path).predict_proba(X)


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    rows, preds = [], []
    for universe in ("lncRNA", "protein_coding"):
        ds = load_dataset(cfg, universe)
        X_ex, y_ex = ds.split("external")
        for name in MODEL_NAMES:
            path = res / "models" / f"{universe}__{name}.joblib"
            if not path.exists():
                continue
            proba = run_isolated(_predict, path, X_ex)
            m = evaluate(y_ex, proba, seed=cfg.seed).set_index("metric")
            rows.append(
                {
                    "universe": universe,
                    "model": name,
                    "n": len(y_ex),
                    **{
                        f"{k}{s}": m.at[met, col]
                        for met in ("accuracy", "top3_accuracy")
                        for k, s, col in [
                            (met, "", "value"),
                            (met, "_ci_low", "ci_low"),
                            (met, "_ci_high", "ci_high"),
                        ]
                    },
                }
            )
            preds.append(
                pd.DataFrame(
                    {
                        "universe": universe,
                        "model": name,
                        "true": ds.classes[y_ex],
                        "pred": ds.classes[proba.argmax(1)],
                    }
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(res / "tables" / "external_metastases.csv", index=False)
    wrong = pd.concat(preds).query("true != pred").groupby(["universe", "model", "pred"]).size()
    wrong.rename("n").reset_index().to_csv(
        res / "tables" / "external_misclassified_as.csv", index=False
    )
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
