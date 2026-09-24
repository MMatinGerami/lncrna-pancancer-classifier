"""How many lncRNAs are needed? CV macro-F1 vs number of features (training split only)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from lncpan.config import load_config
from lncpan.io import load_dataset
from lncpan.models import build_model


def main() -> None:
    cfg = load_config()
    cv = StratifiedKFold(cfg["split"]["cv_folds"], shuffle=True, random_state=cfg.seed)
    rows = []
    for universe in ("lncRNA", "protein_coding"):
        X, y = load_dataset(cfg, universe).split("train")
        for k in cfg["features"]["budget_grid"]:
            model = build_model("logreg", k, cfg.seed)
            scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
            rows += [
                {"universe": universe, "k": k, "fold": i, "macro_f1": s}
                for i, s in enumerate(scores)
            ]
            print(
                f"{universe:15s} k={k:5d} F1={scores.mean():.3f} ± {scores.std():.3f}", flush=True
            )
    pd.DataFrame(rows).to_csv(cfg.path("results") / "tables" / "feature_budget.csv", index=False)


if __name__ == "__main__":
    main()
