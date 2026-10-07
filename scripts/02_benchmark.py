"""Tune (5-fold CV on training split), then evaluate each model x gene universe on the test split.

Usage:
  python scripts/02_benchmark.py [--models logreg xgboost mlp]
                                 [--universes lncRNA protein_coding]
                                 [--config configs/default.yaml] [--seed 42]
                                 [--processed DIR] [--results DIR]

The last four options exist for the Nextflow pipeline (main.nf), which runs every
universe x model x seed combination as its own task in its own directory.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from lncpan.config import Config, load_config, with_overrides
from lncpan.evaluate import evaluate
from lncpan.io import load_dataset
from lncpan.isolation import run_isolated
from lncpan.models import MODEL_NAMES, build_model, param_grid


def run_one(cfg: Config, universe: str, name: str) -> None:
    """Tune, refit and evaluate one model on one gene universe (runs in its own process)."""
    res = cfg.path("results")
    k = cfg["features"]["top_k_variance"]
    cv = StratifiedKFold(cfg["split"]["cv_folds"], shuffle=True, random_state=cfg.seed)
    ds = load_dataset(cfg, universe)
    X_tr, y_tr = ds.split("train")
    X_te, y_te = ds.split("test")
    tag = f"{universe}__{name}"
    t0 = time.time()
    # sklearn-level parallelism only for logreg; XGBoost threads internally, the MLP uses the GPU
    search = GridSearchCV(
        build_model(name, k, cfg.seed),
        param_grid(name, cfg["models"]),
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1 if name == "logreg" else 1,
        refit=True,
    )
    search.fit(X_tr, y_tr)
    proba = search.predict_proba(X_te)
    metrics = evaluate(y_te, proba, seed=cfg.seed).assign(universe=universe, model=name)
    metrics.to_csv(res / "tables" / f"metrics__{tag}.csv", index=False)
    pd.DataFrame(proba, index=X_te.index, columns=ds.classes).assign(
        true=ds.classes[y_te]
    ).to_parquet(res / "predictions" / f"test__{tag}.parquet")
    pd.DataFrame(search.cv_results_).drop(columns="params").to_csv(
        res / "tables" / f"cv__{tag}.csv", index=False
    )
    joblib.dump(search.best_estimator_, res / "models" / f"{tag}.joblib")
    summary = {
        "universe": universe,
        "model": name,
        "seed": cfg.seed,
        "best_params": search.best_params_,
        "cv_macro_f1": search.best_score_,
        "minutes": (time.time() - t0) / 60,
    }
    (res / "tables" / f"search__{tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    f1 = metrics.set_index("metric").loc["macro_f1"]
    print(
        f"{tag}: CV F1 {search.best_score_:.3f} | test F1 {f1.value:.3f} "
        f"[{f1.ci_low:.3f}-{f1.ci_high:.3f}] | {summary['minutes']:.1f} min",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    ap.add_argument("--universes", nargs="+", default=["lncRNA", "protein_coding"])
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed")
    ap.add_argument("--processed", default=None, help="directory with the processed matrices")
    ap.add_argument("--results", default=None, help="output directory")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    cfg = with_overrides(cfg, args.seed, processed=args.processed, results=args.results)
    res = cfg.path("results")
    for sub in ("models", "predictions", "tables"):
        (res / sub).mkdir(exist_ok=True)
    for universe in args.universes:
        for name in args.models:
            run_isolated(run_one, cfg, universe, name)


if __name__ == "__main__":
    main()
