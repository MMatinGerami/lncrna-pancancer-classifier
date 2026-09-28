"""Command-line prediction for new samples.

    lncpan predict expression.parquet [--alpha 0.1] [--class-conditional] [-o out.csv]

The input is a samples x genes table of log2(TPM + 1) lncRNA expression with Ensembl gene
IDs as columns (the same universe the model was trained on, `data/processed/genes.parquet`).
Genes missing from the input are filled with zero and reported. The output has, per sample,
the predicted cancer type, its probability, and the conformal prediction set at the requested
error rate, using the thresholds saved by scripts/08_conformal.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lncpan.conformal import prediction_sets

DEFAULT_MODEL_DIR = Path("results/models")


def load_artifacts(model_dir: Path, universe: str = "lncRNA"):
    thresholds = json.loads((model_dir / f"{universe}__conformal_thresholds.json").read_text())
    # joblib loads a scikit-learn pipeline produced by this repository's own scripts
    model = joblib.load(model_dir / f"{universe}__logreg_conformal.joblib")
    return model, thresholds


def align_genes(X: pd.DataFrame, model) -> tuple[pd.DataFrame, list[str]]:
    wanted = list(model.feature_names_in_)
    missing = [g for g in wanted if g not in X.columns]
    X = X.reindex(columns=wanted, fill_value=0.0).astype(np.float32)
    return X, missing


def predict_frame(
    X: pd.DataFrame, model, thresholds: dict, alpha: float, class_conditional: bool
) -> pd.DataFrame:
    classes = np.array(thresholds["classes"])
    proba = model.predict_proba(X)
    key = str(alpha)
    if key not in thresholds["lac"]:
        raise SystemExit(f"alpha {alpha} not calibrated; available: {list(thresholds['lac'])}")
    q = thresholds["lac"][key]["class_conditional" if class_conditional else "marginal"]
    sets = prediction_sets(proba, np.asarray(q), "lac")
    top = proba.argmax(1)
    return pd.DataFrame(
        {
            "sample": X.index,
            "predicted": classes[top],
            "probability": proba[np.arange(len(X)), top].round(4),
            "set_size": sets.sum(1),
            "prediction_set": [";".join(classes[row]) for row in sets],
        }
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lncpan")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="predict tissue of origin with a conformal set")
    p.add_argument("expression", type=Path, help="parquet or CSV, samples x Ensembl gene IDs")
    p.add_argument("--alpha", type=float, default=0.1, help="target error rate of the set")
    p.add_argument("--class-conditional", action="store_true", help="per-class thresholds")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument(
        "-o", "--output", type=Path, default=None, help="write CSV here instead of stdout"
    )
    args = ap.parse_args(argv)

    X = (
        pd.read_parquet(args.expression)
        if args.expression.suffix == ".parquet"
        else pd.read_csv(args.expression, index_col=0)
    )
    model, thresholds = load_artifacts(args.model_dir)
    X, missing = align_genes(X, model)
    if missing:
        print(
            f"{len(missing)} of {len(model.feature_names_in_)} model genes missing, filled with 0",
            file=sys.stderr,
        )
    out = predict_frame(X, model, thresholds, args.alpha, args.class_conditional)
    if args.output:
        out.to_csv(args.output, index=False)
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
