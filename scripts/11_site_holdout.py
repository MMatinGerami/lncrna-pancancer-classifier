"""Held-out hospital sites: does the classifier generalise to centres it never saw?

TCGA sample barcodes encode the tissue source site (characters 6 and 7). Random
cross-validation lets the same hospital appear in training and test folds, so a model can
score well partly by learning site-specific batch effects. Here the training split is
cross-validated twice with identical models: stratified folds drawn at random, and
stratified folds that keep every site in a single fold (StratifiedGroupKFold). The gap
between the two is the cost of moving to an unseen centre.

Macro-F1 in each fold is computed over the classes present in that fold's training data,
because a cancer type collected at a single site (LAML) cannot be learned when that site is
held out.

Writes results/tables/site_holdout.csv and figures/fig11_site_holdout.png.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from lncpan.config import load_config
from lncpan.data import parse_barcode
from lncpan.io import load_dataset
from lncpan.models import build_model
from lncpan.plotting import UNIVERSE_COLORS, UNIVERSE_LABELS, apply_style

N_FOLDS = 5


def site_of(sample: str) -> str:
    return sample.split("-")[1]


def fold_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    seen = np.unique(y_train)
    m = np.isin(y_true, seen)
    return f1_score(y_true[m], y_pred[m], average="macro", labels=seen, zero_division=0)


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures = res / "tables", res / "figures"
    k = cfg["features"]["top_k_variance"]
    rows = []
    for universe in ("lncRNA", "protein_coding"):
        ds = load_dataset(cfg, universe)
        X, y = ds.split("train")
        sites = np.array([site_of(s) for s in X.index])
        patients = np.array([parse_barcode(s)[0] for s in X.index])
        assert len(np.unique(patients)) == len(patients)
        best = json.loads((tables / f"search__{universe}__logreg.json").read_text())["best_params"]
        splitters = {
            "random": StratifiedKFold(N_FOLDS, shuffle=True, random_state=cfg.seed),
            "held-out sites": StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=cfg.seed),
        }
        for scheme, cv in splitters.items():
            for fold, (tr, te) in enumerate(cv.split(X, y, groups=sites)):
                model = build_model("logreg", k, cfg.seed)
                model.set_params(**best)
                model.fit(X.iloc[tr], y[tr])
                pred = model.predict(X.iloc[te])
                rows.append(
                    {
                        "universe": universe,
                        "scheme": scheme,
                        "fold": fold,
                        "n_test": len(te),
                        "n_test_sites": len(np.unique(sites[te])),
                        "sites_shared_with_train": int(np.isin(sites[te], sites[tr]).sum()),
                        "macro_f1": fold_macro_f1(y[te], pred, y[tr]),
                        "accuracy": float((pred == y[te]).mean()),
                    }
                )
                print(
                    f"{universe} {scheme} fold {fold}: macro-F1 {rows[-1]['macro_f1']:.3f}",
                    flush=True,
                )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "site_holdout.csv", index=False)
    summary = out.groupby(["universe", "scheme"])["macro_f1"].agg(["mean", "std"]).round(4)
    print(summary.to_string())
    plot(out, figures)


def plot(out: pd.DataFrame, figures) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    schemes = ["random", "held-out sites"]
    x = np.arange(len(schemes))
    for i, universe in enumerate(("lncRNA", "protein_coding")):
        d = out[out.universe == universe]
        means = [d[d.scheme == s]["macro_f1"].mean() for s in schemes]
        sds = [d[d.scheme == s]["macro_f1"].std() for s in schemes]
        ax.bar(
            x + (i - 0.5) * 0.36,
            means,
            yerr=sds,
            width=0.36,
            capsize=3,
            color=UNIVERSE_COLORS[universe],
            label=UNIVERSE_LABELS[universe],
        )
        for xi, s in zip(x, schemes, strict=True):
            for _, r in d[d.scheme == s].iterrows():
                ax.scatter(xi + (i - 0.5) * 0.36, r["macro_f1"], s=8, color="#0b0b0b", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Random folds", "Held-out sites"])
    ax.set(
        ylim=(0.85, 1.0),
        ylabel="Macro-F1 (5-fold CV, training split)",
        title="Generalisation to unseen hospital sites",
    )
    ax.legend(loc="upper right")
    fig.savefig(figures / "fig11_site_holdout.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
