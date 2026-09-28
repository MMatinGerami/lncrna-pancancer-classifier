"""Learning curve: how many training tumours does each gene universe need?

Stratified subsamples of the training split (10% to 100%), the tuned logistic-regression
model, test macro-F1. Three random subsamples per fraction; the full split is run once.

Writes results/tables/learning_curve.csv and figures/fig10_learning_curve.png.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from lncpan.config import load_config
from lncpan.io import load_dataset
from lncpan.models import build_model
from lncpan.plotting import UNIVERSE_COLORS, UNIVERSE_LABELS, apply_style

FRACTIONS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
N_REPEATS = 3


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures = res / "tables", res / "figures"
    k = cfg["features"]["top_k_variance"]
    rows = []
    for universe in ("lncRNA", "protein_coding"):
        ds = load_dataset(cfg, universe)
        X_tr, y_tr = ds.split("train")
        X_te, y_te = ds.split("test")
        labels = np.arange(len(ds.classes))
        best = json.loads((tables / f"search__{universe}__logreg.json").read_text())["best_params"]
        for frac in FRACTIONS:
            repeats = 1 if frac == 1.0 else N_REPEATS
            for rep in range(repeats):
                if frac == 1.0:
                    idx = np.arange(len(y_tr))
                else:
                    sss = StratifiedShuffleSplit(1, train_size=frac, random_state=cfg.seed + rep)
                    idx, _ = next(sss.split(X_tr, y_tr))
                model = build_model("logreg", k, cfg.seed)
                model.set_params(**best)
                model.fit(X_tr.iloc[idx], y_tr[idx])
                f1 = f1_score(
                    y_te, model.predict(X_te), average="macro", labels=labels, zero_division=0
                )
                rows.append(
                    {
                        "universe": universe,
                        "fraction": frac,
                        "n_train": len(idx),
                        "repeat": rep,
                        "macro_f1": f1,
                    }
                )
                print(f"{universe} {frac:.2f} n={len(idx)} rep={rep} macro-F1 {f1:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(tables / "learning_curve.csv", index=False)
    plot(out, figures)


def plot(out: pd.DataFrame, figures) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for universe, sub in out.groupby("universe"):
        g = sub.groupby("n_train")["macro_f1"].agg(["mean", "min", "max"]).reset_index()
        ax.plot(
            g["n_train"],
            g["mean"],
            "o-",
            color=UNIVERSE_COLORS[universe],
            label=UNIVERSE_LABELS[universe],
        )
        ax.fill_between(
            g["n_train"], g["min"], g["max"], color=UNIVERSE_COLORS[universe], alpha=0.15, lw=0
        )
    ax.set(
        xscale="log",
        xlabel="Training tumours",
        ylabel="Test macro-F1",
        title="Learning curve (logistic regression)",
    )
    sizes = sorted(out["n_train"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes], fontsize=7)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(loc="lower right")
    fig.savefig(figures / "fig10_learning_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
