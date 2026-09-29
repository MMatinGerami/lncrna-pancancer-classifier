"""How much hospital signal do the expression profiles carry?

scripts/11 showed that holding out tissue source sites costs about four points of macro-F1.
This script measures the batch signal directly: within each cancer type, can the site a
tumour came from be predicted from its expression? For every cancer type with at least
three sites of 15 or more training tumours, a logistic regression is cross-validated on
the site label (five stratified folds, feature selection inside each fold) for the lncRNA
and protein-coding gene sets, and the balanced accuracy is compared with chance (1 / number
of sites). A value far above chance means the profiles carry a site signature that a
cancer-type classifier can pick up; the comparison between gene sets asks whether lncRNAs
carry more or less of it.

Writes results/tables/site_signal.csv and figures/fig15_site_signal.png.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from lncpan.config import load_config
from lncpan.io import load_dataset
from lncpan.models import build_model
from lncpan.plotting import INK_2, INK_3, UNIVERSE_COLORS, UNIVERSE_LABELS, apply_style

MIN_SITE_N = 15
MIN_SITES = 3
N_FOLDS = 5


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures = res / "tables", res / "figures"
    k = cfg["features"]["top_k_variance"]
    rows = []
    for universe in ("lncRNA", "protein_coding"):
        ds = load_dataset(cfg, universe)
        X, y = ds.split("train")
        site = pd.Series([s.split("-")[1] for s in X.index], index=X.index)
        for c, cancer in enumerate(ds.classes):
            m = y == c
            counts = site[m].value_counts()
            keep_sites = counts[counts >= MIN_SITE_N].index
            if len(keep_sites) < MIN_SITES:
                continue
            sel = m & site.isin(keep_sites).to_numpy()
            Xs, ys = X[sel], site[sel].to_numpy()
            pred = np.empty_like(ys)
            cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=cfg.seed)
            for tr, te in cv.split(Xs, ys):
                model = build_model("logreg", k, cfg.seed).fit(Xs.iloc[tr], ys[tr])
                pred[te] = model.predict(Xs.iloc[te])
            rows.append(
                {
                    "universe": universe,
                    "cancer_type": cancer,
                    "n": int(sel.sum()),
                    "n_sites": len(keep_sites),
                    "chance": 1 / len(keep_sites),
                    "balanced_accuracy": balanced_accuracy_score(ys, pred),
                }
            )
            print(
                f"{universe} {cancer}: {len(keep_sites)} sites, balanced accuracy "
                f"{rows[-1]['balanced_accuracy']:.3f} (chance {rows[-1]['chance']:.3f})",
                flush=True,
            )
    out = pd.DataFrame(rows)
    out["above_chance"] = out["balanced_accuracy"] - out["chance"]
    out.to_csv(tables / "site_signal.csv", index=False)
    print(
        out.groupby("universe")[["balanced_accuracy", "chance", "above_chance"]]
        .mean()
        .round(3)
        .to_string()
    )
    plot(out, figures)


def plot(out: pd.DataFrame, figures) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    order = out[out.universe == "lncRNA"].sort_values("balanced_accuracy")["cancer_type"].tolist()
    x = np.arange(len(order))
    for universe, dx in (("lncRNA", -0.2), ("protein_coding", 0.2)):
        d = out[out.universe == universe].set_index("cancer_type").loc[order]
        ax.bar(
            x + dx,
            d["balanced_accuracy"],
            width=0.4,
            color=UNIVERSE_COLORS[universe],
            label=UNIVERSE_LABELS[universe],
        )
    chance = out[out.universe == "lncRNA"].set_index("cancer_type").loc[order]
    ax.scatter(x, chance["chance"], marker="_", s=300, color=INK_3, zorder=3, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n{int(n)} sites" for c, n in zip(order, chance["n_sites"], strict=True)],
        fontsize=6.5,
    )
    ax.set(
        ylim=(0, 1.02),
        ylabel="Balanced accuracy (5-fold CV)",
        title="Predicting the tissue source site within a cancer type",
    )
    ax.set_xlabel(
        "Cancer type (training tumours at sites with at least 15; sorted by lncRNA accuracy)",
        color=INK_2,
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(figures / "fig15_site_signal.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
