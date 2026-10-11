"""Are the marker lncRNAs stable, or an artefact of one training set?

scripts/04 lists the ten lncRNAs with the largest SHAP contribution to each cancer type.
A marker list read off a single fit can change with the training sample. Here the
logistic-regression model is refit on bootstrap resamples of the training split (feature
selection redone inside each fit), the ten largest standardised coefficients per class are
recorded each time, and two things are measured per cancer type:

- the selection frequency of each gene in the full-data top ten across resamples, and
- the Jaccard overlap between the resample's top ten and the full-data top ten.

The SHAP markers from the XGBoost model are then checked against the logistic-regression
lists: a gene that is a top marker under both models and in most resamples is a marker one
can trust; a gene that appears once is not.

Writes results/tables/marker_stability{,_per_gene}.csv and figures/fig14_marker_stability.png.
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lncpan.config import load_config
from lncpan.io import load_dataset
from lncpan.models import build_model
from lncpan.plotting import DOUBLE, INK, INK_3, MM, UNIVERSE_COLORS, apply_style, panel_label, save

UNIVERSE = "lncRNA"
N_BOOT = 30
TOP = 10


def top_genes(model, columns: pd.Index) -> list[np.ndarray]:
    """Per class, the TOP genes with the largest positive coefficient (standardised inputs)."""
    coef = model.named_steps["clf"].coef_
    selected = columns[model.named_steps["select"].support_]
    return [selected[np.argsort(-row)[:TOP]].to_numpy() for row in coef]


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures = res / "tables", res / "figures"
    ds = load_dataset(cfg, UNIVERSE)
    X, y = ds.split("train")
    best = json.loads((tables / f"search__{UNIVERSE}__logreg.json").read_text())["best_params"]
    k = cfg["features"]["top_k_variance"]

    full = build_model("logreg", k, cfg.seed).set_params(**best).fit(X, y)
    reference = top_genes(full, X.columns)

    rng = np.random.default_rng(cfg.seed)
    counts = [dict.fromkeys(genes, 0) for genes in reference]
    jaccard = np.zeros((N_BOOT, len(ds.classes)))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        model = build_model("logreg", k, cfg.seed + b).set_params(**best).fit(X.iloc[idx], y[idx])
        for c, genes in enumerate(top_genes(model, X.columns)):
            ref = set(reference[c])
            got = set(genes)
            jaccard[b, c] = len(ref & got) / len(ref | got)
            for g in ref & got:
                counts[c][g] += 1
        print(f"resample {b + 1}/{N_BOOT}: mean Jaccard {jaccard[b].mean():.3f}", flush=True)

    genes = pd.read_parquet(cfg.path("processed") / "genes.parquet").set_index("gene_id")
    shap = pd.read_csv(tables / "shap_markers_per_cancer.csv")
    per_gene = []
    for c, cancer in enumerate(ds.classes):
        shap_top = set(shap.loc[shap["cancer_type"] == cancer, "gene_id"])
        for rank, g in enumerate(reference[c], start=1):
            per_gene.append(
                {
                    "cancer_type": cancer,
                    "rank": rank,
                    "gene_id": g,
                    "gene_name": genes.at[g, "gene_name"],
                    "selection_frequency": counts[c][g] / N_BOOT,
                    "in_shap_top10": g in shap_top,
                }
            )
    per_gene = pd.DataFrame(per_gene)
    per_gene.to_csv(tables / "marker_stability_per_gene.csv", index=False)

    n_train = pd.Series(y).value_counts().reindex(range(len(ds.classes))).to_numpy()
    summary = pd.DataFrame(
        {
            "cancer_type": ds.classes,
            "n_train": n_train,
            "jaccard_mean": jaccard.mean(0),
            "jaccard_low": np.percentile(jaccard, 2.5, axis=0),
            "jaccard_high": np.percentile(jaccard, 97.5, axis=0),
            "markers_selected_in_90pct": per_gene.groupby("cancer_type")["selection_frequency"]
            .apply(lambda s: int((s >= 0.9).sum()))
            .reindex(ds.classes)
            .to_numpy(),
            "shap_overlap": per_gene.groupby("cancer_type")["in_shap_top10"]
            .sum()
            .reindex(ds.classes)
            .to_numpy(),
        }
    )
    summary.to_csv(tables / "marker_stability.csv", index=False)
    print(summary.round(3).to_string(index=False))
    stable = per_gene[per_gene["selection_frequency"] >= 0.9]
    print(
        f"{len(stable)} of {len(per_gene)} full-data markers are selected in at least 90% of "
        f"resamples; {int(per_gene['in_shap_top10'].sum())} are also SHAP top-10 markers"
    )
    plot(summary, per_gene, figures)


def plot(summary: pd.DataFrame, per_gene: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE * 0.9, 58 * MM), gridspec_kw={"width_ratios": [3, 1.1]}
    )
    colour = UNIVERSE_COLORS[UNIVERSE]
    ax = axes[0]
    d = summary.sort_values("n_train")
    x = np.arange(len(d))
    ax.bar(x, d["jaccard_mean"], color=colour, lw=0)
    ax.errorbar(
        x,
        d["jaccard_mean"],
        yerr=[d["jaccard_mean"] - d["jaccard_low"], d["jaccard_high"] - d["jaccard_mean"]],
        fmt="none",
        ecolor=INK,
        elinewidth=0.8,
    )
    ax.set_xticks(
        x,
        [f"{c} ({n})" for c, n in zip(d["cancer_type"], d["n_train"], strict=True)],
        rotation=90,
        fontsize=5,
    )
    ax.set(
        xlim=(-0.7, len(d) - 0.3),
        ylim=(0, 1.0),
        ylabel="Jaccard overlap, top-10 markers",
        xlabel="Cancer type (number of training tumours)",
    )
    panel_label(ax, "a", x=-0.08, y=1.06)
    ax = axes[1]
    ax.hist(
        per_gene["selection_frequency"],
        bins=np.linspace(0, 1, 11),
        color=colour,
        edgecolor="white",
        lw=0.4,
    )
    ax.axvline(0.9, color=INK_3, ls="--", lw=0.6)
    ax.set(xlabel="Selection frequency across resamples", ylabel="Number of markers")
    panel_label(ax, "b", x=-0.3, y=1.06)
    fig.tight_layout(w_pad=2)
    save(fig, figures / "fig14_marker_stability")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from the saved tables, no refitting
        res = load_config().path("results")
        plot(
            pd.read_csv(res / "tables" / "marker_stability.csv"),
            pd.read_csv(res / "tables" / "marker_stability_per_gene.csv"),
            res / "figures",
        )
    else:
        main()
