"""All figures for the README, from the saved tables/predictions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score

from lncpan.config import load_config
from lncpan.evaluate import paired_bootstrap_difference
from lncpan.features import TopVarianceSelector
from lncpan.io import load_dataset
from lncpan.plotting import (
    DIVERGING,
    INK_2,
    INK_3,
    MODEL_LABELS,
    SEQUENTIAL,
    UNIVERSE_COLORS,
    UNIVERSE_LABELS,
    apply_style,
)

cfg = load_config()
RES = cfg.path("results")
TAB, FIG, PRED = RES / "tables", RES / "figures", RES / "predictions"
MODELS = [m for m in MODEL_LABELS if (PRED / f"test__lncRNA__{m}.parquet").exists()]


def load_metrics() -> pd.DataFrame:
    return pd.concat(pd.read_csv(p) for p in sorted(TAB.glob("metrics__*.csv")))


def best_model(universe: str, metrics: pd.DataFrame) -> str:
    m = metrics[(metrics.metric == "macro_f1") & (metrics.universe == universe)]
    return m.sort_values("value").iloc[-1]["model"]


def fig_tsne() -> None:
    ds = load_dataset(cfg, "lncRNA")
    X, y = ds.split("train")
    Z = TopVarianceSelector(2000).fit_transform(X.to_numpy())
    Z = PCA(50, random_state=cfg.seed).fit_transform((Z - Z.mean(0)) / (Z.std(0) + 1e-6))
    emb = TSNE(2, perplexity=40, init="pca", random_state=cfg.seed).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(emb[:, 0], emb[:, 1], s=3, c="#2a78d6", alpha=0.35, linewidths=0)
    for c, name in enumerate(ds.classes):
        cx, cy = np.median(emb[y == c], axis=0)
        ax.text(
            cx,
            cy,
            name,
            fontsize=7.5,
            weight="bold",
            ha="center",
            va="center",
            color="#0b0b0b",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        )
    ax.set(xticks=[], yticks=[], xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.grid(False)
    ax.set_title(
        f"lncRNA expression alone separates 33 cancer types (n = {len(y):,} training tumours)",
        loc="left",
    )
    fig.savefig(FIG / "fig1_tsne_lncrna.png")
    plt.close(fig)


def fig_benchmark(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, label in zip(
        axes, ["macro_f1", "top3_accuracy"], ["Macro-F1", "Top-3 accuracy"], strict=True
    ):
        m = metrics[metrics.metric == metric]
        x = np.arange(len(MODELS))
        for i, u in enumerate(["lncRNA", "protein_coding"]):
            d = m[m.universe == u].set_index("model").loc[MODELS]
            ax.errorbar(
                x + (i - 0.5) * 0.22,
                d.value,
                yerr=[d.value - d.ci_low, d.ci_high - d.value],
                fmt="o",
                ms=7,
                capsize=3,
                color=UNIVERSE_COLORS[u],
                label=UNIVERSE_LABELS[u],
                lw=1.5,
            )
        ax.set_xticks(x, [MODEL_LABELS[k] for k in MODELS])
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_ylabel(f"{label} (test set, 95% CI)")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(
        "lncRNAs match protein-coding genes for tissue-of-origin classification",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig2_benchmark.png")
    plt.close(fig)


def fig_confusion(model: str) -> None:
    p = pd.read_parquet(PRED / f"test__lncRNA__{model}.parquet")
    classes = p.columns[:-1].to_numpy()
    pred = classes[p[classes].to_numpy().argmax(1)]
    cm = confusion_matrix(p["true"], pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set(xlabel="Predicted", ylabel="True")
    ax.grid(False)
    for i, j in zip(*np.where((cm >= 0.08) & ~np.eye(len(classes), dtype=bool)), strict=True):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Fraction of true class")
    ax.set_title(
        f"Confusion matrix, lncRNA {MODEL_LABELS[model]} (test set, row-normalised)", loc="left"
    )
    fig.savefig(FIG / "fig3_confusion_lncrna.png")
    plt.close(fig)


def fig_per_class(metrics: pd.DataFrame) -> None:
    f1 = {}
    for u in ("lncRNA", "protein_coding"):
        p = pd.read_parquet(PRED / f"test__{u}__{best_model(u, metrics)}.parquet")
        classes = p.columns[:-1].to_numpy()
        pred = classes[p[classes].to_numpy().argmax(1)]
        f1[u] = pd.Series(f1_score(p["true"], pred, labels=classes, average=None), index=classes)
    df = pd.DataFrame(f1)
    df.to_csv(TAB / "per_class_f1.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls="--")
    ax.scatter(
        df.protein_coding, df.lncRNA, s=36, color="#2a78d6", edgecolor="white", lw=1, zorder=3
    )
    for name, r in df.iterrows():
        if r.min() < 0.9 or abs(r.lncRNA - r.protein_coding) > 0.05:
            ax.annotate(
                name,
                (r.protein_coding, r.lncRNA),
                xytext=(4, -3),
                textcoords="offset points",
                fontsize=7,
                color=INK_2,
            )
    lo = max(0, df.min().min() - 0.05)
    ax.set(
        xlim=(lo, 1.01),
        ylim=(lo, 1.01),
        xlabel="F1, protein-coding model",
        ylabel="F1, lncRNA model",
    )
    ax.set_title("Per-cancer F1 (test set)", loc="left")
    fig.savefig(FIG / "fig4_per_class_f1.png")
    plt.close(fig)


def fig_budget() -> None:
    b = pd.read_csv(TAB / "feature_budget.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for u, d in b.groupby("universe"):
        s = d.groupby("k").macro_f1.agg(["mean", "std"])
        ax.plot(
            s.index, s["mean"], marker="o", ms=5, color=UNIVERSE_COLORS[u], label=UNIVERSE_LABELS[u]
        )
        ax.fill_between(
            s.index,
            s["mean"] - s["std"],
            s["mean"] + s["std"],
            color=UNIVERSE_COLORS[u],
            alpha=0.15,
            lw=0,
        )
    ax.set_xscale("log")
    ax.set(
        xlabel="Number of genes (most variable, selected within each fold)",
        ylabel="Macro-F1 (5-fold CV, ±1 SD)",
    )
    ax.legend(loc="lower right")
    ax.set_title("A few hundred lncRNAs are enough", loc="left")
    fig.savefig(FIG / "fig5_feature_budget.png")
    plt.close(fig)


def fig_calibration(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls="--")
    bins = np.linspace(0, 1, 11)
    for u in ("lncRNA", "protein_coding"):
        model = best_model(u, metrics)
        p = pd.read_parquet(PRED / f"test__{u}__{model}.parquet")
        classes = p.columns[:-1].to_numpy()
        proba = p[classes].to_numpy()
        conf, correct = proba.max(1), classes[proba.argmax(1)] == p["true"].to_numpy()
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        xs = [conf[idx == i].mean() for i in range(10) if (idx == i).sum() >= 10]
        ys = [correct[idx == i].mean() for i in range(10) if (idx == i).sum() >= 10]
        ece = metrics.query("metric=='ece' and universe==@u and model==@model").value.iloc[0]
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=5,
            color=UNIVERSE_COLORS[u],
            label=f"{UNIVERSE_LABELS[u]}, {MODEL_LABELS[model]} (ECE {ece:.3f})",
        )
    ax.set(
        xlabel="Predicted confidence", ylabel="Observed accuracy", xlim=(0, 1.02), ylim=(0, 1.02)
    )
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Calibration (test set)", loc="left")
    fig.savefig(FIG / "fig6_calibration.png")
    plt.close(fig)


def fig_markers() -> None:
    mk = pd.read_csv(TAB / "shap_markers_per_cancer.csv").query("rank <= 2 and mean_shap > 0")
    ds = load_dataset(cfg, "lncRNA")
    X, y = ds.split("train")
    genes = list(dict.fromkeys(mk.gene_id))
    means = pd.DataFrame(X[genes].to_numpy(), columns=genes).groupby(ds.classes[y]).mean()
    z = (means - means.mean()) / means.std()
    names = mk.drop_duplicates("gene_id").set_index("gene_id").loc[genes, "gene_name"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    im = ax.imshow(z.to_numpy(), cmap=DIVERGING, vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(range(len(z)), z.index, fontsize=7)
    ax.set_xticks(range(len(genes)), names, rotation=90, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.02, label="Mean expression, z-score across cancer types")
    ax.set_title(
        "Top SHAP lncRNA markers per cancer type (2 per type), mean expression in training tumours",
        loc="left",
    )
    fig.savefig(FIG / "fig7_shap_markers.png")
    plt.close(fig)


def table_paired_differences() -> None:
    """lncRNA minus protein-coding macro-F1 per model, paired over the same test tumours."""
    rows = []
    for model in MODELS:
        a = pd.read_parquet(PRED / f"test__lncRNA__{model}.parquet")
        b = pd.read_parquet(PRED / f"test__protein_coding__{model}.parquet").loc[a.index]
        classes = a.columns[:-1].to_numpy()
        y = np.searchsorted(classes, a["true"].to_numpy())
        res = paired_bootstrap_difference(
            y, a[classes].to_numpy(), b[classes].to_numpy(), seed=cfg.seed
        )
        rows.append({"model": model, **res})
    pd.DataFrame(rows).to_csv(TAB / "paired_difference_lncrna_minus_pc.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main() -> None:
    apply_style()
    FIG.mkdir(exist_ok=True)
    metrics = load_metrics()
    metrics.to_csv(TAB / "metrics_all.csv", index=False)
    table_paired_differences()
    fig_benchmark(metrics)
    fig_confusion(best_model("lncRNA", metrics))
    fig_per_class(metrics)
    fig_calibration(metrics)
    if (TAB / "feature_budget.csv").exists():
        fig_budget()
    if (TAB / "shap_markers_per_cancer.csv").exists():
        fig_markers()
    fig_tsne()
    print(sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
