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
    BLUE,
    DIVERGING,
    DOUBLE,
    INK,
    INK_2,
    INK_3,
    MM,
    MODEL_LABELS,
    ONE_HALF,
    SEQUENTIAL,
    SINGLE,
    UNIVERSE_COLORS,
    UNIVERSE_LABELS,
    apply_style,
    panel_label,
    save,
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


def tsne_embedding() -> pd.DataFrame:
    """2-D t-SNE of the training tumours, cached so figures can be redrawn without refitting."""
    path = TAB / "tsne_lncrna.csv"
    if path.exists():
        return pd.read_csv(path)
    ds = load_dataset(cfg, "lncRNA")
    X, y = ds.split("train")
    Z = TopVarianceSelector(2000).fit_transform(X.to_numpy())
    Z = PCA(50, random_state=cfg.seed).fit_transform((Z - Z.mean(0)) / (Z.std(0) + 1e-6))
    emb = TSNE(2, perplexity=40, init="pca", random_state=cfg.seed).fit_transform(Z)
    out = pd.DataFrame(
        {"sample": X.index, "tsne_1": emb[:, 0], "tsne_2": emb[:, 1], "cancer_type": ds.classes[y]}
    )
    out.to_csv(path, index=False)
    return out


def _spread_labels(xy: np.ndarray, box: tuple[float, float], n_iter: int = 500) -> np.ndarray:
    """Move label centres apart until their boxes (width, height in data units) do not overlap."""
    xy = xy.astype(float).copy()
    w, h = box
    for _ in range(n_iter):
        moved = False
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                dx, dy = xy[j] - xy[i]
                ox, oy = w - abs(dx), h - abs(dy)
                if ox > 0 and oy > 0:  # overlap: separate along the axis that needs less movement
                    if ox / w < oy / h:
                        step = np.array([np.sign(dx) or 1.0, 0.0]) * ox / 2
                    else:
                        step = np.array([0.0, np.sign(dy) or 1.0]) * oy / 2
                    xy[i] -= step
                    xy[j] += step
                    moved = True
        if not moved:
            break
    return xy


def fig_tsne() -> None:
    emb = tsne_embedding()
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE))
    ax.scatter(emb.tsne_1, emb.tsne_2, s=0.6, c=BLUE, alpha=0.35, linewidths=0, rasterized=True)
    centres = emb.groupby("cancer_type")[["tsne_1", "tsne_2"]].median()
    span = np.ptp(emb[["tsne_1", "tsne_2"]].to_numpy(), axis=0).max()
    pos = _spread_labels(centres.to_numpy(), box=(0.085 * span, 0.032 * span))
    for name, (x, y) in zip(centres.index, pos, strict=True):
        ax.text(x, y, name, fontsize=5, ha="center", va="center", color=INK)
    ax.set(xticks=[], yticks=[], xlabel="t-SNE 1", ylabel="t-SNE 2")
    save(fig, FIG / "fig1_tsne_lncrna")


def fig_benchmark(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.75, 52 * MM))
    for ax, letter, metric, label in zip(
        axes, "ab", ["macro_f1", "top3_accuracy"], ["Macro-F1", "Top-3 accuracy"], strict=True
    ):
        m = metrics[metrics.metric == metric]
        x = np.arange(len(MODELS))
        for i, u in enumerate(["lncRNA", "protein_coding"]):
            d = m[m.universe == u].set_index("model").loc[MODELS]
            ax.errorbar(
                x + (i - 0.5) * 0.24,
                d.value,
                yerr=[d.value - d.ci_low, d.ci_high - d.value],
                fmt="o",
                color=UNIVERSE_COLORS[u],
                label=UNIVERSE_LABELS[u],
                elinewidth=0.8,
            )
        ax.set_xticks(x, [MODEL_LABELS[k] for k in MODELS])
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_ylabel(f"{label} (test set)")
        panel_label(ax, letter, x=-0.2)
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=2.5)
    save(fig, FIG / "fig2_benchmark")


def fig_confusion(model: str) -> None:
    p = pd.read_parquet(PRED / f"test__lncRNA__{model}.parquet")
    classes = p.columns[:-1].to_numpy()
    pred = classes[p[classes].to_numpy().argmax(1)]
    cm = confusion_matrix(p["true"], pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(ONE_HALF, ONE_HALF * 0.92))
    im = ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=5)
    ax.set_yticks(range(len(classes)), classes, fontsize=5)
    ax.tick_params(length=1.5, width=0.4)
    ax.set(xlabel="Predicted cancer type", ylabel="True cancer type")
    # every off-diagonal cell holding at least 5% of a class is labelled (rule stated in caption)
    for i, j in zip(*np.where((cm >= 0.05) & ~np.eye(len(classes), dtype=bool)), strict=True):
        colour = "white" if cm[i, j] > 0.5 else INK
        ax.text(
            j,
            i,
            f"{cm[i, j]:.2f}".lstrip("0"),
            ha="center",
            va="center",
            fontsize=3.8,
            color=colour,
        )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Fraction of true class")
    cb.outline.set_linewidth(0.4)
    save(fig, FIG / "fig3_confusion_lncrna")


def fig_per_class(metrics: pd.DataFrame) -> None:
    f1 = {}
    for u in ("lncRNA", "protein_coding"):
        p = pd.read_parquet(PRED / f"test__{u}__{best_model(u, metrics)}.parquet")
        classes = p.columns[:-1].to_numpy()
        pred = classes[p[classes].to_numpy().argmax(1)]
        f1[u] = pd.Series(f1_score(p["true"], pred, labels=classes, average=None), index=classes)
    df = pd.DataFrame(f1)
    df.to_csv(TAB / "per_class_f1.csv")
    fig, ax = plt.subplots(figsize=(SINGLE * 0.8, SINGLE * 0.8))
    lo = max(0, df.min().min() - 0.05)
    ax.plot([lo, 1.01], [lo, 1.01], color=INK_3, lw=0.6, ls="--", zorder=1)
    ax.scatter(df.protein_coding, df.lncRNA, s=10, color=BLUE, edgecolor="white", lw=0.4, zorder=3)
    labelled = df[(df.min(axis=1) < 0.9) | ((df.lncRNA - df.protein_coding).abs() > 0.05)]
    pts = labelled[["protein_coding", "lncRNA"]].to_numpy()
    # labels start to the right of their point and are pushed apart; a thin leader line
    # connects any label that had to move
    order = np.argsort(pts[:, 0])
    offsets = np.zeros_like(pts)
    offsets[order] = [(0.025, 0.02 if i % 2 == 0 else -0.02) for i in range(len(order))]
    offsets[pts[:, 1] > 0.95] = (-0.09, 0.0)  # near the dense top-right corner: label leftwards
    pos = _spread_labels(pts + offsets, box=(0.09, 0.026))
    for name, pt, xy in zip(labelled.index, pts, pos, strict=True):
        ax.annotate(
            name,
            pt,
            xytext=xy,
            textcoords="data",
            fontsize=5,
            color=INK_2,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.3, "color": INK_3, "shrinkA": 0, "shrinkB": 1.5},
        )
    ax.set(
        xlim=(lo, 1.01),
        ylim=(lo, 1.01),
        xlabel="F1, protein-coding model",
        ylabel="F1, lncRNA model",
        aspect="equal",
    )
    save(fig, FIG / "fig4_per_class_f1")


def fig_budget() -> None:
    b = pd.read_csv(TAB / "feature_budget.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 55 * MM))
    for u, d in b.groupby("universe"):
        s = d.groupby("k").macro_f1.agg(["mean", "std"])
        ax.plot(s.index, s["mean"], marker="o", color=UNIVERSE_COLORS[u], label=UNIVERSE_LABELS[u])
        ax.fill_between(
            s.index,
            s["mean"] - s["std"],
            s["mean"] + s["std"],
            color=UNIVERSE_COLORS[u],
            alpha=0.2,
            lw=0,
        )
    ax.set_xscale("log")
    ks = sorted(b["k"].unique())
    ax.set_xticks(ks, [f"{k:,}" for k in ks])
    ax.minorticks_off()
    ax.set(xlabel="Number of most variable genes", ylabel="Macro-F1 (5-fold CV)")
    ax.legend(loc="lower right")
    save(fig, FIG / "fig5_feature_budget")


def fig_calibration(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE * 0.8, SINGLE * 0.8))
    ax.plot([0, 1], [0, 1], color=INK_3, lw=0.6, ls="--")
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
            color=UNIVERSE_COLORS[u],
            label=f"{UNIVERSE_LABELS[u]} (ECE {ece:.3f})",
        )
    ax.set(
        xlabel="Predicted confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
        aspect="equal",
    )
    ax.legend(loc="upper left")
    save(fig, FIG / "fig6_calibration")


def fig_markers() -> None:
    mk = pd.read_csv(TAB / "shap_markers_per_cancer.csv").query("rank <= 2 and mean_shap > 0")
    ds = load_dataset(cfg, "lncRNA")
    X, y = ds.split("train")
    genes = list(dict.fromkeys(mk.gene_id))
    means = pd.DataFrame(X[genes].to_numpy(), columns=genes).groupby(ds.classes[y]).mean()
    z = (means - means.mean()) / means.std()
    names = mk.drop_duplicates("gene_id").set_index("gene_id").loc[genes, "gene_name"]
    fig, ax = plt.subplots(figsize=(DOUBLE, 100 * MM))
    im = ax.imshow(z.to_numpy(), cmap=DIVERGING, vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(range(len(z)), z.index, fontsize=5)
    ax.set_xticks(range(len(genes)), names, rotation=90, fontsize=4.5)
    ax.tick_params(length=1.5, width=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("Mean expression (z-score across cancer types)")
    cb.outline.set_linewidth(0.4)
    save(fig, FIG / "fig7_shap_markers")


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
    print(sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
