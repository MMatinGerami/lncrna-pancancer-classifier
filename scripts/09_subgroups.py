"""Subgroup performance: where does the classifier work less well?

A single macro-F1 hides differences between patient groups. This script takes the saved
test-set probabilities of the logistic-regression models and reports accuracy, top-3
accuracy and calibration error with bootstrap confidence intervals for:

  - patient sex, age group and AJCC stage (from the TCGA clinical table);
  - cancer-type rarity (training-set size of the true class).

It also relates per-class F1 to the number of training tumours per class.

Writes results/tables/subgroup_performance.csv, per_class_size_vs_f1.csv and
figures/fig9_subgroups.png.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, top_k_accuracy_score

from lncpan.config import load_config
from lncpan.evaluate import expected_calibration_error
from lncpan.plotting import (
    DOUBLE,
    INK_2,
    INK_3,
    MM,
    UNIVERSE_COLORS,
    UNIVERSE_LABELS,
    apply_style,
    panel_label,
    save,
)

N_BOOT = 1000
AGE_BINS = [0, 50, 65, 200]
AGE_LABELS = ["<50", "50 to 64", "65+"]
RARITY_BINS = [0, 100, 300, 10_000]
RARITY_LABELS = ["<100", "100 to 299", "300+"]


def read_clinical(path) -> pd.DataFrame:
    cols = [
        "_PATIENT",
        "gender",
        "age_at_initial_pathologic_diagnosis",
        "ajcc_pathologic_tumor_stage",
    ]
    clin = (
        pd.read_csv(path, sep="\t", usecols=cols).drop_duplicates("_PATIENT").set_index("_PATIENT")
    )
    clin["sex"] = clin["gender"].str.title()
    clin["age_group"] = pd.cut(
        clin["age_at_initial_pathologic_diagnosis"], AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(object)
    stage = parse_stage(clin["ajcc_pathologic_tumor_stage"])
    clin["stage"] = stage.fillna("Not reported")
    return clin[["sex", "age_group", "stage"]]


def parse_stage(raw: pd.Series) -> pd.Series:
    """'Stage IIIA' -> 'III'; sub-stages collapse to their main stage, anything else is NaN."""
    return raw.astype(str).str.extract(r"^Stage (IV|III|II|I)[ABC]?\d?$")[0]


def group_metrics(y: np.ndarray, proba: np.ndarray, n_classes: int, seed: int) -> dict[str, float]:
    labels = np.arange(n_classes)
    fns = {
        "accuracy": lambda y, p: float((p.argmax(1) == y).mean()),
        "top3_accuracy": lambda y, p: top_k_accuracy_score(y, p, k=3, labels=labels),
        "ece": expected_calibration_error,
    }
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in fns.items():
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            boots.append(f(y[idx], proba[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out |= {name: f(y, proba), f"{name}_low": lo, f"{name}_high": hi}
    return out


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures, preds = res / "tables", res / "figures", res / "predictions"
    samples = pd.read_parquet(cfg.path("processed") / "samples.parquet").set_index("sample")
    clin = read_clinical(cfg.path("raw") / cfg["data"]["clinical"])
    cohort = pd.read_csv(tables / "cohort.csv").set_index("cancer_type")["train"]

    rows, size_rows = [], []
    for universe in ("lncRNA", "protein_coding"):
        pred = pd.read_parquet(preds / f"test__{universe}__logreg.parquet")
        classes = np.array([c for c in pred.columns if c != "true"])
        proba = pred[classes].to_numpy()
        y = np.searchsorted(classes, pred["true"].to_numpy())
        meta = samples.loc[pred.index, ["patient"]].join(clin, on="patient")
        meta["rarity"] = pd.cut(
            cohort.loc[pred["true"]].to_numpy(), RARITY_BINS, labels=RARITY_LABELS, right=False
        ).astype(object)
        meta["sex"] = meta["sex"].fillna("Not reported")
        meta["age_group"] = meta["age_group"].fillna("Not reported")

        for factor in ("sex", "age_group", "stage", "rarity"):
            for level, m in meta.groupby(factor, observed=True).indices.items():
                if len(m) < 30:
                    continue
                rows.append(
                    {"universe": universe, "factor": factor, "level": level, "n": len(m)}
                    | group_metrics(y[m], proba[m], len(classes), cfg.seed)
                )
        f1 = f1_score(y, proba.argmax(1), average=None, labels=np.arange(len(classes)))
        size_rows.append(
            pd.DataFrame(
                {
                    "universe": universe,
                    "cancer_type": classes,
                    "n_train": cohort.loc[classes].to_numpy(),
                    "f1": f1,
                }
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "subgroup_performance.csv", index=False)
    sizes = pd.concat(size_rows, ignore_index=True)
    sizes.to_csv(tables / "per_class_size_vs_f1.csv", index=False)
    print(out[out.universe == "lncRNA"].round(3).to_string(index=False))

    plot(out, sizes, figures)


LEVEL_ORDER = {"rarity": RARITY_LABELS, "age_group": AGE_LABELS, "stage": ["I", "II", "III", "IV"]}
FACTOR_NAMES = {
    "rarity": "Training tumours in class",
    "stage": "AJCC stage",
    "age_group": "Age (years)",
    "sex": "Sex",
}


def plot(out: pd.DataFrame, sizes: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE * 0.85, 68 * MM), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    ax = axes[0]
    d = out[out.universe == "lncRNA"]
    d = d[d.level != "Not reported"]
    labels, ticks, y_pos = [], [], 0.0
    for factor in ("rarity", "stage", "age_group", "sex"):
        ax.text(
            -0.02,
            y_pos,
            FACTOR_NAMES[factor],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6,
            fontweight="bold",
        )
        y_pos += 1
        rows = d[d.factor == factor]
        order = LEVEL_ORDER.get(factor)
        if order:
            rows = rows.set_index("level").loc[[lv for lv in order if lv in set(rows.level)]]
            rows = rows.reset_index()
        for _, r in rows.iterrows():
            ax.errorbar(
                r["accuracy"],
                y_pos,
                xerr=[[r["accuracy"] - r["accuracy_low"]], [r["accuracy_high"] - r["accuracy"]]],
                fmt="o",
                color=UNIVERSE_COLORS["lncRNA"],
                elinewidth=0.8,
            )
            labels.append(f"{r['level']} (n = {r['n']})")
            ticks.append(y_pos)
            y_pos += 1
        y_pos += 0.4
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y_pos - 0.6, -0.8)
    ax.set_xlabel("Accuracy (test set)")
    panel_label(ax, "a", x=-0.62, y=1.02)

    ax = axes[1]
    for universe, sub in sizes.groupby("universe"):
        ax.scatter(
            sub["n_train"],
            sub["f1"],
            s=9,
            color=UNIVERSE_COLORS[universe],
            label=UNIVERSE_LABELS[universe],
            edgecolor="white",
            lw=0.3,
        )
    lnc = sizes[sizes.universe == "lncRNA"]
    for _, r in lnc[lnc.f1 < 0.8].iterrows():
        ax.annotate(
            r["cancer_type"],
            (r["n_train"], r["f1"]),
            fontsize=5,
            color=INK_2,
            ha="right",
            xytext=(-3, 2),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ticks_n = [30, 100, 300, 1000]
    ax.set_xticks(ticks_n, [f"{t:,}" for t in ticks_n])
    ax.minorticks_off()
    ax.set(xlabel="Training tumours per class", ylabel="Per-class F1 (test set)")
    ax.axvline(100, color=INK_3, lw=0.6, ls=":")
    ax.legend(loc="lower right")
    panel_label(ax, "b", x=-0.22, y=1.02)
    fig.tight_layout(w_pad=2)
    save(fig, figures / "fig9_subgroups")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from the saved tables, no refitting
        res = load_config().path("results")
        plot(
            pd.read_csv(res / "tables" / "subgroup_performance.csv"),
            pd.read_csv(res / "tables" / "per_class_size_vs_f1.csv"),
            res / "figures",
        )
    else:
        main()
