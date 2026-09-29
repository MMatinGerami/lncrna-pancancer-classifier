"""Conformal prediction sets: which cancer types can be ruled out with a stated error rate?

The benchmark reports top-1 and top-3 accuracy, which say nothing about any single tumour.
Split conformal prediction turns the classifier's probabilities into a set of candidate
cancer types per tumour with a guaranteed error rate. Steps:

  1. hold out 15% of the training split as a calibration set (stratified, never used to fit);
  2. refit the logistic-regression model on the rest with the tuned C;
  3. compute non-conformity scores on the calibration set and their (1 - alpha) quantile;
  4. build sets on the test split and on the external melanoma metastases.

Two variants are compared: one global threshold (marginal coverage) and one threshold per
class (class-conditional coverage), for the LAC, APS and RAPS scores at alpha = 0.10 and
0.05. The two RAPS constants are chosen on 30% of the calibration set, which is then left
out of the calibration for that score, so its guarantee rests on the remaining 70%.

Writes results/tables/conformal_{summary,per_class}.csv and figures/fig8_conformal.png.
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from lncpan.config import load_config
from lncpan.conformal import (
    SCORES,
    class_conditional_quantiles,
    nonconformity,
    per_class_coverage,
    prediction_sets,
    quantile,
    summarize,
    tune_raps,
)
from lncpan.io import load_dataset
from lncpan.models import build_model
from lncpan.plotting import INK_2, INK_3, UNIVERSE_COLORS, apply_style

SERIES = [UNIVERSE_COLORS["lncRNA"], UNIVERSE_COLORS["protein_coding"]]

ALPHAS = (0.10, 0.05)
CALIBRATION_FRACTION = 0.15
RAPS_TUNING_FRACTION = 0.30
UNIVERSE = "lncRNA"


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures = res / "tables", res / "figures"
    ds = load_dataset(cfg, UNIVERSE)
    X_tr, y_tr = ds.split("train")
    X_te, y_te = ds.split("test")
    X_ex, y_ex = ds.split("external")

    fit_idx, cal_idx = next(
        StratifiedShuffleSplit(1, test_size=CALIBRATION_FRACTION, random_state=cfg.seed).split(
            X_tr, y_tr
        )
    )
    best = json.loads((tables / f"search__{UNIVERSE}__logreg.json").read_text())["best_params"]
    model = build_model("logreg", cfg["features"]["top_k_variance"], cfg.seed)
    model.set_params(**best)
    model.fit(X_tr.iloc[fit_idx], y_tr[fit_idx])
    p_cal, p_te, p_ex = (model.predict_proba(X) for X in (X_tr.iloc[cal_idx], X_te, X_ex))
    y_cal = y_tr[cal_idx]
    n_classes = len(ds.classes)
    tune_idx, keep_idx = next(
        StratifiedShuffleSplit(1, test_size=1 - RAPS_TUNING_FRACTION, random_state=cfg.seed).split(
            p_cal, y_cal
        )
    )

    rows, per_class, raps_constants = [], [], {}
    for score in SCORES:
        for alpha in ALPHAS:
            if score == "raps":
                k_reg, lam = tune_raps(p_cal[tune_idx], y_cal[tune_idx], alpha)
                raps_constants[alpha] = {"k_reg": k_reg, "lam": lam}
                kw = {"k_reg": k_reg, "lam": lam}
                p_c, y_c = p_cal[keep_idx], y_cal[keep_idx]
            else:
                kw = {}
                p_c, y_c = p_cal, y_cal
            s_cal = nonconformity(p_c, y_c, score, **kw)
            thresholds = {
                "marginal": quantile(s_cal, alpha),
                "class_conditional": class_conditional_quantiles(s_cal, y_c, alpha, n_classes),
            }
            for variant, q in thresholds.items():
                for split, p, y in (("test", p_te, y_te), ("external", p_ex, y_ex)):
                    sets = prediction_sets(p, q, score, **kw)
                    rows.append(
                        {"score": score, "alpha": alpha, "variant": variant, "split": split}
                        | ({"n_calibration": len(y_c)} | kw if score == "raps" else {})
                        | summarize(sets, y)
                    )
                    if split == "test":
                        per_class.append(
                            per_class_coverage(sets, y, ds.classes).assign(
                                score=score, alpha=alpha, variant=variant
                            )
                        )
    # the refit model and its calibration thresholds, for `lncpan predict`
    joblib.dump(model, res / "models" / f"{UNIVERSE}__logreg_conformal.joblib")
    thresholds = {
        "universe": UNIVERSE,
        "classes": ds.classes.tolist(),
        "calibration_samples": int(len(y_cal)),
        "lac": {
            str(a): {
                "marginal": quantile(nonconformity(p_cal, y_cal, "lac"), a),
                "class_conditional": class_conditional_quantiles(
                    nonconformity(p_cal, y_cal, "lac"), y_cal, a, n_classes
                ).tolist(),
            }
            for a in ALPHAS
        },
    }
    (res / "models" / f"{UNIVERSE}__conformal_thresholds.json").write_text(
        json.dumps(thresholds, indent=1)
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "conformal_summary.csv", index=False)
    for alpha, c in raps_constants.items():
        print(f"RAPS at alpha={alpha}: k_reg={c['k_reg']}, lam={c['lam']}")
    per_class = pd.concat(per_class, ignore_index=True)
    per_class.to_csv(tables / "conformal_per_class.csv", index=False)
    print(summary.round(3).to_string(index=False))

    plot(summary, per_class, figures)


def plot(summary: pd.DataFrame, per_class: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [3, 2, 2]})

    ax = axes[0]
    pc = per_class[(per_class.score == "lac") & (per_class.alpha == 0.10)]
    order = pc[pc.variant == "marginal"].sort_values("n")["cancer_type"].tolist()
    x = np.arange(len(order))
    for variant, colour, dx in (
        ("marginal", SERIES[0], -0.18),
        ("class_conditional", SERIES[1], 0.18),
    ):
        d = pc[pc.variant == variant].set_index("cancer_type").loc[order]
        ax.bar(x + dx, d["coverage"], width=0.36, color=colour, label=variant.replace("_", " "))
    ax.axhline(0.9, color=INK_3, lw=1, ls="--")
    ax.set(
        xticks=x,
        ylim=(0.5, 1.02),
        ylabel="Coverage on test tumours",
        title="Per-class coverage at α = 0.10 (LAC score)",
    )
    ns = pc[pc.variant == "marginal"].set_index("cancer_type").loc[order]["n"]
    ax.set_xticklabels(
        [f"{c} ({n})" for c, n in zip(order, ns, strict=True)], fontsize=6, rotation=90
    )
    ax.set_xlabel("Cancer type and number of test tumours (sorted by size)", color=INK_2)
    ax.legend(loc="lower right")

    ax = axes[1]
    s = summary[(summary.split == "test") & (summary.score == "lac")]
    for variant, colour in (("marginal", SERIES[0]), ("class_conditional", SERIES[1])):
        d = s[s.variant == variant].sort_values("alpha")
        ax.plot(1 - d["alpha"], d["mean_size"], "o-", color=colour, label=variant.replace("_", " "))
    ex = summary[(summary.split == "external") & (summary.score == "lac") & (summary.alpha == 0.10)]
    for _, r in ex.iterrows():
        ax.scatter(
            0.9,
            r["mean_size"],
            marker="s",
            s=40,
            color=SERIES[0] if r.variant == "marginal" else SERIES[1],
            zorder=3,
        )
    ax.set(
        xlabel="Nominal coverage (1 - α)",
        ylabel="Mean set size (cancer types)",
        title="Set size vs guarantee",
    )
    ax.text(
        0.905, ex["mean_size"].max(), "metastases (external)", fontsize=7, color=INK_2, va="bottom"
    )
    ax.legend(loc="upper left")

    ax = axes[2]
    s = summary[(summary.split == "test") & (summary.alpha == 0.10)]
    scores = ["lac", "aps", "raps"]
    x = np.arange(len(scores))
    for variant, colour, dx in (
        ("marginal", SERIES[0], -0.18),
        ("class_conditional", SERIES[1], 0.18),
    ):
        d = s[s.variant == variant].set_index("score").loc[scores]
        ax.bar(x + dx, d["mean_size"], width=0.36, color=colour, label=variant.replace("_", " "))
        for xi, (size, cov) in enumerate(zip(d["mean_size"], d["coverage"], strict=True)):
            ax.text(xi + dx, size + 0.3, f"{cov:.2f}", ha="center", fontsize=6.5, color=INK_2)
    ax.set(
        xticks=x,
        xticklabels=[sc.upper() for sc in scores],
        ylabel="Mean set size on test tumours",
        title="Score comparison at α = 0.10 (coverage above bars)",
        ylim=(0, 18),
    )
    ax.legend(loc="upper right")
    fig.savefig(figures / "fig8_conformal.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
