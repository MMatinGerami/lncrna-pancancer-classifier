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
out of the calibration for that score, so its guarantee rests on the remaining 70%. APS and
RAPS use the randomised score (one uniform draw per tumour, fixed seed); the deterministic
form gives an empty set to exactly the tumours the model is most certain about.

Writes results/tables/conformal_{summary,per_class}.csv and figures/fig8_conformal.png.
"""

from __future__ import annotations

import json
import sys

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
    uniform_draws,
)
from lncpan.io import load_dataset
from lncpan.models import build_model
from lncpan.plotting import (
    DOUBLE,
    INK_2,
    INK_3,
    MM,
    UNIVERSE_COLORS,
    apply_style,
    panel_label,
    save,
)

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

    u_cal, u_te, u_ex = (
        uniform_draws(len(y), cfg.seed + i) for i, y in enumerate((y_cal, y_te, y_ex))
    )

    rows, per_class, raps_constants = [], [], {}
    for score in SCORES:
        for alpha in ALPHAS:
            if score == "raps":
                k_reg, lam = tune_raps(p_cal[tune_idx], y_cal[tune_idx], alpha, u=u_cal[tune_idx])
                raps_constants[alpha] = {"k_reg": k_reg, "lam": lam}
                kw = {"k_reg": k_reg, "lam": lam}
                p_c, y_c, u_c = p_cal[keep_idx], y_cal[keep_idx], u_cal[keep_idx]
            else:
                kw = {}
                p_c, y_c, u_c = p_cal, y_cal, u_cal
            s_cal = nonconformity(p_c, y_c, score, u=u_c, **kw)
            thresholds = {
                "marginal": quantile(s_cal, alpha),
                "class_conditional": class_conditional_quantiles(s_cal, y_c, alpha, n_classes),
            }
            for variant, q in thresholds.items():
                for split, p, y, u in (("test", p_te, y_te, u_te), ("external", p_ex, y_ex, u_ex)):
                    sets = prediction_sets(p, q, score, u=u, **kw)
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
        "raps": {
            str(a): {
                "k_reg": c["k_reg"],
                "lam": c["lam"],
                "marginal": quantile(
                    nonconformity(p_cal[keep_idx], y_cal[keep_idx], "raps", u=u_cal[keep_idx], **c),
                    a,
                ),
                "class_conditional": class_conditional_quantiles(
                    nonconformity(p_cal[keep_idx], y_cal[keep_idx], "raps", u=u_cal[keep_idx], **c),
                    y_cal[keep_idx],
                    a,
                    n_classes,
                ).tolist(),
            }
            for a, c in raps_constants.items()
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
    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE, 62 * MM), gridspec_kw={"width_ratios": [3, 1.5, 1.5]}
    )
    variants = (
        ("marginal", "Marginal", SERIES[0]),
        ("class_conditional", "Class-conditional", SERIES[1]),
    )

    ax = axes[0]
    pc = per_class[(per_class.score == "lac") & (per_class.alpha == 0.10)]
    order = pc[pc.variant == "marginal"].sort_values("n")["cancer_type"].tolist()
    x = np.arange(len(order))
    for (variant, label, colour), dx in zip(variants, (-0.2, 0.2), strict=True):
        d = pc[pc.variant == variant].set_index("cancer_type").loc[order]
        ax.bar(x + dx, d["coverage"], width=0.4, color=colour, label=label, lw=0)
    ax.axhline(0.9, color=INK_3, lw=0.6, ls="--")
    ns = pc[pc.variant == "marginal"].set_index("cancer_type").loc[order]["n"]
    ax.set_xticks(
        x, [f"{c} ({n})" for c, n in zip(order, ns, strict=True)], rotation=90, fontsize=4.5
    )
    ax.tick_params(axis="x", length=1.5)
    ax.set(xlim=(-0.7, len(order) - 0.3), ylim=(0, 1.0), ylabel="Coverage (test set)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    panel_label(ax, "a", x=-0.07, y=1.08)

    ax = axes[1]
    s = summary[(summary.split == "test") & (summary.score == "lac")]
    for variant, label, colour in variants:
        d = s[s.variant == variant].sort_values("alpha")
        ax.plot(1 - d["alpha"], d["mean_size"], "o-", color=colour, label=label)
    ex = summary[(summary.split == "external") & (summary.score == "lac") & (summary.alpha == 0.10)]
    for _, r in ex.iterrows():
        ax.scatter(
            0.9,
            r["mean_size"],
            marker="s",
            s=14,
            facecolor="white",
            edgecolor=SERIES[0] if r.variant == "marginal" else SERIES[1],
            lw=0.8,
            zorder=4,
        )
    ax.annotate(
        "External\nmetastases",
        (0.9, ex["mean_size"].max()),
        xytext=(0.915, ex["mean_size"].max() - 2.6),
        fontsize=5,
        color=INK_2,
        arrowprops={"arrowstyle": "-", "lw": 0.4, "color": INK_3},
    )
    ax.set(
        xticks=[0.9, 0.95],
        xlim=(0.89, 0.96),
        xlabel="Nominal coverage (1 − α)",
        ylabel="Mean set size (cancer types)",
    )
    panel_label(ax, "b", x=-0.3, y=1.08)

    ax = axes[2]
    s = summary[(summary.split == "test") & (summary.alpha == 0.10)]
    scores = ["lac", "aps", "raps"]
    x = np.arange(len(scores))
    for (variant, label, colour), dx in zip(variants, (-0.2, 0.2), strict=True):
        d = s[s.variant == variant].set_index("score").loc[scores]
        ax.bar(x + dx, d["mean_size"], width=0.4, color=colour, label=label, lw=0)
        for xi, (size, cov) in enumerate(zip(d["mean_size"], d["coverage"], strict=True)):
            ax.text(xi + dx, size + 0.25, f"{cov:.2f}", ha="center", fontsize=4.5, color=INK_2)
    ax.set(
        xticks=x,
        xticklabels=[sc.upper() for sc in scores],
        ylabel="Mean set size (test set)",
        ylim=(0, 12),
    )
    panel_label(ax, "c", x=-0.3, y=1.08)
    fig.tight_layout(w_pad=1.5)
    save(fig, figures / "fig8_conformal")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from the saved tables, no refitting
        tab = load_config().path("results") / "tables"
        plot(
            pd.read_csv(tab / "conformal_summary.csv"),
            pd.read_csv(tab / "conformal_per_class.csv"),
            load_config().path("results") / "figures",
        )
    else:
        main()
