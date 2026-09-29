"""Calibration under shift and results by biopsy site on MET500.

The protein-coding model recovers the primary site of two thirds of MET500 patients, but a
classifier used on tumours of unknown origin also needs its confidence to mean something
on the new cohort. This script asks two questions of that model.

1. Are its probabilities still calibrated after the shift, and does the usual fix help?
   The model is refit on 85% of the TCGA training split; the other 15% is a calibration
   set on which a single temperature is fitted (temperature scaling). Expected calibration
   error is then measured on the TCGA test split and on MET500, before and after scaling,
   with reliability diagrams. Temperature scaling is fitted on TCGA only, as it would be
   in practice, so it cannot know about the shift.

2. Does the biopsy site matter? MET500 records where each metastasis was sampled (liver,
   lymph node, soft tissue, lung, bone marrow, ...). Top-1 and top-3 accuracy are reported
   per biopsy site with at least 20 patients, together with the fraction of patients whose
   prediction is the biopsy organ itself, which is the failure mode a tissue-of-origin
   classifier should be checked for.

Writes results/tables/met500_calibration.csv and met500_by_biopsy_site.csv and
figures/fig13_met500_calibration.png.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from lncpan.calibration import apply_temperature, fit_temperature, reliability_curve
from lncpan.config import load_config
from lncpan.io import load_dataset
from lncpan.models import build_model
from lncpan.plotting import INK_2, INK_3, UNIVERSE_COLORS, apply_style

UNIVERSE = "protein_coding"
CALIBRATION_FRACTION = 0.15
MIN_SITE_N = 20
N_BOOT = 1000
# TCGA class whose tissue is the biopsy organ; a prediction of it on a metastasis sampled
# there is most likely the host tissue, not the primary
BIOPSY_ORGAN_CLASS = {
    "liver": {"LIHC", "CHOL"},
    "lung": {"LUAD", "LUSC"},
    "skin": {"SKCM"},
    "brain": {"GBM", "LGG"},
    "lymph_node": {"DLBC"},
    "bone_marrow": {"LAML"},
}


def met500_module():
    spec = importlib.util.spec_from_file_location("met500", Path("scripts/12_met500.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bootstrap_mean(x: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return float(x.mean()), *np.percentile(boots, [2.5, 97.5])


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures = res / "tables", res / "figures"
    m12 = met500_module()
    ds = load_dataset(cfg, UNIVERSE)
    X_tr, y_tr = ds.split("train")
    X_te, y_te = ds.split("test")
    fit_idx, cal_idx = next(
        StratifiedShuffleSplit(1, test_size=CALIBRATION_FRACTION, random_state=cfg.seed).split(
            X_tr, y_tr
        )
    )
    best = json.loads((tables / f"search__{UNIVERSE}__logreg.json").read_text())["best_params"]
    model = build_model("logreg", cfg["features"]["top_k_variance"], cfg.seed)
    model.set_params(**best)
    model.fit(X_tr.iloc[fit_idx], y_tr[fit_idx])
    p_cal, p_te = model.predict_proba(X_tr.iloc[cal_idx]), model.predict_proba(X_te)
    t = fit_temperature(p_cal, y_tr[cal_idx])
    print(f"temperature fitted on {len(cal_idx)} TCGA calibration tumours: T = {t:.2f}")

    genes = pd.read_parquet(cfg.path("processed") / "genes.parquet")
    X_met, meta = m12.load_met500(cfg.path("raw"))
    X_met, _ = m12.align(X_met, model, genes)
    p_met = model.predict_proba(X_met)
    allowed = [m12.COHORT_TO_TCGA[c] for c in meta["cohort"]]
    top1_met, top3_met = m12.correct_flags(p_met, ds.classes, allowed)
    # probability mass on the admissible TCGA classes, for the log-loss on MET500
    allowed_idx = [np.flatnonzero(np.isin(ds.classes, sorted(a))) for a in allowed]

    rows, curves = [], {}
    for split, p, correct_fn in (
        ("TCGA test", p_te, lambda q: q.argmax(1) == y_te),
        ("MET500", p_met, lambda q: top1_met),
    ):
        for stage, q in (("raw", p), ("temperature scaled", apply_temperature(p, t))):
            correct, conf = correct_fn(q), q.max(1)
            if split == "TCGA test":
                nll = -np.log(np.clip(q[np.arange(len(y_te)), y_te], 1e-12, None)).mean()
            else:
                nll = -np.log(
                    np.clip([q[i, ix].sum() for i, ix in enumerate(allowed_idx)], 1e-12, None)
                ).mean()
            rows.append(
                {
                    "split": split,
                    "stage": stage,
                    "n": len(q),
                    "temperature": t if stage != "raw" else 1.0,
                    "accuracy": float(correct.mean()),
                    "mean_confidence": float(conf.mean()),
                    "ece": ece_from_flags(correct, conf),
                    "log_loss": float(nll),
                }
            )
            curves[(split, stage)] = reliability_curve(correct, conf)
    calib = pd.DataFrame(rows)
    calib.to_csv(tables / "met500_calibration.csv", index=False)
    print(calib.round(3).to_string(index=False))

    site_rows = []
    pred = ds.classes[p_met.argmax(1)]
    for site, idx in meta.groupby("biopsy_tissue").indices.items():
        if len(idx) < MIN_SITE_N:
            continue
        a, lo, hi = bootstrap_mean(top1_met[idx], cfg.seed)
        organ = BIOPSY_ORGAN_CLASS.get(site, set())
        site_rows.append(
            {
                "biopsy_site": site,
                "n": len(idx),
                "accuracy": a,
                "accuracy_low": lo,
                "accuracy_high": hi,
                "top3": float(top3_met[idx].mean()),
                "predicted_biopsy_organ": float(np.isin(pred[idx], sorted(organ)).mean())
                if organ
                else np.nan,
                "mean_confidence": float(p_met[idx].max(1).mean()),
            }
        )
    sites = pd.DataFrame(site_rows).sort_values("n", ascending=False)
    sites.to_csv(tables / "met500_by_biopsy_site.csv", index=False)
    print(sites.round(3).to_string(index=False))
    plot(curves, calib, sites, figures)


def ece_from_flags(correct: np.ndarray, conf: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE from per-sample correctness flags (same bins as lncpan.evaluate)."""
    acc, mean_conf, n = reliability_curve(correct, conf, n_bins)
    m = n > 0
    return float((n[m] / n.sum() * np.abs(acc[m] - mean_conf[m])).sum())


def plot(curves, calib: pd.DataFrame, sites: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    colour = UNIVERSE_COLORS[UNIVERSE]
    for ax, split in zip(axes[:2], ("TCGA test", "MET500"), strict=True):
        ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls="--")
        for stage, style in (("raw", "o-"), ("temperature scaled", "s--")):
            acc, conf, n = curves[(split, stage)]
            m = n >= 5
            e = calib[(calib.split == split) & (calib.stage == stage)]["ece"].item()
            ax.plot(
                conf[m],
                acc[m],
                style,
                color=colour,
                alpha=1.0 if stage == "raw" else 0.55,
                label=f"{stage} (ECE {e:.3f})",
            )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy", title=split)
        ax.legend(loc="upper left", fontsize=7)
    ax = axes[2]
    x = np.arange(len(sites))
    ax.bar(x - 0.2, sites["accuracy"], width=0.4, color=colour, label="top-1 correct")
    ax.errorbar(
        x - 0.2,
        sites["accuracy"],
        yerr=[
            sites["accuracy"] - sites["accuracy_low"],
            sites["accuracy_high"] - sites["accuracy"],
        ],
        fmt="none",
        ecolor=INK_2,
        capsize=2,
        lw=1,
    )
    ax.bar(
        x + 0.2,
        sites["predicted_biopsy_organ"].fillna(0),
        width=0.4,
        color="#b8b7b3",
        label="predicted the biopsy organ",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{s.replace('_', ' ')}\n(n={n})"
            for s, n in zip(sites["biopsy_site"], sites["n"], strict=True)
        ],
        fontsize=7,
        rotation=30,
        ha="right",
    )
    ax.set(ylim=(0, 1.02), ylabel="Fraction of patients", title="MET500 by biopsy site")
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(figures / "fig13_met500_calibration.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
