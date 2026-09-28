"""External validation on MET500: metastatic biopsies from another centre and pipeline.

MET500 (Robinson et al., Nature 2017; UCSC Xena) holds RNA-seq of metastatic tumours with
the primary site recorded. Nothing about it was seen in training: different institution,
sequencing (capture and poly-A libraries), quantification (FPKM) and mostly metastatic
biopsy sites (liver, lymph node, soft tissue). It is the closest public analogue to the
cancer-of-unknown-primary setting the classifier is meant for.

Preparation: FPKM is converted to TPM over all quantified genes, then log2(TPM + 1) as in
training; genes are matched to the model's GENCODE v23 IDs without version; genes absent
from MET500 are set to zero and counted. MET500 cohorts that map to several TCGA classes
(LUNG, COLO, KDNY) count as correct when the prediction is any of them. One library per
patient is kept, preferring poly-A (the TCGA protocol); the capture-only patients form a
platform-shift subgroup.

The public MET500 matrix quantifies a coding-centred gene set and contains almost none of
the lncRNA universe, so the lncRNA model receives no informative input on it: its results
show how the model behaves without its genes (it abstains), and the protein-coding model
carries the external test.

Writes results/tables/met500_{summary,per_cohort,conformal}.csv and figures/fig12_met500.png.
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lncpan.config import load_config
from lncpan.conformal import prediction_sets
from lncpan.plotting import INK_2, INK_3, UNIVERSE_COLORS, apply_style

COHORT_TO_TCGA = {
    "BRCA": {"BRCA"},
    "PRAD": {"PRAD"},
    "SARC": {"SARC"},
    "CHOL": {"CHOL"},
    "HNSC": {"HNSC"},
    "LUNG": {"LUAD", "LUSC"},
    "BLCA": {"BLCA"},
    "SKCM": {"SKCM"},
    "PAAD": {"PAAD"},
    "OV": {"OV"},
    "ESCA": {"ESCA"},
    "COLO": {"COAD", "READ"},
    "ACC": {"ACC"},
    "STAD": {"STAD"},
    "KDNY": {"KIRC", "KIRP", "KICH"},
    "HCC": {"LIHC"},
    "GBM": {"GBM"},
    "THYM": {"THYM"},
    "THCA": {"THCA"},
    "TGCT": {"TGCT"},
}
N_BOOT = 1000


def load_met500(raw) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(raw / "MET500_M.meta.plus.txt", sep="\t")
    meta = meta[meta["sample_type"] == "tumor"]
    meta = meta[meta["cohort"].isin(COHORT_TO_TCGA)].copy()
    meta["library"] = np.where(meta["Sample_id"].str.contains("-poly-"), "poly-A", "capture")
    # one library per patient, poly-A preferred
    meta = meta.sort_values(["sample_source", "library"], ascending=[True, False])
    meta = meta.drop_duplicates("sample_source").set_index("Sample_id")
    expr = pd.read_csv(raw / "MET500_M.mx.log2.txt.gz", sep="\t", index_col=0)
    expr = expr[meta.index]
    fpkm = np.clip(np.exp2(expr.to_numpy(dtype=np.float64)) - 0.001, 0, None)
    tpm = fpkm / fpkm.sum(axis=0, keepdims=True) * 1e6
    X = pd.DataFrame(np.log2(tpm + 1).T.astype(np.float32), index=meta.index, columns=expr.index)
    X.columns = X.columns.str.replace(r"\.\d+$", "", regex=True)
    return X, meta


def align(X: pd.DataFrame, model, genes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    wanted = list(model.feature_names_in_)
    unversioned = pd.Series(
        genes["gene_id"].str.replace(r"\.\d+$", "", regex=True).to_numpy(), index=genes["gene_id"]
    )
    lookup = {unversioned[g]: g for g in wanted if g in unversioned.index}
    present = {u: v for u, v in lookup.items() if u in X.columns}
    aligned = X[list(present)].rename(columns=present).reindex(columns=wanted, fill_value=0.0)
    return aligned.astype(np.float32), len(wanted) - len(present)


def correct_flags(
    proba: np.ndarray, classes: np.ndarray, allowed: list[set[str]]
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-proba, axis=1)
    top1 = np.array([classes[o[0]] in a for o, a in zip(order, allowed, strict=True)])
    top3 = np.array(
        [any(classes[i] in a for i in o[:3]) for o, a in zip(order, allowed, strict=True)]
    )
    return top1, top3


def bootstrap_mean(x: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return float(x.mean()), *np.percentile(boots, [2.5, 97.5])


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures, models = res / "tables", res / "figures", res / "models"
    genes = pd.read_parquet(cfg.path("processed") / "genes.parquet")
    X_all, meta = load_met500(cfg.path("raw"))
    allowed = [COHORT_TO_TCGA[c] for c in meta["cohort"]]
    print(
        f"MET500: {len(meta)} patients, {meta['cohort'].nunique()} cohorts, "
        f"{(meta['library'] == 'poly-A').sum()} poly-A and "
        f"{(meta['library'] == 'capture').sum()} capture libraries"
    )

    rows, per_cohort = [], []
    for universe in ("lncRNA", "protein_coding"):
        # joblib loads a scikit-learn pipeline produced by this repository's own scripts
        model = joblib.load(models / f"{universe}__logreg.joblib")
        X, n_missing = align(X_all, model, genes)
        classes = np.array(
            json.loads((models / "lncRNA__conformal_thresholds.json").read_text())["classes"]
        )
        proba = model.predict_proba(X)
        top1, top3 = correct_flags(proba, classes, allowed)
        for group, mask in (
            ("all", np.ones(len(meta), bool)),
            *[(lib, (meta["library"] == lib).to_numpy()) for lib in ("poly-A", "capture")],
        ):
            a, lo, hi = bootstrap_mean(top1[mask], cfg.seed)
            t, tlo, thi = bootstrap_mean(top3[mask], cfg.seed)
            rows.append(
                {
                    "universe": universe,
                    "group": group,
                    "n": int(mask.sum()),
                    "genes_missing": n_missing,
                    "accuracy": a,
                    "accuracy_low": lo,
                    "accuracy_high": hi,
                    "top3": t,
                    "top3_low": tlo,
                    "top3_high": thi,
                }
            )
        for cohort, idx in meta.groupby("cohort").indices.items():
            per_cohort.append(
                {
                    "universe": universe,
                    "cohort": cohort,
                    "tcga_classes": "/".join(sorted(COHORT_TO_TCGA[cohort])),
                    "n": len(idx),
                    "accuracy": top1[idx].mean(),
                    "top3": top3[idx].mean(),
                }
            )
        if universe == "lncRNA":
            conf_model = joblib.load(models / "lncRNA__logreg_conformal.joblib")
            thr = json.loads((models / "lncRNA__conformal_thresholds.json").read_text())
            Xc, _ = align(X_all, conf_model, genes)
            pc = conf_model.predict_proba(Xc)
            y_idx = np.array(
                [np.flatnonzero(np.isin(classes, list(a))) for a in allowed], dtype=object
            )
            conf_rows = []
            for alpha, variants in thr["lac"].items():
                for variant, q in variants.items():
                    sets = prediction_sets(pc, np.asarray(q), "lac")
                    covered = np.array([sets[i, ix].any() for i, ix in enumerate(y_idx)])
                    size = sets.sum(1)
                    conf_rows.append(
                        {
                            "alpha": float(alpha),
                            "variant": variant,
                            "coverage": covered.mean(),
                            "mean_size": size.mean(),
                            "empty_frac": (size == 0).mean(),
                            "coverage_poly_a": covered[
                                (meta["library"] == "poly-A").to_numpy()
                            ].mean(),
                            "coverage_capture": covered[
                                (meta["library"] == "capture").to_numpy()
                            ].mean(),
                        }
                    )
            pd.DataFrame(conf_rows).to_csv(tables / "met500_conformal.csv", index=False)
            print(pd.DataFrame(conf_rows).round(3).to_string(index=False))
    summary, pc_df = pd.DataFrame(rows), pd.DataFrame(per_cohort)
    summary.to_csv(tables / "met500_summary.csv", index=False)
    pc_df.to_csv(tables / "met500_per_cohort.csv", index=False)
    print(summary.round(3).to_string(index=False))
    plot(summary, pc_df, figures)


def plot(summary: pd.DataFrame, pc: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1.4]})
    ax = axes[0]
    d = pc[pc.universe == "protein_coding"].sort_values("n", ascending=False)
    x = np.arange(len(d))
    ax.bar(
        x - 0.18, d["accuracy"], width=0.36, color=UNIVERSE_COLORS["protein_coding"], label="top-1"
    )
    ax.bar(x + 0.18, d["top3"], width=0.36, color="#f5b79b", label="top-3")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c} ({n})" for c, n in zip(d["cohort"], d["n"], strict=True)],
        fontsize=7,
        rotation=90,
    )
    ax.set(
        ylim=(0, 1.02),
        ylabel="Fraction of patients",
        title="MET500 metastases, protein-coding model: primary site recovered",
    )
    ax.legend(loc="lower left", fontsize=7)
    ax = axes[1]
    s = summary[summary.universe == "protein_coding"].set_index("group")
    groups = ["all", "poly-A", "capture"]
    ax.bar(
        np.arange(3),
        s.loc[groups, "accuracy"],
        yerr=[
            s.loc[groups, "accuracy"] - s.loc[groups, "accuracy_low"],
            s.loc[groups, "accuracy_high"] - s.loc[groups, "accuracy"],
        ],
        capsize=3,
        color=UNIVERSE_COLORS["protein_coding"],
    )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"{g}\nn={int(s.loc[g, 'n'])}" for g in groups], fontsize=8)
    ax.set(ylim=(0, 1.02), title="By library type")
    ax.axhline(s.loc["all", "top3"], color=INK_3, ls="--", lw=1)
    ax.text(2.45, s.loc["all", "top3"], "top-3", fontsize=7, color=INK_2, va="bottom", ha="right")
    fig.savefig(figures / "fig12_met500.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
