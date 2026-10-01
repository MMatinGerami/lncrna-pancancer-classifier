"""Is the classifier reading the tumour or its neighbourhood?

Bulk RNA-seq mixes tumour cells with stroma and immune cells (README, Limitations). This script
asks two questions with the protein-coding expression already in data/processed:

1. Errors. Are misclassified held-out tumours richer in immune or stromal signal than correctly
   classified tumours *of the same cancer type*? And does the model's confidence fall as that
   signal rises (all 1,868 test tumours, more power than the ~60 errors)?
2. Markers. Do the top SHAP marker lncRNAs of each cancer type track immune or stromal signal
   within that type, i.e. are some "markers" reporting the micro-environment?

Immune and stromal scores are a transparent proxy for ESTIMATE (Yoshihara et al., Nat Commun
2013, doi:10.1038/ncomms3612): the mean z-score of a small panel of canonical leukocyte and
fibroblast/extracellular-matrix genes. Every score is then standardised *within* cancer type,
because cancer types differ in stroma for reasons that are part of their identity.

Outputs: results/tables/microenvironment_{errors,confidence,markers}.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from lncpan.config import load_config

IMMUNE = ["PTPRC", "CD2", "CD3E", "CD53", "LAPTM5", "CORO1A", "CD68", "CD14"]
STROMAL = ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "DCN", "LUM", "FAP", "PDGFRB"]
MODEL = "lncRNA__logreg"
TOP_MARKERS = 10
STRONG = 0.5  # |Spearman rho| above which a marker is called micro-environmental
N_BOOT = 2000


def panel_score(expr: pd.DataFrame, genes: pd.DataFrame, panel: list[str]) -> pd.Series:
    sym = genes.set_index("gene_id").gene_name
    cols = [c for c in expr.columns if sym.get(c) in set(panel)]
    z = (expr[cols] - expr[cols].mean()) / expr[cols].std()
    return z.mean(axis=1)


def within_type(score: pd.Series, cancer: pd.Series) -> pd.Series:
    g = score.groupby(cancer)
    return (score - g.transform("mean")) / g.transform("std")


def main() -> None:
    cfg = load_config()
    proc, res = cfg.path("processed"), cfg.path("results")
    tables = res / "tables"
    genes = pd.read_parquet(proc / "genes.parquet")
    samples = pd.read_parquet(proc / "samples.parquet").set_index("sample")
    pc = pd.read_parquet(proc / "expression_protein_coding.parquet")
    cancer = samples.cancer_type.reindex(pc.index)
    found = {
        k: sorted(set(genes.gene_name[genes.gene_id.isin(pc.columns)]) & set(p))
        for k, p in {"immune": IMMUNE, "stromal": STROMAL}.items()
    }
    print("panel genes found:", found, flush=True)
    scores = pd.DataFrame(
        {
            "immune": within_type(panel_score(pc, genes, IMMUNE), cancer),
            "stromal": within_type(panel_score(pc, genes, STROMAL), cancer),
        }
    )

    # 1. errors and confidence on the held-out test set
    pred = pd.read_parquet(res / "predictions" / f"test__{MODEL}.parquet")
    proba = pred.drop(columns="true")
    t = pd.DataFrame(
        {
            "true": pred["true"],
            "pred": proba.idxmax(axis=1),
            "confidence": proba.max(axis=1),
        }
    ).join(scores)
    t["error"] = t["true"] != t["pred"]
    rng = np.random.default_rng(cfg.seed)
    err_rows, conf_rows = [], []
    for s in ["immune", "stromal"]:
        e, c = t.loc[t.error, s].dropna(), t.loc[~t.error, s].dropna()
        diff = e.median() - c.median()
        boots = [
            np.median(rng.choice(e, len(e))) - np.median(rng.choice(c, len(c)))
            for _ in range(N_BOOT)
        ]
        err_rows.append(
            dict(
                score=s,
                n_errors=len(e),
                n_correct=len(c),
                median_errors=e.median(),
                median_correct=c.median(),
                difference=diff,
                ci_low=np.quantile(boots, 0.025),
                ci_high=np.quantile(boots, 0.975),
                mannwhitney_p=mannwhitneyu(e, c).pvalue,
            )
        )
        rho, p = spearmanr(t[s], t.confidence, nan_policy="omit")
        tert = pd.qcut(t[s], 3, labels=["low", "middle", "high"])
        conf_rows.append(
            dict(
                score=s,
                n=int(t[s].notna().sum()),
                spearman_rho_with_confidence=rho,
                p=p,
                **{
                    f"error_rate_{k}": v
                    for k, v in t.groupby(tert, observed=True).error.mean().items()
                },
            )
        )
    errors, conf = pd.DataFrame(err_rows), pd.DataFrame(conf_rows)
    errors.to_csv(tables / "microenvironment_errors.csv", index=False)
    conf.to_csv(tables / "microenvironment_confidence.csv", index=False)
    print(errors.round(4).to_string(index=False))
    print(conf.round(4).to_string(index=False))

    # 2. do top markers track the micro-environment within their own cancer type?
    lnc = pd.read_parquet(proc / "expression_lncRNA.parquet")
    markers = pd.read_csv(tables / "shap_markers_per_cancer.csv")
    markers = markers[markers["rank"] <= TOP_MARKERS]
    rows = []
    for _, m in markers.iterrows():
        idx = cancer.index[cancer == m.cancer_type].intersection(lnc.index)
        if m.gene_id not in lnc.columns or len(idx) < 20:
            continue
        x = lnc.loc[idx, m.gene_id]
        r_imm = spearmanr(x, scores.loc[idx, "immune"], nan_policy="omit")[0]
        r_str = spearmanr(x, scores.loc[idx, "stromal"], nan_policy="omit")[0]
        rows.append(
            dict(
                cancer_type=m.cancer_type,
                rank=m["rank"],
                gene_name=m.gene_name,
                n=len(idx),
                rho_immune=r_imm,
                rho_stromal=r_str,
                microenvironmental=max(abs(r_imm), abs(r_str)) >= STRONG,
            )
        )
    mk = pd.DataFrame(rows)
    mk.to_csv(tables / "microenvironment_markers.csv", index=False)
    print(
        f"markers: {mk.microenvironmental.sum()} of {len(mk)} top-{TOP_MARKERS} markers have "
        f"|rho| >= {STRONG} with the immune or stromal score within their cancer type"
    )
    print(mk[mk.microenvironmental].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
