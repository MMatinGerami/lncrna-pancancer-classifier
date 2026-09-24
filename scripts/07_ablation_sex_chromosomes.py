"""Ablation: what happens if chrX/chrY lncRNAs are allowed back in?

The main analysis removes sex-chromosome genes so that models cannot recognise sex-specific
cancers (OV, UCEC, UCS, CESC, PRAD, TGCT) from the patient's sex. This script tests whether
that choice matters: it adds the chrX/chrY lncRNAs back, retrains the logistic-regression
model with identical settings, and reports (i) test performance per cancer type and (ii) where
sex-linked lncRNAs such as XIST rank among the model's features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from lncpan.annotation import SEX_CHROMOSOMES, gene_universe, read_gene_annotation
from lncpan.config import load_config
from lncpan.data import load_expression
from lncpan.evaluate import evaluate
from lncpan.io import load_dataset
from lncpan.models import build_model

SEX_SPECIFIC = ["OV", "UCEC", "UCS", "CESC", "PRAD", "TGCT", "BRCA"]


def main() -> None:
    cfg = load_config()
    raw, tables = cfg.path("raw"), cfg.path("results") / "tables"
    d = cfg["data"]
    annot = read_gene_annotation(raw / d["annotation_gtf"])
    sex_lnc = gene_universe(annot, "lncRNA", exclude_sex_chromosomes=False)
    sex_lnc = sex_lnc[sex_lnc["chrom"].isin(SEX_CHROMOSOMES)]

    ds = load_dataset(cfg, "lncRNA")
    extra = load_expression(raw / d["expression"], sex_lnc["gene_id"], ds.X.index)
    variants = {"autosomal (main analysis)": ds.X, "with chrX/chrY lncRNAs": ds.X.join(extra)}
    names = annot.set_index("gene_id")["gene_name"]

    rows, per_class, ranks = [], {}, []
    mask_tr = (ds.samples["split"] == "train").to_numpy()
    mask_te = (ds.samples["split"] == "test").to_numpy()
    y_all = np.searchsorted(ds.classes, ds.samples["cancer_type"].to_numpy())
    for label, X in variants.items():
        model = build_model("logreg", cfg["features"]["top_k_variance"], cfg.seed)
        model.set_params(clf__C=0.1)
        model.fit(X[mask_tr], y_all[mask_tr])
        proba = model.predict_proba(X[mask_te])
        y_te = y_all[mask_te]
        m = evaluate(y_te, proba, seed=cfg.seed).set_index("metric")
        rows.append(
            {
                "variant": label,
                "n_genes": X.shape[1],
                "macro_f1": m.at["macro_f1", "value"],
                "ci_low": m.at["macro_f1", "ci_low"],
                "ci_high": m.at["macro_f1", "ci_high"],
            }
        )
        f1 = f1_score(y_te, proba.argmax(1), labels=np.arange(len(ds.classes)), average=None)
        per_class[label] = pd.Series(f1, index=ds.classes)

        if label.startswith("with"):
            sel = model.named_steps["select"].support_
            coefs = np.abs(model.named_steps["clf"].coef_).max(axis=0)
            feats = X.columns[sel]
            order = pd.Series(coefs, index=feats).rank(ascending=False).astype(int)
            for gid in feats.intersection(extra.columns):
                ranks.append({"gene_id": gid, "gene_name": names[gid], "rank_of_2000": order[gid]})

    summary = pd.DataFrame(rows)
    pcs = pd.DataFrame(per_class).loc[SEX_SPECIFIC]
    ranks = pd.DataFrame(ranks).sort_values("rank_of_2000") if ranks else pd.DataFrame()
    summary.to_csv(tables / "ablation_sex_chromosomes.csv", index=False)
    pcs.to_csv(tables / "ablation_sex_chromosomes_per_class.csv")
    ranks.to_csv(tables / "ablation_sex_chromosomes_feature_ranks.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(pcs.round(3).to_string())
    print(ranks.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
