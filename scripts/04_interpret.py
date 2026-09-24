"""SHAP attribution for the lncRNA XGBoost model: which lncRNAs mark each cancer type?"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap

from lncpan.config import load_config
from lncpan.io import load_dataset


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    ds = load_dataset(cfg, "lncRNA")
    X_te, y_te = ds.split("test")
    pipe = joblib.load(res / "models" / "lncRNA__xgboost.joblib")
    select, clf = pipe.named_steps["select"], pipe.named_steps["clf"]
    Xs = select.transform(X_te.to_numpy())
    features = X_te.columns[select.support_]

    sv = shap.TreeExplainer(clf).shap_values(Xs)  # (n_samples, n_features, n_classes)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] == len(ds.classes) and sv.shape[1] == len(Xs):
        sv = np.moveaxis(sv, 0, -1)

    genes = pd.read_parquet(cfg.path("processed") / "genes.parquet").set_index("gene_id")
    rows = []
    for c, cancer in enumerate(ds.classes):
        in_class = y_te == c
        # mean signed SHAP towards this class, among tumours of this class
        contrib = sv[in_class, :, c].mean(axis=0)
        for rank, j in enumerate(np.argsort(-contrib)[:10], start=1):
            gid = features[j]
            rows.append(
                {
                    "cancer_type": cancer,
                    "rank": rank,
                    "gene_id": gid,
                    "gene_name": genes.at[gid, "gene_name"],
                    "gene_type": genes.at[gid, "gene_type"],
                    "mean_shap": contrib[j],
                }
            )
    markers = pd.DataFrame(rows)
    markers.to_csv(res / "tables" / "shap_markers_per_cancer.csv", index=False)

    glob = pd.DataFrame(
        {
            "gene_id": features,
            "gene_name": genes.loc[features, "gene_name"].to_numpy(),
            "gene_type": genes.loc[features, "gene_type"].to_numpy(),
            "mean_abs_shap": np.abs(sv).sum(axis=2).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    glob.to_csv(res / "tables" / "shap_global.csv", index=False)
    print(markers[markers["rank"] <= 3].to_string(index=False))


if __name__ == "__main__":
    main()
