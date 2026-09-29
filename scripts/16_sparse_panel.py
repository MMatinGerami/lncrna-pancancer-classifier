"""A small lncRNA panel: how few genes can a sparse model get away with?

The feature-budget curve (scripts/03) picks genes by variance and saturates around 500. A
targeted assay (capture RNA-seq, NanoString, qPCR) would rather have a few dozen genes chosen
for the task. Here an L1-penalised multinomial logistic regression is fitted on the training
split for a range of penalties; the genes with a non-zero coefficient for any class form the
panel, and a plain L2 model restricted to that panel is then refit and scored on the test
split and on the external melanoma metastases. The L2 refit separates gene selection from
prediction, so the panel's score is not inflated by the shrinkage that chose it.

Writes results/tables/sparse_panel.csv, results/tables/sparse_panel_genes.csv (the smallest
panel within two points of the full model) and figures/fig16_sparse_panel.png.
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lncpan.config import load_config
from lncpan.features import TopVarianceSelector
from lncpan.io import load_dataset
from lncpan.models import build_model
from lncpan.plotting import INK_2, INK_3, UNIVERSE_COLORS, apply_style

UNIVERSE = "lncRNA"
L1_C = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
TOLERANCE = 0.02


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables, figures = res / "tables", res / "figures"
    ds = load_dataset(cfg, UNIVERSE)
    X_tr, y_tr = ds.split("train")
    X_te, y_te = ds.split("test")
    X_ex, y_ex = ds.split("external")
    k = cfg["features"]["top_k_variance"]
    best = json.loads((tables / f"search__{UNIVERSE}__logreg.json").read_text())["best_params"]

    full = build_model("logreg", k, cfg.seed).set_params(**best).fit(X_tr, y_tr)
    full_f1 = f1_score(y_te, full.predict(X_te), average="macro")
    print(f"full model ({k} genes): test macro-F1 {full_f1:.3f}")

    select = TopVarianceSelector(k=k).fit(X_tr.to_numpy())
    candidates = X_tr.columns[select.support_]
    rows, panels = [], {}
    for C in L1_C:
        t0 = time.time()
        l1 = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        l1_ratio=1.0, C=C, solver="saga", max_iter=2000, random_state=cfg.seed
                    ),
                ),
            ]
        ).fit(X_tr[candidates], y_tr)
        coef = l1.named_steps["clf"].coef_
        panel = candidates[(np.abs(coef) > 0).any(axis=0)]
        if len(panel) == 0:
            continue
        refit = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=best["clf__C"], max_iter=3000, random_state=cfg.seed)),
            ]
        ).fit(X_tr[panel], y_tr)
        pred_te, pred_ex = refit.predict(X_te[panel]), refit.predict(X_ex[panel])
        rows.append(
            {
                "C": C,
                "n_genes": len(panel),
                "genes_per_class_mean": float((np.abs(coef) > 0).sum(1).mean()),
                "test_macro_f1": f1_score(y_te, pred_te, average="macro"),
                "test_accuracy": accuracy_score(y_te, pred_te),
                "test_top3": top3(refit, X_te[panel], y_te),
                "external_accuracy": accuracy_score(y_ex, pred_ex),
                "minutes": (time.time() - t0) / 60,
            }
        )
        panels[len(panel)] = (panel, coef)
        print(
            f"C={C}: {len(panel)} genes, test macro-F1 {rows[-1]['test_macro_f1']:.3f}, "
            f"metastases accuracy {rows[-1]['external_accuracy']:.3f} "
            f"({rows[-1]['minutes']:.1f} min)",
            flush=True,
        )
    out = pd.DataFrame(rows)
    out["full_model_macro_f1"] = full_f1
    out.to_csv(tables / "sparse_panel.csv", index=False)

    ok = out[out["test_macro_f1"] >= full_f1 - TOLERANCE]
    chosen = int(ok["n_genes"].min()) if not ok.empty else int(out["n_genes"].max())
    panel, coef = panels[chosen]
    genes = pd.read_parquet(cfg.path("processed") / "genes.parquet").set_index("gene_id")
    used_by = [
        ";".join(ds.classes[np.flatnonzero(np.abs(coef[:, j]) > 0)])
        for j in np.flatnonzero(candidates.isin(panel))
    ]
    pd.DataFrame(
        {
            "gene_id": panel,
            "gene_name": genes.loc[panel, "gene_name"].to_numpy(),
            "gene_type": genes.loc[panel, "gene_type"].to_numpy(),
            "n_classes_using": [len(u.split(";")) for u in used_by],
            "classes_using": used_by,
        }
    ).to_csv(tables / "sparse_panel_genes.csv", index=False)
    print(f"panel within {TOLERANCE:.0%} of the full model: {chosen} genes")
    plot(out, full_f1, chosen, figures)


def top3(model, X, y) -> float:
    p = model.predict_proba(X)
    top = np.argsort(-p, axis=1)[:, :3]
    return float((top == y[:, None]).any(1).mean())


def plot(out: pd.DataFrame, full_f1: float, chosen: int, figures) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colour = UNIVERSE_COLORS[UNIVERSE]
    ax.plot(out["n_genes"], out["test_macro_f1"], "o-", color=colour, label="test macro-F1")
    ax.plot(
        out["n_genes"],
        out["external_accuracy"],
        "s--",
        color=colour,
        alpha=0.55,
        label="metastases accuracy",
    )
    ax.axhline(full_f1, color=INK_3, lw=1, ls=":")
    ax.text(
        out["n_genes"].min(), full_f1 + 0.004, "full model (2,000 genes)", fontsize=7, color=INK_2
    )
    ax.axvline(chosen, color=INK_3, lw=1, ls="--")
    ax.text(
        chosen * 1.04,
        0.62,
        f"{chosen} genes:\nwithin 2 points\nof the full model",
        fontsize=7,
        color=INK_2,
    )
    ax.set(
        xscale="log",
        xticks=out["n_genes"].tolist(),
        xticklabels=[str(n) for n in out["n_genes"]],
        xlabel="Genes in the panel (L1 selection, L2 refit)",
        ylabel="Score",
        ylim=(0.45, 1.0),
        title="A sparse lncRNA panel",
    )
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(figures / "fig16_sparse_panel.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
