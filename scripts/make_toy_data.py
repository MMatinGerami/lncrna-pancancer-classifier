"""Write a small synthetic dataset in the layout of data/processed/.

Used by the Nextflow test profile and CI, which have no TCGA data. Classes differ in the
mean of a few genes, so every model should score well above chance.

Usage:
  python scripts/make_toy_data.py OUT_DIR [--classes 4] [--per-class 40] [--genes 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_toy(out: Path, n_classes: int, per_class: int, n_genes: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    types = [f"T{i}" for i in range(n_classes)]
    labels = np.repeat(types, per_class)
    samples = pd.DataFrame(
        {
            "sample": [f"TOY-{i:04d}-01" for i in range(len(labels))],
            "patient": [f"TOY-{i:04d}" for i in range(len(labels))],
            "sample_type": "01",
            "cancer_type": labels,
            "cohort": "primary",
            "split": np.where(rng.random(len(labels)) < 0.25, "test", "train"),
        }
    )
    samples.to_parquet(out / "samples.parquet")
    genes = []
    for universe in ("lncRNA", "protein_coding"):
        ids = [f"ENSG{universe[:3].upper()}{j:05d}.1" for j in range(n_genes)]
        X = rng.normal(0.0, 1.0, (len(labels), n_genes))
        for c in range(n_classes):  # five marker genes per class
            X[labels == types[c], c * 5 : c * 5 + 5] += 0.8
        pd.DataFrame(X.astype(np.float32), index=samples["sample"], columns=ids).to_parquet(
            out / f"expression_{universe}.parquet"
        )
        genes.append(pd.DataFrame({"gene_id": ids, "gene_name": ids, "universe": universe}))
    pd.concat(genes).to_parquet(out / "genes.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--genes", type=int, default=60)
    args = ap.parse_args()
    make_toy(args.out, args.classes, args.per_class, args.genes)


if __name__ == "__main__":
    main()
