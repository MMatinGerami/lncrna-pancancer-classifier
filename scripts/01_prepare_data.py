"""Build the analysis-ready matrices from the raw Xena / GENCODE downloads.

Outputs (data/processed):
  genes.parquet                   gene_id, gene_name, gene_type, chrom, universe
  expression_{lncRNA,protein_coding}.parquet   samples x genes, log2(TPM + 1)
  samples.parquet                 sample, patient, sample_type, cancer_type, cohort, split
"""

from __future__ import annotations

import gzip

import pandas as pd
from sklearn.model_selection import train_test_split

from lncpan.annotation import gene_universe, read_gene_annotation
from lncpan.config import load_config
from lncpan.data import load_expression, read_patient_labels, select_samples


def main() -> None:
    cfg = load_config()
    raw, out, tables = cfg.path("raw"), cfg.path("processed"), cfg.path("results") / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]

    annot = read_gene_annotation(raw / d["annotation_gtf"])
    universes = {
        kind: gene_universe(annot, kind, d["exclude_sex_chromosomes"]).assign(universe=kind)
        for kind in ("lncRNA", "protein_coding")
    }
    genes = pd.concat(universes.values(), ignore_index=True)

    with gzip.open(raw / d["expression"], "rt") as fh:
        all_samples = fh.readline().rstrip("\n").split("\t")[1:]
    labels = read_patient_labels(raw / d["clinical"])

    primary = select_samples(all_samples, labels, d["primary_sample_types"]).assign(
        cohort="primary"
    )
    counts = primary["cancer_type"].value_counts()
    keep_types = counts[counts >= d["min_class_size"]].index
    primary = primary[primary["cancer_type"].isin(keep_types)]
    metastatic = select_samples(all_samples, labels, d["metastatic_sample_types"]).assign(
        cohort="metastatic"
    )
    # external test = metastases from patients with no primary tumour in the cohort,
    # so no patient contributes to both training and external evaluation
    metastatic = metastatic[
        metastatic["cancer_type"].isin(keep_types) & ~metastatic["patient"].isin(primary["patient"])
    ]

    train_idx, test_idx = train_test_split(
        primary.index,
        test_size=cfg["split"]["test_size"],
        stratify=primary["cancer_type"],
        random_state=cfg.seed,
    )
    primary.loc[train_idx, "split"] = "train"
    primary.loc[test_idx, "split"] = "test"
    metastatic["split"] = "external"
    samples = pd.concat([primary, metastatic], ignore_index=True)

    for kind, g in universes.items():
        expr = load_expression(raw / d["expression"], g["gene_id"], samples["sample"])
        expr.to_parquet(out / f"expression_{kind}.parquet")
        print(f"{kind}: {expr.shape[0]} samples x {expr.shape[1]} genes")
    present = set(pd.read_parquet(out / "expression_lncRNA.parquet").columns) | set(
        pd.read_parquet(out / "expression_protein_coding.parquet").columns
    )
    genes[genes["gene_id"].isin(present)].to_parquet(out / "genes.parquet")
    samples.to_parquet(out / "samples.parquet")

    cohort = (
        samples.pivot_table(index="cancer_type", columns="split", values="sample", aggfunc="count")
        .fillna(0)
        .astype(int)
        .reindex(columns=["train", "test", "external"], fill_value=0)
    )
    cohort.to_csv(tables / "cohort.csv")
    print(cohort.sum())


if __name__ == "__main__":
    main()
