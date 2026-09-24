"""GENCODE annotation: gene biotypes and the lncRNA / protein-coding gene universes."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd

# GENCODE long non-coding biotypes (pre-v31 nomenclature, as used by GENCODE v23).
# `processed_transcript` is excluded on purpose: it mixes retained-intron fragments of
# protein-coding loci with genuine non-coding genes.
LNCRNA_BIOTYPES = frozenset(
    {
        "lincRNA",
        "antisense",
        "sense_intronic",
        "sense_overlapping",
        "3prime_overlapping_ncrna",
        "macro_lncRNA",
        "bidirectional_promoter_lncRNA",
        "non_coding",
    }
)
SEX_CHROMOSOMES = frozenset({"chrX", "chrY"})

_ATTR = re.compile(r'(gene_id|gene_type|gene_name) "([^"]+)"')


def read_gene_annotation(gtf_path: str | Path) -> pd.DataFrame:
    """Return one row per gene with columns gene_id, gene_name, gene_type, chrom."""
    rows = []
    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 8)
            if fields[2] != "gene":
                continue
            attrs = dict(_ATTR.findall(fields[8]))
            rows.append((attrs["gene_id"], attrs["gene_name"], attrs["gene_type"], fields[0]))
    return pd.DataFrame(rows, columns=["gene_id", "gene_name", "gene_type", "chrom"])


def gene_universe(
    annotation: pd.DataFrame, kind: str, exclude_sex_chromosomes: bool = True
) -> pd.DataFrame:
    """Subset the annotation to `kind` in {"lncRNA", "protein_coding"}."""
    if kind == "lncRNA":
        mask = annotation["gene_type"].isin(LNCRNA_BIOTYPES)
    elif kind == "protein_coding":
        mask = annotation["gene_type"].eq("protein_coding")
    else:
        raise ValueError(f"unknown gene universe: {kind!r}")
    if exclude_sex_chromosomes:
        mask &= ~annotation["chrom"].isin(SEX_CHROMOSOMES)
    return annotation.loc[mask].reset_index(drop=True)
