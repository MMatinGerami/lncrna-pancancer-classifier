"""TCGA sample selection and expression loading."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def parse_barcode(sample: str) -> tuple[str, str]:
    """Split a TCGA sample barcode (e.g. TCGA-A1-A0SB-01) into (patient, sample type code)."""
    parts = sample.split("-")
    if len(parts) < 4 or not sample.startswith("TCGA-"):
        raise ValueError(f"not a TCGA sample barcode: {sample!r}")
    return "-".join(parts[:3]), parts[3][:2]


def select_samples(
    samples: Iterable[str],
    patient_labels: pd.Series,
    sample_types: Iterable[str],
) -> pd.DataFrame:
    """Keep samples of the requested types whose patient has a label; one sample per patient.

    When a patient has several eligible samples, the earliest type in `sample_types` wins,
    so the output is deterministic and patients never appear twice (no train/test leakage).
    """
    order = {t: i for i, t in enumerate(sample_types)}
    rows = []
    for s in samples:
        try:
            patient, stype = parse_barcode(s)
        except ValueError:
            continue
        if stype in order and patient in patient_labels.index:
            rows.append((s, patient, stype, order[stype], patient_labels[patient]))
    df = pd.DataFrame(rows, columns=["sample", "patient", "sample_type", "_rank", "cancer_type"])
    df = df.sort_values(["patient", "_rank", "sample"]).drop_duplicates("patient", keep="first")
    return df.drop(columns="_rank").sort_values("sample").reset_index(drop=True)


def read_patient_labels(clinical_path: str | Path) -> pd.Series:
    """Cancer type abbreviation per patient from the PanCanAtlas clinical table (TCGA-CDR)."""
    clin = pd.read_csv(clinical_path, sep="\t", usecols=["_PATIENT", "cancer type abbreviation"])
    clin = clin.dropna().drop_duplicates("_PATIENT")
    return clin.set_index("_PATIENT")["cancer type abbreviation"]


def xena_log2tpm_to_log2tpm1(x: np.ndarray) -> np.ndarray:
    """Convert Xena log2(TPM + 0.001) values to log2(TPM + 1).

    The +1 pseudocount compresses the noisy low-expression range, which matters for
    lowly expressed lncRNAs.
    """
    tpm = np.clip(np.exp2(x) - 0.001, 0.0, None)
    return np.log2(tpm + 1.0).astype(np.float32)


def load_expression(
    path: str | Path,
    gene_ids: Iterable[str],
    samples: Iterable[str],
    chunksize: int = 4000,
) -> pd.DataFrame:
    """Stream the (genes x samples) Xena matrix and return a samples x genes float32 frame."""
    wanted_genes = set(gene_ids)
    samples = list(samples)
    usecols = ["sample", *samples]
    parts = []
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        index_col="sample",
        chunksize=chunksize,
        dtype={s: np.float32 for s in samples},
    )
    for chunk in reader:
        keep = chunk.loc[chunk.index.isin(wanted_genes), samples]
        if len(keep):
            parts.append(keep)
    mat = pd.concat(parts)
    out = pd.DataFrame(xena_log2tpm_to_log2tpm1(mat.to_numpy().T), index=samples, columns=mat.index)
    out.index.name, out.columns.name = "sample", "gene_id"
    return out
