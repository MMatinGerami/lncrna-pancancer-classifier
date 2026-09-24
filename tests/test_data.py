import gzip

import numpy as np
import pandas as pd
import pytest

from lncpan.annotation import gene_universe, read_gene_annotation
from lncpan.data import parse_barcode, select_samples, xena_log2tpm_to_log2tpm1


def test_parse_barcode():
    assert parse_barcode("TCGA-A1-A0SB-01") == ("TCGA-A1-A0SB", "01")
    with pytest.raises(ValueError):
        parse_barcode("GTEX-1117F-0226")


def test_select_samples_one_per_patient_and_priority():
    labels = pd.Series({"TCGA-AA-0001": "COAD", "TCGA-AA-0002": "LAML"})
    samples = ["TCGA-AA-0001-01", "TCGA-AA-0001-11", "TCGA-AA-0002-03", "TCGA-AA-0003-01"]
    out = select_samples(samples, labels, ["01", "03"])
    assert out["patient"].is_unique
    assert set(out["sample"]) == {
        "TCGA-AA-0001-01",
        "TCGA-AA-0002-03",
    }  # normal and unlabeled dropped


def test_xena_transform_roundtrip():
    tpm = np.array([0.0, 1.0, 100.0])
    x = np.log2(tpm + 0.001)
    np.testing.assert_allclose(xena_log2tpm_to_log2tpm1(x), np.log2(tpm + 1), atol=1e-4)


def test_gene_universes(tmp_path):
    gtf = tmp_path / "mini.gtf.gz"
    rows = [
        ("chr1", "gene", "G1", "lincRNA", "L1"),
        ("chr1", "gene", "G2", "protein_coding", "P1"),
        ("chrX", "gene", "G3", "lincRNA", "XIST"),
        ("chr1", "transcript", "G1", "lincRNA", "L1"),
    ]
    lines = [
        f"{chrom}\tHAVANA\t{feat}\t1\t10\t.\t+\t.\t"
        f'gene_id "{gid}"; gene_type "{gtype}"; gene_name "{name}";'
        for chrom, feat, gid, gtype, name in rows
    ]
    with gzip.open(gtf, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    annot = read_gene_annotation(gtf)
    assert len(annot) == 3
    assert gene_universe(annot, "lncRNA")["gene_id"].tolist() == ["G1"]
    assert gene_universe(annot, "lncRNA", exclude_sex_chromosomes=False)["gene_id"].tolist() == [
        "G1",
        "G3",
    ]
    assert gene_universe(annot, "protein_coding")["gene_id"].tolist() == ["G2"]
