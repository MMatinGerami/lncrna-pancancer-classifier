import numpy as np
import pandas as pd
import pytest

from lncpan.io import Dataset


def _dataset(labels, classes):
    samples = pd.DataFrame({"split": ["test"] * len(labels), "cancer_type": labels})
    X = pd.DataFrame(np.zeros((len(labels), 2)))
    return Dataset(X=X, samples=samples, classes=np.array(classes))


def test_split_encodes_labels_as_class_indices():
    _, y = _dataset(["LUAD", "BRCA", "LUSC"], ["BRCA", "LUAD", "LUSC"]).split("test")
    assert y.tolist() == [1, 0, 2]


def test_split_rejects_a_label_outside_the_classes():
    with pytest.raises(ValueError, match="CHOL"):
        _dataset(["BRCA", "CHOL"], ["BRCA", "LUAD"]).split("test")
