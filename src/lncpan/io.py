"""Loading the processed matrices with a consistent label encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lncpan.config import Config


@dataclass
class Dataset:
    X: pd.DataFrame
    samples: pd.DataFrame
    classes: np.ndarray  # sorted cancer-type abbreviations; y is an index into this array

    def split(self, name: str) -> tuple[pd.DataFrame, np.ndarray]:
        m = (self.samples["split"] == name).to_numpy()
        y = np.searchsorted(self.classes, self.samples.loc[m, "cancer_type"].to_numpy())
        return self.X.loc[m], y


def load_dataset(cfg: Config, universe: str) -> Dataset:
    proc = cfg.path("processed")
    samples = pd.read_parquet(proc / "samples.parquet").set_index("sample", drop=False)
    X = pd.read_parquet(proc / f"expression_{universe}.parquet").loc[samples.index]
    classes = np.sort(samples.loc[samples["cohort"] == "primary", "cancer_type"].unique())
    return Dataset(X=X, samples=samples.reset_index(drop=True), classes=classes)
