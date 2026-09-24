"""Leakage-safe feature selection."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class TopVarianceSelector(TransformerMixin, BaseEstimator):
    """Keep the `k` most variable features, estimated on the data passed to `fit` only.

    Used as the first pipeline step so that, inside cross-validation, feature selection
    never sees the held-out fold.
    """

    def __init__(self, k: int = 2000):
        self.k = k

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float32)
        var = X.var(axis=0)
        k = min(self.k, X.shape[1])
        # stable sort so ties are broken by column order -> deterministic
        self.support_ = np.sort(np.argsort(-var, kind="stable")[:k])
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = validate_data(self, X, dtype=np.float32, reset=False)
        return X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        names = (
            np.asarray(input_features)
            if input_features is not None
            else getattr(
                self, "feature_names_in_", np.array([f"x{i}" for i in range(self.n_features_in_)])
            )
        )
        return names[self.support_]
