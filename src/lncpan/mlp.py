"""A scikit-learn compatible PyTorch multilayer perceptron classifier."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden_dims: tuple[int, ...], n_out: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """MLP trained with AdamW, label smoothing and early stopping on an inner validation split.

    Inputs are expected to be standardised upstream (e.g. by a StandardScaler in a Pipeline).
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 15,
        label_smoothing: float = 0.05,
        val_fraction: float = 0.1,
        random_state: int = 0,
        device: str | None = None,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.label_smoothing = label_smoothing
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device

    def _device(self) -> torch.device:
        return torch.device(self.device) if self.device else default_device()

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float32)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y_idx, test_size=self.val_fraction, stratify=y_idx, random_state=self.random_state
        )
        dev = self._device()
        model = _MLP(X.shape[1], tuple(self.hidden_dims), len(self.classes_), self.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        Xt, yt = torch.from_numpy(X_tr).to(dev), torch.from_numpy(y_tr).long().to(dev)
        Xv, yv = torch.from_numpy(X_va).to(dev), torch.from_numpy(y_va).long().to(dev)

        best_loss, best_state, bad_epochs = np.inf, None, 0
        self.history_: list[dict[str, float]] = []
        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.from_numpy(rng.permutation(len(Xt))).to(dev)
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.cross_entropy(model(Xv), yv).item()
            sched.step(val_loss)
            self.history_.append({"epoch": epoch, "val_loss": val_loss})
            if val_loss < best_loss - 1e-4:
                best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        model.load_state_dict(best_state)
        self.model_ = model.cpu().eval()
        self.best_val_loss_ = best_loss
        self.n_epochs_ = len(self.history_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = validate_data(self, X, dtype=np.float32, reset=False)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
