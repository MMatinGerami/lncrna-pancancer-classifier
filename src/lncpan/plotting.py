"""Shared figure style."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e0"
UNIVERSE_COLORS = {"lncRNA": "#2a78d6", "protein_coding": "#eb6834"}
UNIVERSE_LABELS = {"lncRNA": "lncRNA", "protein_coding": "Protein-coding"}
MODEL_LABELS = {"logreg": "Logistic regression", "xgboost": "XGBoost", "mlp": "MLP (PyTorch)"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7f6f3", "#9cc3ef", "#2a78d6", "#0d3f80"]
)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#efeeea", "#e34948"])


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": INK_3,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )
