"""Pool benchmark runs over training seeds (the last step of the Nextflow pipeline).

The test split is fixed by 01_prepare_data.py; the seed changes the CV folds, model
initialisation and the bootstrap. The question is whether seed-to-seed spread is small next
to the test-set bootstrap interval, and whether the lncRNA vs protein-coding gap keeps its
sign across seeds.

Usage:
  python scripts/18_seed_stability.py RUN_DIR [RUN_DIR ...] --out DIR

Each RUN_DIR is the results directory of one 02_benchmark.py call (tables/metrics__*.csv
and tables/search__*.json).

Writes seed_runs.csv, seed_stability.csv and seed_paired_difference.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["macro_f1", "accuracy", "log_loss", "ece"]


def load_runs(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        for search in sorted((d / "tables").glob("search__*.json")):
            info = json.loads(search.read_text())
            tag = search.stem.removeprefix("search__")
            m = pd.read_csv(d / "tables" / f"metrics__{tag}.csv").set_index("metric")
            for metric in METRICS:
                r = m.loc[metric]
                rows.append(
                    {
                        "universe": info["universe"],
                        "model": info["model"],
                        "seed": info["seed"],
                        "metric": metric,
                        "value": r.value,
                        "ci_half_width": (r.ci_high - r.ci_low) / 2,
                        "cv_macro_f1": info["cv_macro_f1"],
                        "minutes": info["minutes"],
                    }
                )
    if not rows:
        raise SystemExit("no runs found")
    runs = pd.DataFrame(rows)
    dup = runs.duplicated(["universe", "model", "seed", "metric"])
    if dup.any():
        raise SystemExit(f"duplicate runs:\n{runs[dup]}")
    return runs


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["universe", "model", "metric"])
    out = g["value"].agg(n_seeds="count", mean="mean", sd="std", min="min", max="max")
    # seed noise relative to test-set sampling noise; < 1 means the bootstrap CI dominates
    out["sd_over_ci_half_width"] = out["sd"] / g["ci_half_width"].mean()
    return out.reset_index()


def paired_difference(runs: pd.DataFrame) -> pd.DataFrame:
    w = runs.pivot_table(index=["model", "seed", "metric"], columns="universe", values="value")
    if not {"lncRNA", "protein_coding"} <= set(w.columns):
        return pd.DataFrame()
    diff = (w["lncRNA"] - w["protein_coding"]).rename("diff").dropna().reset_index()
    g = diff.groupby(["model", "metric"])["diff"]
    return g.agg(
        n_seeds="count",
        mean="mean",
        min="min",
        max="max",
        same_sign=lambda d: bool((d > 0).all() or (d < 0).all()),
    ).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    runs.to_csv(args.out / "seed_runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(args.out / "seed_stability.csv", index=False)
    paired_difference(runs).to_csv(args.out / "seed_paired_difference.csv", index=False)
    f1 = summary[summary.metric == "macro_f1"]
    print(f1.drop(columns="metric").to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
