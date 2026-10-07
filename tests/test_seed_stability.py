import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from lncpan.config import load_config, with_overrides
from lncpan.io import load_dataset

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seed_stability = _load("seed_stability", "18_seed_stability.py")
toy_data = _load("toy_data", "make_toy_data.py")


def _write_run(root, universe, model, seed, f1):
    tables = root / f"{universe}__{model}__seed{seed}" / "tables"
    tables.mkdir(parents=True)
    tag = f"{universe}__{model}"
    rows = [(m, f1 if m == "macro_f1" else 0.5) for m in seed_stability.METRICS]
    pd.DataFrame(
        [{"metric": m, "value": v, "ci_low": v - 0.02, "ci_high": v + 0.02} for m, v in rows]
    ).to_csv(tables / f"metrics__{tag}.csv", index=False)
    info = {"universe": universe, "model": model, "seed": seed, "cv_macro_f1": f1, "minutes": 1}
    (tables / f"search__{tag}.json").write_text(json.dumps(info))
    return tables.parent


def test_summary_and_paired_difference(tmp_path):
    dirs = [
        _write_run(tmp_path, "lncRNA", "logreg", 1, 0.95),
        _write_run(tmp_path, "lncRNA", "logreg", 2, 0.93),
        _write_run(tmp_path, "protein_coding", "logreg", 1, 0.94),
        _write_run(tmp_path, "protein_coding", "logreg", 2, 0.94),
    ]
    runs = seed_stability.load_runs(dirs)
    s = seed_stability.summarise(runs).set_index(["universe", "metric"])
    lnc = s.loc[("lncRNA", "macro_f1")]
    assert lnc.n_seeds == 2
    assert lnc["mean"] == pytest.approx(0.94)
    # sd of (0.95, 0.93) is 0.01414; the CI half-width is 0.02
    assert lnc.sd_over_ci_half_width == pytest.approx(0.01414 / 0.02, rel=1e-3)
    d = seed_stability.paired_difference(runs).set_index("metric").loc["macro_f1"]
    assert (d["min"], d["max"]) == pytest.approx((-0.01, 0.01))
    assert not d.same_sign


def test_duplicate_seed_is_an_error(tmp_path):
    a = _write_run(tmp_path / "a", "lncRNA", "logreg", 1, 0.95)
    b = _write_run(tmp_path / "b", "lncRNA", "logreg", 1, 0.95)
    with pytest.raises(SystemExit, match="duplicate"):
        seed_stability.load_runs([a, b])


def test_overrides_point_the_dataset_at_another_directory(tmp_path):
    toy_data.make_toy(tmp_path / "toy", n_classes=3, per_class=10, n_genes=20)
    base = load_config()
    cfg = with_overrides(base, seed=7, processed=tmp_path / "toy", results=tmp_path / "out")
    assert cfg.seed == 7 and base.seed != 7  # the original config is untouched
    assert cfg.path("results") == (tmp_path / "out").resolve()
    ds = load_dataset(cfg, "lncRNA")
    assert ds.X.shape == (30, 20)
    assert ds.classes.tolist() == ["T0", "T1", "T2"]
