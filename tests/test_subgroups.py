import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "subgroups", Path(__file__).parents[1] / "scripts" / "09_subgroups.py"
)
subgroups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subgroups)


def test_sub_stages_collapse_to_main_stage():
    raw = pd.Series(["Stage I", "Stage IIIA", "Stage IVB", "Stage IIC", "Stage IA"])
    assert subgroups.parse_stage(raw).tolist() == ["I", "III", "IV", "II", "I"]


def test_non_stages_are_missing():
    raw = pd.Series(["Stage X", "Stage 0", "IS", "I/II NOS", "[Discrepancy]", None])
    assert subgroups.parse_stage(raw).isna().all()
