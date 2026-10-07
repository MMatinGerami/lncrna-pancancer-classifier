"""Configuration loading."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        p = REPO_ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str | Path = REPO_ROOT / "configs" / "default.yaml") -> Config:
    with open(path) as fh:
        return Config(yaml.safe_load(fh))


def with_overrides(cfg: Config, seed: int | None = None, **paths: str | Path | None) -> Config:
    """Copy of `cfg` with another seed and/or absolute data and results directories.

    Used when a workflow engine runs each task in its own directory.
    """
    raw = copy.deepcopy(cfg.raw)
    if seed is not None:
        raw["seed"] = int(seed)
    raw["paths"].update({k: str(Path(v).resolve()) for k, v in paths.items() if v is not None})
    return Config(raw)
