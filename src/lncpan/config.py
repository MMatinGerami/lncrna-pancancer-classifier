"""Configuration loading."""

from __future__ import annotations

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
