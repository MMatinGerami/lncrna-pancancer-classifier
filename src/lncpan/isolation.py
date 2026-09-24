"""Run work in a fresh interpreter.

XGBoost (Homebrew libomp) and PyTorch (bundled libomp) ship different OpenMP runtimes that
cannot coexist in one macOS process, whichever loads first. Each model therefore trains and
predicts in its own spawned process, and the models module imports each library lazily.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def run_isolated(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args, **kwargs).result()
