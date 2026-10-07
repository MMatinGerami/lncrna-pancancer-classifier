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


def _call_and_release(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    finally:
        # joblib keeps its loky workers alive for 300 s after the last job (n_jobs=-1 in
        # GridSearchCV), and the spawned process cannot exit until they do
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True, kill_workers=True)


def run_isolated(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(_call_and_release, fn, *args, **kwargs).result()
