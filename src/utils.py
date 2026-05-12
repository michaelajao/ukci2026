"""Shared infrastructure helpers for the UKCI 2026 codebase."""

from __future__ import annotations

import hashlib
import os
import random
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator

import numpy as np


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Return ``data/`` under the repository root."""
    return repo_root() / "data"


def raw_data_dir(*parts: str) -> Path:
    """Return a path under ``data/raw``."""
    return data_dir() / "raw" / Path(*parts)


def processed_data_dir(*parts: str) -> Path:
    """Return a path under ``data/processed``."""
    return data_dir() / "processed" / Path(*parts)


def results_dir(*parts: str) -> Path:
    """Return a path under ``results``."""
    return repo_root() / "results" / Path(*parts)


def figures_dir(*parts: str) -> Path:
    """Return a path under ``figures``."""
    return repo_root() / "figures" / Path(*parts)


def configure_utf8_stdout() -> None:
    """Use UTF-8 stdout where the runtime supports reconfiguration."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def set_windows_openmp_env() -> None:
    """Allow duplicate OpenMP runtimes on Windows scientific stacks."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def elapsed_timer() -> Iterator[Callable[[], float]]:
    """Yield a callable returning elapsed seconds since context entry."""
    start = perf_counter()
    yield lambda: perf_counter() - start


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python's ``random``, NumPy, and optionally PyTorch/CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
