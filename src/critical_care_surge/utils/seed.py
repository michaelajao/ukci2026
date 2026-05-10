"""Reproducible seeding for `random`, `numpy`, and `torch`."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python's `random`, NumPy, and (if available) PyTorch + CUDA.

    Args:
        seed: Integer seed.
        deterministic_torch: If True, set ``torch.backends.cudnn.deterministic``
            and ``torch.use_deterministic_algorithms`` for full reproducibility.
            Slightly slower; turn off only for hyperparameter sweeps.
    """
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
