"""Data loading, chronological-by-wave splitting, and sliding-window batching.

Single source of truth for everything between ``data/processed/regional_daily.csv``
and a model's ``forward()``. Loads the tidy CSV, partitions by epidemic wave per
``01_RESEARCH_PROGRAMME.md`` §6 (Alpha train / Delta validation / Omicron test),
applies per-region standardisation fitted on the training period only, and
yields ``(X, y)`` mini-batches of shape ``(B, L, F)`` × ``(B, H_max)`` for
multi-horizon forecasting.

Public entry points:
    load_regional_csv   — read tidy CSV into a DataFrame.
    split_by_wave       — chronological splits (train/val/test) on a date axis.
    RegionalDataset     — torch Dataset of sliding windows.
    make_dataloaders    — convenience: returns three DataLoaders.

The forecasting target is mechanical-ventilation bed occupancy (``mv_beds``)
unless overridden. Covariates default to ``admissions``, ``hospital_cases``,
``occupied_beds``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# torch is imported lazily in `RegionalDataset.__getitem__` and `make_dataloaders`
# so that the pure-numpy utilities (loading, splitting, scaling) work in
# environments without torch installed.


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "regional_daily.csv"

DEFAULT_TARGET = "mv_beds"
DEFAULT_COVARIATES: tuple[str, ...] = ("admissions", "hospital_cases", "occupied_beds")

# Chronological-by-wave split per 01_RESEARCH_PROGRAMME.md §6.
# Train: 1 Aug 2020 – 31 May 2021 (Alpha + early vaccination).
# Val:   1 Jun 2021 – 30 Nov 2021 (Delta).
# Test:  1 Dec 2021 – 31 Aug 2022 (Omicron and beyond).
DEFAULT_SPLIT_DATES: dict[str, tuple[str, str]] = {
    "train": ("2020-08-01", "2021-05-31"),
    "val":   ("2021-06-01", "2021-11-30"),
    "test":  ("2021-12-01", "2022-08-31"),
}

# Default sliding-window settings per 02_METHODOLOGY.md §0.
DEFAULT_LOOKBACK = 28
DEFAULT_HORIZONS: tuple[int, ...] = (7, 14, 21, 28)

# NHS region code → canonical pivot order. Stable order matters for graph code.
REGION_ORDER: tuple[str, ...] = ("Y56", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63")


# ---------------------------------------------------------------------------
# Loading + splitting
# ---------------------------------------------------------------------------

def load_regional_csv(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Load the tidy regional daily CSV produced by ``build_regional_dataset.py``.

    Returns a DataFrame with one row per (date, region) and columns:
    ``date`` (datetime), ``region_code``, ``region_name``, plus the four metrics.
    Sorted by ``(date, region_code)``.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["date", "region_code"]).reset_index(drop=True)
    return df


def split_by_wave(
    df: pd.DataFrame,
    split_dates: dict[str, tuple[str, str]] = DEFAULT_SPLIT_DATES,
) -> dict[str, pd.DataFrame]:
    """Split the long-form regional DataFrame into three chronological subsets.

    Returns a dict ``{split_name: subset_df}`` where each subset is a copy
    filtered by inclusive date range.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, (start, end) in split_dates.items():
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        out[name] = df.loc[mask].reset_index(drop=True)
    return out


def pivot_to_wide(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    region_order: Sequence[str] = REGION_ORDER,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Pivot a long-form subset into a 3D array ``(T, R, F)``.

    Args:
        df: long-form DataFrame (date, region_code, metric_columns ...).
        feature_cols: which metric columns to stack as features.
        region_order: ordering of regions along axis 1. Defaults to ``REGION_ORDER``.

    Returns:
        ``(values, dates)`` where ``values`` is a NumPy array of shape
        ``(T, R, F)`` and ``dates`` is the matching ``DatetimeIndex`` of length T.
    """
    pivots = [
        df.pivot(index="date", columns="region_code", values=col)
        .reindex(columns=list(region_order))
        for col in feature_cols
    ]
    arr = np.stack([p.to_numpy(dtype=np.float32) for p in pivots], axis=-1)
    dates = pivots[0].index
    return arr, dates


# ---------------------------------------------------------------------------
# Standardisation (per-region z-score; statistics fitted on train only)
# ---------------------------------------------------------------------------

@dataclass
class RegionScaler:
    """Per-region z-score standardiser fitted on training data only."""

    mean_: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (R, F)
    std_: np.ndarray = field(default_factory=lambda: np.ones(0))     # (R, F)

    def fit(self, x: np.ndarray) -> "RegionScaler":
        # x shape (T, R, F). Statistics over time axis only.
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        # Guard against zero-variance features (replace 0 with 1 to avoid /0).
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform_target(
        self, y_norm: np.ndarray, target_index: int
    ) -> np.ndarray:
        """Invert standardisation for the forecast target only.

        ``y_norm`` shape ``(..., R)``; returns the same shape on the original scale.
        """
        m = self.mean_[:, target_index]
        s = self.std_[:, target_index]
        return y_norm * s + m


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------

def _torch():
    """Lazy import of torch; raises a helpful error if not installed."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required for RegionalDataset / make_dataloaders. "
            "Install via: pip install torch"
        ) from e
    return torch


class RegionalDataset:
    """Sliding-window dataset over a 3D ``(T, R, F)`` regional time series.

    For a series of length T and lookback L with horizons H = (h_1, ..., h_K),
    a sample at index i covers time ``[i, i+L)`` for X and timesteps
    ``[i+L+h_k-1]`` for the K targets. Valid range of ``i`` is therefore
    ``[0, T - L - max(H))``.

    Each item:
        X      tensor (L, R, F)   covariates including the target as a feature
        y      tensor (K, R)      target values at the K horizon offsets
        idx_t  int                index of the last observed timestep (i + L - 1)

    Subclasses ``torch.utils.data.Dataset`` at construction time so the class
    is usable as a torch dataset; falls back to a duck-typed dataset when
    torch is absent (only ``__len__`` and ``__getitem__`` are required).
    """

    def __init__(
        self,
        x: np.ndarray,
        target_index: int,
        lookback: int = DEFAULT_LOOKBACK,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
    ) -> None:
        self.x = x.astype(np.float32, copy=False)
        self.target_index = target_index
        self.lookback = int(lookback)
        self.horizons = tuple(int(h) for h in horizons)
        self._valid_start = self._compute_valid_starts()

    def _compute_valid_starts(self) -> np.ndarray:
        T = self.x.shape[0]
        max_h = max(self.horizons)
        last = T - self.lookback - max_h + 1
        return np.arange(max(0, last))

    def __len__(self) -> int:
        return len(self._valid_start)

    def __getitem__(self, i: int):
        start = int(self._valid_start[i])
        L = self.lookback
        x_window = self.x[start : start + L]                         # (L, R, F)
        y_target_idx = [start + L + h - 1 for h in self.horizons]
        y = self.x[y_target_idx, :, self.target_index]               # (K, R)
        torch = _torch()
        return (
            torch.from_numpy(x_window.copy()),
            torch.from_numpy(y.copy()),
            start + L - 1,
        )


# ---------------------------------------------------------------------------
# Convenience: build full pipeline in one call
# ---------------------------------------------------------------------------

@dataclass
class SplitTensors:
    """Container for fitted-and-standardised arrays + the scaler used."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    scaler: RegionScaler
    feature_cols: tuple[str, ...]
    target_index: int


def build_split_tensors(
    csv_path: Path | str = DEFAULT_CSV,
    target: str = DEFAULT_TARGET,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    split_dates: dict[str, tuple[str, str]] = DEFAULT_SPLIT_DATES,
) -> SplitTensors:
    """Load CSV → split by wave → pivot to (T, R, F) → fit scaler on train → return all three."""
    df = load_regional_csv(csv_path)
    splits = split_by_wave(df, split_dates)

    feature_cols: tuple[str, ...] = (target,) + tuple(c for c in covariates if c != target)
    target_index = 0  # by construction, target is the first feature

    train_arr, train_dates = pivot_to_wide(splits["train"], feature_cols)
    val_arr,   val_dates   = pivot_to_wide(splits["val"],   feature_cols)
    test_arr,  test_dates  = pivot_to_wide(splits["test"],  feature_cols)

    scaler = RegionScaler().fit(train_arr)
    return SplitTensors(
        train=scaler.transform(train_arr),
        val=scaler.transform(val_arr),
        test=scaler.transform(test_arr),
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        scaler=scaler,
        feature_cols=feature_cols,
        target_index=target_index,
    )


def make_dataloaders(
    csv_path: Path | str = DEFAULT_CSV,
    target: str = DEFAULT_TARGET,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    lookback: int = DEFAULT_LOOKBACK,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Convenience: load CSV, split + standardise, build torch DataLoaders for each split.

    Returns:
        (train_loader, val_loader, test_loader, split_tensors).
        ``split_tensors.scaler`` is needed at evaluation time to invert
        standardisation back to the original MV-bed scale.
    """
    torch = _torch()
    from torch.utils.data import DataLoader

    s = build_split_tensors(csv_path, target, covariates)

    def loader(arr: np.ndarray, shuffle: bool):
        ds = RegionalDataset(arr, s.target_index, lookback, horizons)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=False,
        )

    return (
        loader(s.train, shuffle=True),
        loader(s.val, shuffle=False),
        loader(s.test, shuffle=False),
        s,
    )


# ---------------------------------------------------------------------------
# Module smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    s = build_split_tensors()
    print(f"feature_cols: {s.feature_cols}")
    print(f"target_index: {s.target_index}")
    print(f"train: shape={s.train.shape}  dates={s.train_dates[0].date()}..{s.train_dates[-1].date()}")
    print(f"val:   shape={s.val.shape}  dates={s.val_dates[0].date()}..{s.val_dates[-1].date()}")
    print(f"test:  shape={s.test.shape}  dates={s.test_dates[0].date()}..{s.test_dates[-1].date()}")
    print(f"scaler.mean_ shape: {s.scaler.mean_.shape}  std shape: {s.scaler.std_.shape}")

    # Numpy-only sanity check on the windowed dataset.
    train_ds = RegionalDataset(s.train, s.target_index)
    print(f"\nRegionalDataset len(train) = {len(train_ds)}")
    print(f"  expected = T - lookback - max(horizons) + 1 = "
          f"{s.train.shape[0]} - {DEFAULT_LOOKBACK} - {max(DEFAULT_HORIZONS)} + 1 = "
          f"{s.train.shape[0] - DEFAULT_LOOKBACK - max(DEFAULT_HORIZONS) + 1}")

    try:
        import torch  # noqa: F401
        train_loader, val_loader, test_loader, _ = make_dataloaders()
        x, y, idx = next(iter(train_loader))
        print(f"\nbatch X: {tuple(x.shape)}  dtype={x.dtype}")
        print(f"batch y: {tuple(y.shape)}  dtype={y.dtype}")
        print(f"len(train)={len(train_loader.dataset)}  len(val)={len(val_loader.dataset)}  "
              f"len(test)={len(test_loader.dataset)}")
    except ImportError:
        print("\n[torch not installed — skipping DataLoader smoke test]")
