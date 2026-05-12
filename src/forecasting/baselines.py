"""Forecasting baselines for the UKCI 2026 E1 experiment.

Retained baselines after the 12 May 2026 trim (see
``docs/04_IMPLEMENTATION_PLAN.md`` §3.1):

==================  =====  =======================================================
Model               Tier   Purpose
==================  =====  =======================================================
SeasonalNaive(7)    1      Rule floor (:math:`\\hat y_{t+h} = y_{t+h-7}`).
ARIMAPerRegion      1      Non-DL statistical floor via ``statsmodels.SARIMAX``.
GRUPerRegion        1      Non-physics deep-learning control for the PINN ablation.
==================  =====  =======================================================

Tier 3 wrappers (N-BEATS, TFT, DeepAR) live in separate modules to be added
in the D9 implementation pass and are registered into ``REGISTRY`` from
those modules at import time.

All implementations share the ``BaselineModel`` ABC. Each ``fit(history)``
on a ``pandas.DataFrame`` indexed by ``(region, date)``; ``predict(horizons)``
returns a ``DataFrame`` indexed by ``(region, horizon)`` with columns
``y_hat`` plus optional ``q_lo, q_mid, q_hi`` for probabilistic models.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------

HORIZONS_DEFAULT: tuple[int, ...] = (7, 14, 21, 28)


@dataclass
class ForecastFrame:
    """Container for multi-region multi-horizon forecasts.

    Attributes:
        point: ``DataFrame`` indexed by ``(region, horizon)`` with column
            ``y_hat``.
        quantiles: Optional ``DataFrame`` with the same index and columns
            ``q_lo, q_mid, q_hi``.
    """

    point: pd.DataFrame
    quantiles: pd.DataFrame | None = None


class BaselineModel(ABC):
    """Abstract base class for E1 forecasting baselines."""

    name: str = "base"

    @abstractmethod
    def fit(self, history: pd.DataFrame, target_col: str = "y") -> "BaselineModel":
        """Fit the baseline.

        Args:
            history: Long-format DataFrame with at least columns
                ``region``, ``date``, ``target_col``. ``date`` may be a
                ``DatetimeIndex`` value or a sortable scalar.
            target_col: Name of the target column.

        Returns:
            ``self`` for chaining.
        """

    @abstractmethod
    def predict(self, horizons: Iterable[int] = HORIZONS_DEFAULT) -> ForecastFrame:
        """Predict forecasts at the requested horizons.

        Args:
            horizons: Iterable of horizon offsets (in days).

        Returns:
            ``ForecastFrame``.
        """


# ---------------------------------------------------------------------------
# Tier 1 — required floors
# ---------------------------------------------------------------------------


class SeasonalNaive(BaselineModel):
    """Seasonal-naive forecaster.

    :math:`\\hat y_{t+h} = y_{t+h-S}` where ``S`` is the seasonality (7 days
    by default). Restored 12 May after the author decision to keep one trivial
    floor for sanity-checking the metric harness.
    """

    name = "seasonal_naive"

    def __init__(self, season: int = 7) -> None:
        self.season = season
        self._history: pd.DataFrame | None = None
        self._target_col: str = "y"

    def fit(self, history: pd.DataFrame, target_col: str = "y") -> "SeasonalNaive":
        self._history = history.sort_values(["region", "date"]).copy()
        self._target_col = target_col
        return self

    def predict(self, horizons: Iterable[int] = HORIZONS_DEFAULT) -> ForecastFrame:
        if self._history is None:
            raise RuntimeError("SeasonalNaive: call fit() before predict().")
        records: list[dict] = []
        for region, group in self._history.groupby("region", sort=False):
            sorted_group = group.sort_values("date")
            values = sorted_group[self._target_col].to_numpy()
            if values.size == 0:
                continue
            for h in horizons:
                idx = -((h - 1) % self.season + 1)
                y_hat = float(values[idx])
                records.append({"region": region, "horizon": int(h), "y_hat": y_hat})
        df = pd.DataFrame.from_records(records).set_index(["region", "horizon"])
        return ForecastFrame(point=df)


class ARIMAPerRegion(BaselineModel):
    """Per-region SARIMAX baseline with AIC-selected non-seasonal order.

    Uses ``statsmodels.tsa.statespace.SARIMAX`` (non-seasonal, daily data
    is too noisy on 7-day seasonality alone for a meaningful AIC sweep at
    daily scale). Order ``(p, d, q)`` is selected by AIC over the small
    grid ``p,q ∈ {0,1,2}, d ∈ {0,1}``.

    Args:
        max_p, max_q: Upper bounds for AR and MA orders in the AIC search.
        max_d: Maximum differencing order considered.
        enforce_stationarity: Passed through to ``SARIMAX``.
    """

    name = "arima_per_region"

    def __init__(
        self,
        max_p: int = 2,
        max_q: int = 2,
        max_d: int = 1,
        enforce_stationarity: bool = False,
    ) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.enforce_stationarity = enforce_stationarity
        self._models: dict[str, object] = {}
        self._last_values: dict[str, float] = {}

    def fit(self, history: pd.DataFrame, target_col: str = "y") -> "ARIMAPerRegion":
        self._models.clear()
        self._last_values.clear()
        for region, group in history.groupby("region", sort=False):
            series = group.sort_values("date")[target_col].to_numpy(dtype=float)
            if series.size == 0:
                raise ValueError(
                    f"ARIMAPerRegion: empty series for region {region!r}."
                )
            best_aic = float("inf")
            best_result = None
            for p in range(self.max_p + 1):
                for d in range(self.max_d + 1):
                    for q in range(self.max_q + 1):
                        if p == d == q == 0:
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = SARIMAX(
                                series,
                                order=(p, d, q),
                                enforce_stationarity=self.enforce_stationarity,
                                enforce_invertibility=False,
                            )
                            result = model.fit(disp=False)
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_result = result
            if best_result is None:
                raise RuntimeError(
                    f"ARIMAPerRegion: AIC grid found no valid order for region "
                    f"{region!r}. Series length: {series.size}."
                )
            self._models[region] = best_result
            self._last_values[region] = float(series[-1])
        return self

    def predict(self, horizons: Iterable[int] = HORIZONS_DEFAULT) -> ForecastFrame:
        records: list[dict] = []
        max_h = max(horizons)
        for region, last in self._last_values.items():
            result = self._models.get(region)
            if result is None:
                forecast = np.full(max_h, last)
            else:
                forecast = np.asarray(result.forecast(steps=max_h))
            for h in horizons:
                records.append(
                    {"region": region, "horizon": int(h), "y_hat": float(forecast[h - 1])}
                )
        df = pd.DataFrame.from_records(records).set_index(["region", "horizon"])
        return ForecastFrame(point=df)


@dataclass
class GRUConfig:
    """Hyperparameters for the per-region GRU baseline."""

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lookback: int = 28
    horizons: tuple[int, ...] = HORIZONS_DEFAULT
    lr: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 15


class _SingleRegionGRU(nn.Module):
    """Single-region GRU + multi-horizon linear decoder."""

    def __init__(self, cfg: GRUConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.decoders = nn.ModuleDict(
            {f"h{h}": nn.Linear(cfg.hidden_dim, 1) for h in cfg.horizons}
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return torch.cat([self.decoders[f"h{h}"](last) for h in self.cfg.horizons], dim=-1)


class GRUPerRegion(BaselineModel):
    """Independent GRU per region — the **non-physics control** for the
    PINN ablation (``02_METHODOLOGY.md`` §1.4 + §5.1).

    Univariate sequence-to-vector: takes the last ``lookback`` raw target
    values per region and emits multi-horizon point forecasts. No PINN
    features, no graph coupling — this isolates the physics-coupling
    contribution against the proposed PINN-GRU.
    """

    name = "gru_per_region"

    def __init__(self, config: GRUConfig | None = None) -> None:
        self.config = config or GRUConfig()
        self._models: dict[str, _SingleRegionGRU] = {}
        self._scalers: dict[str, tuple[float, float]] = {}
        self._last_window: dict[str, np.ndarray] = {}

    def _fit_single(self, series: np.ndarray) -> tuple[_SingleRegionGRU, tuple[float, float]]:
        cfg = self.config
        mu = float(series.mean())
        sigma = float(series.std() + 1e-6)
        scaled = (series - mu) / sigma
        L = cfg.lookback
        max_h = max(cfg.horizons)
        if scaled.size <= L + max_h:
            raise ValueError(
                f"GRUPerRegion: series of length {scaled.size} is too short "
                f"for lookback {L} + max horizon {max_h}."
            )
        windows_x: list[np.ndarray] = []
        windows_y: list[np.ndarray] = []
        for t in range(L, scaled.size - max_h):
            windows_x.append(scaled[t - L : t])
            windows_y.append(np.array([scaled[t + h - 1] for h in cfg.horizons]))
        X = torch.tensor(np.stack(windows_x), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.stack(windows_y), dtype=torch.float32)

        model = _SingleRegionGRU(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.HuberLoss(delta=1.0)
        best_loss = float("inf")
        best_state: dict | None = None
        plateau = 0
        for _ in range(cfg.epochs):
            model.train()
            y_hat = model(X)
            loss = criterion(y_hat, Y)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            current = float(loss.detach())
            if current < best_loss - 1e-6:
                best_loss = current
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                plateau = 0
            else:
                plateau += 1
                if plateau >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, (mu, sigma)

    def fit(self, history: pd.DataFrame, target_col: str = "y") -> "GRUPerRegion":
        self._models.clear()
        self._scalers.clear()
        self._last_window.clear()
        L = self.config.lookback
        for region, group in history.groupby("region", sort=False):
            series = group.sort_values("date")[target_col].to_numpy(dtype=float)
            model, (mu, sigma) = self._fit_single(series)
            self._models[region] = model
            self._scalers[region] = (mu, sigma)
            self._last_window[region] = series[-L:]
        return self

    def predict(self, horizons: Iterable[int] = HORIZONS_DEFAULT) -> ForecastFrame:
        records: list[dict] = []
        for region, model in self._models.items():
            mu, sigma = self._scalers[region]
            last = self._last_window[region]
            scaled = (last - mu) / sigma
            x = torch.tensor(scaled, dtype=torch.float32).reshape(1, -1, 1)
            model.eval()
            with torch.no_grad():
                pred = model(x).numpy().squeeze(0)
            cfg_horizons = list(self.config.horizons)
            for h in horizons:
                if h not in cfg_horizons:
                    raise ValueError(
                        f"GRUPerRegion: horizon {h} not in trained horizons "
                        f"{cfg_horizons}. Retrain with the requested horizons."
                    )
                idx = cfg_horizons.index(h)
                y_scaled = float(pred[idx])
                records.append(
                    {"region": region, "horizon": int(h), "y_hat": y_scaled * sigma + mu}
                )
        df = pd.DataFrame.from_records(records).set_index(["region", "horizon"])
        return ForecastFrame(point=df)


# ---------------------------------------------------------------------------
# Registry & convenience
# ---------------------------------------------------------------------------
#
# Tier 3 wrappers (N-BEATS via ``neuralforecast``, TFT and DeepAR via
# ``pytorch-forecasting``) are implemented in separate modules to be added
# in the D9 implementation pass. They are registered here when their
# wrapper modules import successfully.

REGISTRY: dict[str, type[BaselineModel]] = {
    "seasonal_naive": SeasonalNaive,
    "arima_per_region": ARIMAPerRegion,
    "gru_per_region": GRUPerRegion,
}


def build_baseline(name: str, **kwargs) -> BaselineModel:
    """Factory: instantiate a baseline by its registry key.

    Args:
        name: Key into ``REGISTRY``.
        **kwargs: Forwarded to the baseline constructor.

    Returns:
        A configured ``BaselineModel`` instance.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)

