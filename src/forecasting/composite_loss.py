"""Direct multi-quantile PinnGRU forecaster (UKCI 2026 proposed model).

This module is the load-bearing methodological contribution of the paper:

1. ``QuantileForecastingHead`` — 2-layer GRU + per-horizon MLP decoder with a
   level + damped-trend anchor skip connection. Outputs ``Q`` quantiles per
   horizon directly (no MC-Dropout sampling at inference).
2. ``PinnGRUQuantileForecaster`` — wires the per-region PINN state/parameter
   networks into the quantile head, producing forecasts at horizons
   ``h ∈ {7, 14, 21, 28}`` and quantiles ``q ∈ {0.1, 0.5, 0.9}``.
3. ``pinball_loss_multiq`` — strictly proper multi-quantile pinball loss
   (Koenker & Bassett 1978; Gneiting & Raftery 2007). The asymmetry at
   ``q = 0.9`` carries the decision-aware "avoid under-prediction" mass.

Cross-references:
    docs/02_METHODOLOGY.md §1.4 (temporal head),
    docs/02_METHODOLOGY.md §1.5 (multi-quantile pinball loss),
    docs/02_METHODOLOGY.md §1.6 (level + trend anchor),
    docs/02_METHODOLOGY.md §6   (hyperparameter table).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from forecasting.pinn_seird import RegionalPINN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZONS: tuple[int, ...] = (7, 14, 21, 28)
"""Forecast horizons in days (``02_METHODOLOGY.md`` §0)."""

LOOKBACK_DEFAULT: int = 28
"""Lookback window length in days (``02_METHODOLOGY.md`` §0)."""


# ---------------------------------------------------------------------------
# Direct multi-quantile head
# ---------------------------------------------------------------------------
#
# Each horizon prediction is ``mlp_h(last_hidden) + alpha_h * y_last_observed
# + beta_h * slope * (h / trend_lag) ** phi_h^q`` so the GRU only learns the
# *delta* from current level. The quantile-specific damping ``phi_h^q``
# carries the decision-aware "avoid under-prediction" mass at q = 0.9.


@dataclass
class QuantileHeadConfig:
    """Hyperparameters for the direct multi-quantile head."""

    input_dim: int = 24
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    horizons: tuple[int, ...] = HORIZONS
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    decoder_hidden: int = 64
    """Width of the per-horizon MLP decoder."""

    target_index_in_extra: int = 0
    """Index of the target (z-scored mv_beds) inside ``x_extra``; used by the
    level + trend anchors. Must match how the caller packs covariates."""

    trend_lag: int = 7
    """Lookback lag for the autoregressive trend anchor. The slope is
    ``y_{t-1} - y_{t-1-trend_lag}`` and is extrapolated by ``(h / trend_lag)``
    for each horizon. Default 7 days = one weekly cycle."""


class QuantileForecastingHead(nn.Module):
    """GRU + per-horizon MLP decoder + level + trend anchor + direct quantile.

    Each horizon-quantile prediction is the sum of three terms:

    .. math::
        \\hat{y}_{h,q} = \\Delta_{h,q}(\\text{hidden})
            + \\alpha_{h,q}\\, y_{t-1}
            + \\gamma_{h,q}\\, \\text{slope}\\, (h / \\text{trend\\_lag})

    where :math:`\\Delta_{h,q}` is the per-horizon MLP, :math:`\\alpha` is the
    level anchor (autoregressive on the last observation), and :math:`\\gamma`
    is the trend anchor (autoregressive on the recent 7-day slope, linearly
    extrapolated by :math:`h/7` weeks). Both anchors are learned per
    ``(horizon, quantile)`` so each output channel can use whatever blend of
    persistence, trend, and learned signal is best for that operating point.

    Forward signature: ``(features, y_last, slope) -> (B, K, Q)``.
    """

    def __init__(self, config: QuantileHeadConfig | None = None) -> None:
        super().__init__()
        cfg = config or QuantileHeadConfig()
        self.config = cfg
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.input_dropout = nn.Dropout(p=cfg.dropout)
        Q = len(cfg.quantiles)
        K = len(cfg.horizons)
        self.decoders = nn.ModuleDict({
            f"h{h}": nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.decoder_hidden),
                nn.GELU(),
                nn.Dropout(p=cfg.dropout),
                nn.Linear(cfg.decoder_hidden, Q),
            )
            for h in cfg.horizons
        })
        # Level anchor in [0, 1] via sigmoid; weak prior (sigmoid(-1.4) ≈ 0.2).
        self.level_anchor = nn.Parameter(torch.full((K, Q), -1.4))
        # Trend anchor in [0, 1] via sigmoid; weak prior. Multiplied by a
        # damped horizon factor so longer horizons get less extrapolation
        # than the naive linear ``(h/trend_lag)`` schedule, which overshoots
        # at h=21/h=28 (see v3 evaluation: h=28 MAE 33.2 vs h=14 11.4).
        self.trend_anchor = nn.Parameter(torch.full((K, Q), -1.4))
        # Learnable per-quantile trend-damping exponent in [0, 1]:
        # horizon factor = (h/trend_lag)^φ_q. φ=1 is linear, φ=0.5 is √,
        # φ→0 is constant (no horizon dependence).
        # Per-quantile prior: q50 starts near linear (φ≈0.88) for sharp
        # point-accuracy tracking; q10/q90 start with √-damping (φ≈0.5)
        # for tighter, better-calibrated bounds. The model can still learn
        # to deviate from this prior during training.
        damping_init = []
        for q in cfg.quantiles:
            damping_init.append(2.0 if abs(q - 0.5) < 1e-6 else 0.0)
        self.trend_damping_logit = nn.Parameter(torch.tensor(damping_init))
        # Buffer of base (h / trend_lag) factors.
        self.register_buffer(
            "horizon_steps",
            torch.tensor(
                [h / cfg.trend_lag for h in cfg.horizons], dtype=torch.float32
            ).view(K, 1),
        )

    def forward(self, features: Tensor, y_last: Tensor, slope: Tensor) -> Tensor:
        out, _ = self.gru(features)
        last = self.input_dropout(out[:, -1, :])
        deltas = torch.stack(
            [self.decoders[f"h{h}"](last) for h in self.config.horizons],
            dim=1,
        )  # (B, K, Q)
        alpha = torch.sigmoid(self.level_anchor).unsqueeze(0)         # (1, K, Q)
        gamma = torch.sigmoid(self.trend_anchor).unsqueeze(0)         # (1, K, Q)
        # Damped horizon schedule: (h/lag)^φ_q broadcast to (K, Q).
        phi = torch.sigmoid(self.trend_damping_logit).view(1, -1)     # (1, Q)
        damped_steps = self.horizon_steps ** phi                       # (K, Q)
        damped_steps = damped_steps.unsqueeze(0)                       # (1, K, Q)
        level = y_last.view(-1, 1, 1)                                 # (B, 1, 1)
        trend = slope.view(-1, 1, 1) * damped_steps                   # (B, K, Q)
        return deltas + alpha * level + gamma * trend


def pinball_loss_multiq(
    y_hat: Tensor, y_true: Tensor, quantiles: tuple[float, ...]
) -> Tensor:
    """Sum of pinball (quantile) losses across a set of quantiles.

    ``y_hat`` shape ``(B, K, Q)``, ``y_true`` shape ``(B, K)``. Returns a
    scalar loss (mean across batch × horizon, summed across quantiles so q90
    receives equal weight to q10 — which is what we want: getting the upper
    bound right matters as much as getting the median right).
    """
    device = y_hat.device
    qs = torch.tensor(quantiles, device=device).view(1, 1, -1)
    diff = y_true.unsqueeze(-1) - y_hat  # (B, K, Q)
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff)
    return loss.mean(dim=(0, 1)).sum()  # mean over B,K then sum over quantiles


class PinnGRUQuantileForecaster(nn.Module):
    """End-to-end per-region quantile forecaster.

    Wraps a (pre-trained, optionally frozen) ``RegionalPINN`` and the new
    ``QuantileForecastingHead`` with level-anchor skip connection.
    """

    def __init__(
        self,
        pinn: RegionalPINN,
        head_config: QuantileHeadConfig | None = None,
        *,
        train_pinn: bool = False,
    ) -> None:
        super().__init__()
        self.pinn = pinn
        self.train_pinn = train_pinn
        if not train_pinn:
            for param in self.pinn.parameters():
                param.requires_grad_(False)
        self.head = QuantileForecastingHead(head_config)
        self._tgt_idx = self.head.config.target_index_in_extra
        self._trend_lag = self.head.config.trend_lag

    def _augmented_features(self, t_seq: Tensor, x_extra: Tensor) -> Tensor:
        batch, lookback, _ = t_seq.shape
        flat = t_seq.reshape(-1, 1)
        state, params = self.pinn(flat)
        state = state.reshape(batch, lookback, -1)
        params = params.reshape(batch, lookback, -1)
        return torch.cat([state, params, x_extra], dim=-1)

    def forward(self, t_seq: Tensor, x_extra: Tensor) -> Tensor:
        feats = self._augmented_features(t_seq, x_extra)
        # Last observed z-scored target and its 7-day slope used by the head.
        y_last = x_extra[:, -1, self._tgt_idx]
        slope = (
            x_extra[:, -1, self._tgt_idx]
            - x_extra[:, -1 - self._trend_lag, self._tgt_idx]
        )
        return self.head(feats, y_last, slope)


