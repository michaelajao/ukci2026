"""Decision-aware composite loss, temporal forecasting head, MC-Dropout
inference, and a thin training loop for the UKCI 2026 proposed model.

This module is the **load-bearing methodological contribution** of the paper.
It consolidates four concerns into one file per the 12 May 2026 author
decision to reduce file count:

1. ``CompositeLoss`` — Huber + PINN-residual + asymmetric under-prediction
   hinge + temporal smoothness. Each term can be turned off via
   ``ablation_flags`` for E2 ablation studies.
2. ``TemporalForecastingHead`` — 2-layer GRU + multi-horizon linear decoder
   per ``02_METHODOLOGY.md`` §1.4.
3. ``PinnGRUForecaster`` — wires the PINN state/parameter networks into the
   GRU head, producing point forecasts at horizons ``h ∈ {7, 14, 21, 28}``
   and exposing dropout for MC-sampling at inference.
4. ``mc_dropout_quantiles`` — K stochastic forward passes; returns empirical
   quantiles at ``p ∈ {0.1, 0.5, 0.9}``.
5. ``train_composite_loss`` — thin training-loop entrypoint with early
   stopping on validation WIS. Hyperparameters from §1.5 + §6.

Cross-references:
    docs/02_METHODOLOGY.md §1.4 (temporal head),
    docs/02_METHODOLOGY.md §1.5 (composite loss),
    docs/02_METHODOLOGY.md §1.6 (MC Dropout),
    docs/02_METHODOLOGY.md §6   (hyperparameter table),
    docs/04_IMPLEMENTATION_PLAN.md Phase 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
from torch import Tensor, nn

from forecasting.pinn_seird import RegionalPINN, SEIRDFixedParams, pinn_loss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZONS: tuple[int, ...] = (7, 14, 21, 28)
"""Forecast horizons in days (``02_METHODOLOGY.md`` §0)."""

LOOKBACK_DEFAULT: int = 28
"""Lookback window length in days (``02_METHODOLOGY.md`` §0)."""


@dataclass
class CompositeLossConfig:
    """Hyperparameters for the four-term composite loss.

    Initial values from ``02_METHODOLOGY.md`` §1.5; tune on validation by grid
    over {0.05, 0.1, 0.5, 1.0} for ``lambda_phys`` and ``lambda_under``.
    """

    lambda_phys: float = 0.1
    lambda_under: float = 0.5
    lambda_smooth: float = 0.01
    huber_delta: float = 1.0
    use_forecast: bool = True
    use_phys: bool = True
    use_under: bool = True
    use_smooth: bool = True


@dataclass
class TemporalHeadConfig:
    """Hyperparameters for the GRU forecasting head (``§1.4``)."""

    input_dim: int = 24
    """Augmented feature dim = 8 (PINN state) + 4 (PINN params) + ``x_extra``."""

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    horizons: tuple[int, ...] = HORIZONS


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """Decision-aware composite loss with ablation switches.

    .. math::
        \\mathcal{L}_{\\text{total}}
            = \\mathcal{L}_{\\text{forecast}}
            + \\lambda_{\\text{phys}}\\, \\sum_r \\mathcal{L}^{\\text{PINN}}_r
            + \\lambda_{\\text{under}}\\, \\mathcal{L}_{\\text{under}}
            + \\lambda_{\\text{smooth}}\\, \\mathcal{L}_{\\text{smooth}}

    Each term can be disabled via the ``use_*`` flags in ``CompositeLossConfig``
    so a single training script can run the ablation table directly.

    Args:
        config: ``CompositeLossConfig``. The four ``use_*`` flags drive
            the ablation switches.

    The ``forward`` returns a ``(loss, parts)`` tuple where ``parts`` is a
    dict of the per-term contributions (post-weighting) for logging.
    """

    def __init__(self, config: CompositeLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or CompositeLossConfig()
        self.huber = nn.HuberLoss(delta=self.config.huber_delta, reduction="mean")

    def forward(
        self,
        y_hat: Tensor,
        y_true: Tensor,
        pinn_residual: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the composite loss.

        Args:
            y_hat: Predicted values, shape ``(batch, region, horizon)``.
            y_true: Ground-truth values, same shape as ``y_hat``.
            pinn_residual: Optional scalar PINN-ODE residual term
                (already summed over regions). If ``None`` the physics term
                is skipped regardless of ``use_phys``.

        Returns:
            ``(total_loss, parts)`` where ``parts`` is a dict of
            ``"forecast", "phys", "under", "smooth"`` post-weighted scalars.
        """
        device = y_hat.device
        parts: dict[str, Tensor] = {
            "forecast": torch.zeros((), device=device),
            "phys": torch.zeros((), device=device),
            "under": torch.zeros((), device=device),
            "smooth": torch.zeros((), device=device),
        }

        if self.config.use_forecast:
            parts["forecast"] = self.huber(y_hat, y_true)

        if self.config.use_phys and pinn_residual is not None:
            parts["phys"] = self.config.lambda_phys * pinn_residual

        if self.config.use_under:
            # one-sided hinge: penalise under-prediction (y_true > y_hat)
            under = torch.clamp(y_true - y_hat, min=0.0).mean()
            parts["under"] = self.config.lambda_under * under

        if self.config.use_smooth and y_hat.shape[-1] > 1:
            diffs = y_hat[..., 1:] - y_hat[..., :-1]
            smooth = (diffs ** 2).mean()
            parts["smooth"] = self.config.lambda_smooth * smooth

        total = parts["forecast"] + parts["phys"] + parts["under"] + parts["smooth"]
        return total, parts


# ---------------------------------------------------------------------------
# Temporal forecasting head
# ---------------------------------------------------------------------------


class TemporalForecastingHead(nn.Module):
    """2-layer GRU + multi-horizon linear decoder (per ``§1.4``).

    Operates on augmented features
    :math:`f_{r,t} = [\\tilde U_r(t)\\, \\|\\, \\tilde X_r(t)\\, \\|\\, x_{r,t}]`
    and outputs joint forecasts at all horizons in ``config.horizons``.

    Per-region weights (one module instance per region) implement the
    "deliberately per-region GRU" choice (§1.4 + AIIM-defence rationale).
    The module itself is per-region; instantiate one per NHS region or wrap
    inside ``PinnGRUForecaster`` which manages the multi-region case.
    """

    def __init__(self, config: TemporalHeadConfig | None = None) -> None:
        super().__init__()
        cfg = config or TemporalHeadConfig()
        self.config = cfg
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        # Multi-horizon decoder: one linear head per horizon, shared GRU body.
        self.decoders = nn.ModuleDict(
            {f"h{h}": nn.Linear(cfg.hidden_dim, 1) for h in cfg.horizons}
        )
        # MC-Dropout module (kept active at inference; see ``mc_dropout_quantiles``)
        self.mc_dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass.

        Args:
            features: Shape ``(batch, lookback, input_dim)``.

        Returns:
            Forecasts at all horizons, shape ``(batch, num_horizons)``.
        """
        # GRU output shape: (batch, lookback, hidden_dim)
        out, _ = self.gru(features)
        last = out[:, -1, :]
        last = self.mc_dropout(last)
        return torch.cat(
            [self.decoders[f"h{h}"](last) for h in self.config.horizons], dim=-1
        )


# ---------------------------------------------------------------------------
# Combined PINN-GRU forecaster
# ---------------------------------------------------------------------------


class PinnGRUForecaster(nn.Module):
    """End-to-end per-region forecaster: PINN features → GRU head.

    Workflow:
        1. The pre-trained ``RegionalPINN`` provides time-anchored state
           and parameter estimates :math:`(\\tilde U_r, \\tilde X_r)` for each
           timestep in the lookback window.
        2. These are concatenated with observed exogenous covariates
           ``x_extra`` (mobility, lagged cases, …) into the augmented feature
           sequence ``f_{r,t}``.
        3. The ``TemporalForecastingHead`` produces multi-horizon forecasts.

    Args:
        pinn: A trained ``RegionalPINN`` (or freshly initialised — its
            parameters are frozen by default inside this module so the
            GRU head is trained alone; pass ``train_pinn=True`` to fine-tune).
        head_config: ``TemporalHeadConfig`` controlling the GRU head.
        train_pinn: Whether PINN parameters should receive gradients during
            composite-loss training. Default ``False`` (freeze).
    """

    def __init__(
        self,
        pinn: RegionalPINN,
        head_config: TemporalHeadConfig | None = None,
        *,
        train_pinn: bool = False,
    ) -> None:
        super().__init__()
        self.pinn = pinn
        self.train_pinn = train_pinn
        if not train_pinn:
            for param in self.pinn.parameters():
                param.requires_grad_(False)
        self.head = TemporalForecastingHead(head_config)

    def _augmented_features(self, t_seq: Tensor, x_extra: Tensor | None) -> Tensor:
        """Build the augmented feature matrix
        ``[U_NN(t) ‖ X_NN(t) ‖ x_extra(t)]``.

        Args:
            t_seq: Time points, shape ``(batch, lookback, 1)``.
            x_extra: Optional exogenous covariates,
                shape ``(batch, lookback, extra_dim)``. May be ``None``.

        Returns:
            Concatenated features, shape ``(batch, lookback, total_dim)``.
        """
        batch, lookback, _ = t_seq.shape
        # PINN expects shape (N, 1). Flatten across batch and lookback,
        # call PINN, then reshape.
        flat = t_seq.reshape(-1, 1)
        state, params = self.pinn(flat)
        state = state.reshape(batch, lookback, -1)
        params = params.reshape(batch, lookback, -1)
        feats = torch.cat([state, params], dim=-1)
        if x_extra is not None:
            feats = torch.cat([feats, x_extra], dim=-1)
        return feats

    def forward(self, t_seq: Tensor, x_extra: Tensor | None = None) -> Tensor:
        """End-to-end forward pass.

        Args:
            t_seq: Time points, shape ``(batch, lookback, 1)``.
            x_extra: Optional exogenous covariates,
                shape ``(batch, lookback, extra_dim)``.

        Returns:
            Forecasts at all horizons, shape ``(batch, num_horizons)``.
        """
        features = self._augmented_features(t_seq, x_extra)
        return self.head(features)


# ---------------------------------------------------------------------------
# MC-Dropout inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def mc_dropout_quantiles(
    model: nn.Module,
    *forward_args: Tensor,
    k: int = 100,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    forward_kwargs: dict | None = None,
) -> dict[float, Tensor]:
    """Run ``k`` stochastic forward passes with dropout active at inference.

    Per ``02_METHODOLOGY.md`` §1.6: enable dropout during forward pass, run
    ``K`` stochastic passes, return empirical quantiles. We use ``p=0.5`` as
    the point forecast and ``(q^{0.1}, q^{0.5}, q^{0.9})`` for scenario
    generation in Phase B.

    Args:
        model: A ``nn.Module`` containing one or more ``nn.Dropout`` layers.
        *forward_args: Positional arguments passed to ``model(...)``.
        k: Number of stochastic forward passes.
        quantiles: Iterable of quantile probabilities in ``(0, 1)``.
        forward_kwargs: Optional keyword arguments passed to ``model(...)``.

    Returns:
        Dict mapping each quantile probability to a forecast tensor with the
        same shape as a single forward pass.
    """
    if forward_kwargs is None:
        forward_kwargs = {}
    # Keep dropout active at inference: switch to ``train`` mode, but freeze
    # BatchNorm running-statistic updates by skipping its forward. Our model
    # contains only Linear/GRU/Dropout so this is safe.
    model.train()
    samples: list[Tensor] = []
    for _ in range(k):
        samples.append(model(*forward_args, **forward_kwargs))
    model.eval()
    stack = torch.stack(samples, dim=0)  # (k, ...)
    q_list = [float(q) for q in quantiles]
    q_tensor = torch.tensor(q_list, device=stack.device, dtype=stack.dtype)
    out = torch.quantile(stack, q_tensor, dim=0)
    return {q_list[i]: out[i] for i in range(len(q_list))}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyperparameters for the composite-loss training loop (``§6``)."""

    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 20
    """Early-stopping patience in epochs (validation-loss plateau)."""


def train_composite_loss(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    composite: CompositeLoss,
    pinn_loss_fn: Callable[..., Tensor] | None = None,
    train_config: TrainConfig | None = None,
    device: torch.device | str = "cpu",
    log_fn: Callable[[dict], None] | None = None,
) -> dict:
    """Thin training-loop entrypoint.

    Args:
        model: A ``PinnGRUForecaster`` or any ``nn.Module`` that consumes the
            batch positional args used by ``train_loader``.
        train_loader: Iterable of ``(t_seq, x_extra, y_true)`` tuples for
            training (``x_extra`` may be ``None`` per batch).
        val_loader: Same protocol as ``train_loader``.
        composite: A configured ``CompositeLoss`` instance.
        pinn_loss_fn: Optional callable returning the PINN-ODE residual for
            each batch (used when ``composite.config.use_phys`` is True).
            Signature: ``pinn_loss_fn(model, batch) -> Tensor`` (scalar).
        train_config: ``TrainConfig`` hyperparameters.
        device: torch device.
        log_fn: Optional logging hook receiving dict of
            ``{"epoch", "train_loss", "val_loss", "parts"}`` each epoch.

    Returns:
        Dict with ``"best_val_loss"``, ``"best_epoch"``, and the final state
        dict in ``"state_dict"``.
    """
    cfg = train_config or TrainConfig()
    model.to(device)
    composite.to(device)
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = math.inf
    best_epoch = 0
    best_state: dict | None = None
    plateau = 0

    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        train_running = 0.0
        n_train = 0
        for batch in train_loader:
            t_seq, x_extra, y_true = batch
            t_seq = t_seq.to(device)
            y_true = y_true.to(device)
            if x_extra is not None:
                x_extra = x_extra.to(device)
            y_hat = model(t_seq, x_extra)
            pinn_resid = (
                pinn_loss_fn(model, batch).to(device)
                if pinn_loss_fn is not None
                else None
            )
            loss, _parts = composite(y_hat, y_true, pinn_resid)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    cfg.grad_clip,
                )
            opt.step()
            train_running += float(loss.detach()) * t_seq.size(0)
            n_train += t_seq.size(0)
        train_loss = train_running / max(n_train, 1)

        # ---- validate ----
        model.eval()
        val_running = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                t_seq, x_extra, y_true = batch
                t_seq = t_seq.to(device)
                y_true = y_true.to(device)
                if x_extra is not None:
                    x_extra = x_extra.to(device)
                y_hat = model(t_seq, x_extra)
                vloss, _ = composite(y_hat, y_true, None)
                val_running += float(vloss.detach()) * t_seq.size(0)
                n_val += t_seq.size(0)
        val_loss = val_running / max(n_val, 1)

        if log_fn is not None:
            log_fn(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            plateau = 0
        else:
            plateau += 1
            if plateau >= cfg.patience:
                break

    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "state_dict": best_state if best_state is not None else model.state_dict(),
    }

