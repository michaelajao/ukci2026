"""Train forecasters and run rolling-origin Omicron evaluation.

Trains four forecasters and produces multi-horizon point + interval forecasts
at fixed rolling-origin dates within the Omicron test period (2021-12-01 to
2022-08-31):

  - SeasonalNaive(7)        rule floor
  - ARIMAPerRegion          per-region SARIMAX, AIC-selected order, refit per origin
  - GRUPerRegion            univariate GRU baseline, fit on train+val, slide on test
  - PinnGRU                 PINN feature extractor + GRU head trained with the
                            decision-aware composite loss (per-region), MC Dropout
                            quantiles at inference

Outputs:
  results/forecasting/forecasts.parquet   long-form: model, origin, region, horizon,
                                          y_hat, q_lo, q_hi (NaN for point models),
                                          y_true
  results/forecasting/table_metrics.csv   metrics per (model, horizon) and overall

Rolling-origin protocol (matches 02_METHODOLOGY.md Sec 4.2):
  - Origins every ORIGIN_STRIDE=14 days from test_start + LOOKBACK to
    test_end - max(HORIZONS).
  - For each origin t the model receives history up to and including date t-1.
  - It predicts y(t + h - 1) for h in HORIZONS = {7, 14, 21, 28}.
"""

from __future__ import annotations

import sys
import time

from utils import repo_root, results_dir, set_windows_openmp_env

# Allow torch's bundled libiomp5md.dll to coexist with the one statsmodels/MKL
# loads via numpy on Windows; without this, SARIMAX aborts with OMP error #15.
set_windows_openmp_env()

import numpy as np
import pandas as pd
import torch

ROOT = repo_root()

from data.loader import DEFAULT_CSV, DEFAULT_SPLIT_DATES, load_regional_csv
from forecasting.baselines import (
    ARIMAPerRegion,
    GRUPerRegion,
    SeasonalNaive,
    XGBoostPerRegion,
)
from forecasting.composite_loss import (
    PinnGRUQuantileForecaster,
    QuantileHeadConfig,
    pinball_loss_multiq,
)
from forecasting.pinn_seird import RegionalPINN, SEIRDFixedParams, pinn_loss

# ---------------------------------------------------------------------------
# Hyperparameters / paths
# ---------------------------------------------------------------------------

HORIZONS: tuple[int, ...] = (7, 14, 21, 28)
LOOKBACK = 28
ORIGIN_STRIDE = 7
TARGET = "mv_beds"
# PinnGRU covariates: mv_beds only (univariate autoregressive).
# admissions/hospital_cases are excluded: Omicron had very high case counts
# with LOW severity, creating covariate shift — those z-scores (calibrated on
# Alpha+Delta training) are ~+0.5 in the test period, driving the model to
# predict high MV beds even though Omicron was mild.  Occupied_beds is
# excluded for the same reason.  The PINN time-embedding features and the
# autoregressive mv_beds signal are sufficient for the paper's contribution
# (decision-aware composite loss ablation).
COVARS: tuple[str, ...] = ("mv_beds",)
SEED = 0
MC_K = 100

OUT_DIR = results_dir("forecasting")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABLATIONS = [
    {"tag": "no_pretrain", "label": "No PINN pretrain (random PINN)", "kwargs": {"skip_pretrain": True}},
    {"tag": "no_level", "label": "No level anchor (alpha = 0)", "kwargs": {"disable_level_anchor": True}},
    {"tag": "no_trend", "label": "No trend anchor (gamma = 0)", "kwargs": {"disable_trend_anchor": True}},
    {"tag": "no_params", "label": "No PINN parameter features (state-only)", "kwargs": {"mask_param_features": True}},
    {
        "tag": "no_decision_aware",
        "label": "No decision-aware loss (MSE on q50 only)",
        "kwargs": {"use_mse_loss": True},
    },
]


# ---------------------------------------------------------------------------
# Rolling-origin schedule
# ---------------------------------------------------------------------------


def build_origins(
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    lookback: int,
    max_h: int,
    stride: int,
) -> pd.DatetimeIndex:
    """Origins for rolling-origin evaluation: first origin requires `lookback`
    days of history within the test period; last origin must leave `max_h`
    days for the longest horizon to be observed.
    """
    first = test_start + pd.Timedelta(days=lookback)
    last = test_end - pd.Timedelta(days=max_h - 1)
    return pd.date_range(first, last, freq=f"{stride}D")


# ---------------------------------------------------------------------------
# Baselines that refit per origin (SeasonalNaive, ARIMA)
# ---------------------------------------------------------------------------


def fit_predict_baseline_per_origin(
    df: pd.DataFrame,
    origins: pd.DatetimeIndex,
    model_factory,
    name: str,
) -> pd.DataFrame:
    """For each origin, fit a fresh model on all (region, date) rows with
    `date < origin` and predict the four horizons.
    """
    records: list[dict] = []
    for origin in origins:
        hist = df[df["date"] < origin][["region_name", "date", TARGET]].rename(
            columns={"region_name": "region", TARGET: "y"}
        )
        model = model_factory()
        model.fit(hist)
        fc = model.predict(HORIZONS).point
        for (region, h), row in fc.iterrows():
            target_date = origin + pd.Timedelta(days=int(h) - 1)
            y_true_rows = df[(df["region_name"] == region) & (df["date"] == target_date)]
            if y_true_rows.empty:
                continue
            records.append({
                "model": name,
                "origin": origin,
                "region": region,
                "horizon": int(h),
                "y_hat": float(row["y_hat"]),
                "q_lo": np.nan,
                "q_hi": np.nan,
                "y_true": float(y_true_rows[TARGET].iloc[0]),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# GRU baseline: fit once on train+val, slide on test
# ---------------------------------------------------------------------------


def fit_gru_per_region(df_train_val: pd.DataFrame) -> GRUPerRegion:
    model = GRUPerRegion()
    hist = df_train_val[["region_name", "date", TARGET]].rename(
        columns={"region_name": "region", TARGET: "y"}
    )
    model.fit(hist)
    return model


def predict_gru_at_origins(
    model: GRUPerRegion,
    df: pd.DataFrame,
    origins: pd.DatetimeIndex,
) -> pd.DataFrame:
    L = model.config.lookback
    records: list[dict] = []
    for origin in origins:
        for region in model._models:
            sub = df[df["region_name"] == region].sort_values("date")
            window = sub[sub["date"] < origin].tail(L)[TARGET].to_numpy(dtype=float)
            if len(window) < L:
                continue
            mu, sigma = model._scalers[region]
            scaled = (window - mu) / sigma
            x = torch.tensor(scaled, dtype=torch.float32).reshape(1, -1, 1)
            net = model._models[region]
            net.eval()
            with torch.no_grad():
                pred = net(x).cpu().numpy().squeeze(0)
            cfg_h = list(model.config.horizons)
            for h in HORIZONS:
                idx = cfg_h.index(h)
                y_hat = float(pred[idx]) * sigma + mu
                target_date = origin + pd.Timedelta(days=int(h) - 1)
                y_rows = sub[sub["date"] == target_date]
                if y_rows.empty:
                    continue
                records.append({
                    "model": model.name,
                    "origin": origin,
                    "region": region,
                    "horizon": int(h),
                    "y_hat": y_hat,
                    "q_lo": np.nan,
                    "q_hi": np.nan,
                    "y_true": float(y_rows[TARGET].iloc[0]),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# PinnGRU: per-region training + MC dropout inference
# ---------------------------------------------------------------------------


def pretrain_pinn_region(
    pinn: RegionalPINN,
    t_norm: np.ndarray,
    h_obs: np.ndarray,
    c_obs: np.ndarray,
    population: float,
    epochs: int = 1500,
    lr: float = 5e-3,
    lambda_ode: float = 0.1,
    n_collocation: int = 256,
) -> dict:
    """Pre-train a per-region PINN on observable compartments before GRU coupling.

    Fits StateNet + ParameterNet so that the network's H and C compartments
    track observed hospital_cases / mv_beds, AND its derivatives satisfy the
    SEIRD ODE system. This is what makes the augmented features non-trivial:
    without this step the PINN outputs collapse to a near-constant ~0.5 across
    all t (sigmoid initialisation), polluting the GRU input with 12 channels
    of zero-variance noise.

    Args:
        pinn: the RegionalPINN to train in place.
        t_norm: (T,) normalised time index in [0, 1] for every observation day.
        h_obs:  (T,) observed hospital occupancy on the original scale (people).
        c_obs:  (T,) observed mechanical-ventilation occupancy (people).
        population: regional population used to normalise observations into
            the PINN's [0, 1] compartment scale.
        epochs: number of full-batch optimisation steps.
        lr: Adam learning rate. PINN needs a larger lr than the GRU head.
        lambda_ode: weight on the SEIRD ODE-residual term.
        n_collocation: number of random collocation points sampled each step
            for the ODE residual.

    Returns:
        dict with ``"data"``, ``"ode"``, ``"total"`` final loss components.
    """
    fixed = SEIRDFixedParams()
    t_data = torch.tensor(t_norm.astype(np.float32), device=DEVICE)
    y_data = {
        "H": torch.tensor((h_obs / population).astype(np.float32), device=DEVICE),
        "C": torch.tensor((c_obs / population).astype(np.float32), device=DEVICE),
    }
    opt = torch.optim.Adam(pinn.parameters(), lr=lr)
    parts = {"data": 0.0, "ode": 0.0, "total": 0.0}
    for _ in range(epochs):
        t_coll = torch.rand(n_collocation, device=DEVICE)
        loss, parts = pinn_loss(pinn, t_data, y_data, t_coll, fixed, lambda_ode)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        opt.step()
    return parts


QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def train_pinn_gru_region(
    series_y: np.ndarray,
    series_x: np.ndarray,
    series_h: np.ndarray,
    t_norm: np.ndarray,
    is_val: np.ndarray,
    population: float,
    epochs: int = 800,
    patience: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pinn_pretrain_epochs: int = 1500,
    skip_pretrain: bool = False,
    disable_level_anchor: bool = False,
    disable_trend_anchor: bool = False,
    mask_param_features: bool = False,
    use_mse_loss: bool = False,
) -> tuple[PinnGRUQuantileForecaster, dict]:
    """Fit one ``PinnGRUQuantileForecaster`` on one region's series.

    Pipeline:
        1. Pre-train the PINN on observed H (hospital_cases) and C (mv_beds)
           with SEIRD ODE-residual regularisation on the **full** train+val
           span. The PINN is just a non-parametric time encoder here, so it
           sees all available data.
        2. Freeze PINN. Build sliding windows; assign each window to *train*
           or *val* by whether its target date falls in the Delta validation
           split. Train the GRU head with multi-quantile pinball loss, early-
           stop on validation pinball.

    Output head produces ``(q10, q50, q90)`` per horizon directly via the
    ``QuantileForecastingHead``. The decision-aware emphasis comes from
    pinball-q90 being part of the loss — no separate asymmetric hinge.

    Args:
        series_y: (T,) target values (mv_beds) on the original scale.
        series_x: (T, F_cov) covariate values on the original scale.
        series_h: (T,) hospital_cases values for PINN pre-training.
        t_norm:   (T,) absolute-day-normalised time index in [0, 1].
        is_val:   (T,) boolean per day: True if the day is in the Delta
            validation period. Window-level split is derived from the
            target date of each window.
        population: regional population for the PINN's state denormalisation.

    Returns:
        ``(model, scalers)``. ``scalers`` exposes the standardisation stats
        and the best validation pinball value reached.
    """
    mu_y = float(series_y.mean())
    sigma_y = float(series_y.std() + 1e-6)
    mu_x = series_x.mean(axis=0)
    sigma_x = series_x.std(axis=0) + 1e-6
    y_scaled = ((series_y - mu_y) / sigma_y).astype(np.float32)
    x_scaled = ((series_x - mu_x) / sigma_x).astype(np.float32)

    L = LOOKBACK
    max_h = max(HORIZONS)
    n = len(y_scaled)
    if n <= L + max_h:
        raise ValueError(f"Series too short: {n} <= L+max_h={L+max_h}")

    wins_t: list[np.ndarray] = []
    wins_x: list[np.ndarray] = []
    wins_y: list[list[float]] = []
    win_is_val: list[bool] = []
    for t0 in range(0, n - L - max_h + 1):
        wins_t.append(t_norm[t0 : t0 + L].astype(np.float32)[:, None])
        wins_x.append(x_scaled[t0 : t0 + L])
        wins_y.append([float(y_scaled[t0 + L + h - 1]) for h in HORIZONS])
        # A window is validation if its **earliest** target falls in val.
        # (Using max_h target would leak training data into val; using min
        # ensures the val set tests genuine extrapolation.)
        first_target = t0 + L + min(HORIZONS) - 1
        win_is_val.append(bool(is_val[first_target]))

    win_is_val_arr = np.array(win_is_val, dtype=bool)
    train_idx = np.where(~win_is_val_arr)[0]
    val_idx = np.where(win_is_val_arr)[0]
    if len(val_idx) == 0:
        # Fallback: last 15% of windows used for early stopping.
        n_win = len(wins_t)
        split = int(n_win * 0.85)
        train_idx = np.arange(0, split)
        val_idx = np.arange(split, n_win)

    T_tensor = torch.tensor(np.stack(wins_t), dtype=torch.float32, device=DEVICE)
    X_tensor = torch.tensor(np.stack(wins_x), dtype=torch.float32, device=DEVICE)
    Y_tensor = torch.tensor(np.stack(wins_y), dtype=torch.float32, device=DEVICE)

    pinn = RegionalPINN(population=float(population), t_min=0.0, t_max=1.0).to(DEVICE)
    if skip_pretrain:
        # Ablation: PINN remains randomly-initialised (sigmoid outputs ≈ 0.5).
        # No data-fit, no ODE-residual; provides 12 near-constant "noise"
        # channels to the GRU.
        pinn_parts = {"data": float("nan"), "ode": float("nan"), "total": float("nan")}
    else:
        pinn_parts = pretrain_pinn_region(
            pinn, t_norm, series_h, series_y, population,
            epochs=pinn_pretrain_epochs,
        )

    head_cfg = QuantileHeadConfig(
        input_dim=8 + 4 + series_x.shape[1],
        hidden_dim=128,
        num_layers=2,
        dropout=0.15,
        horizons=HORIZONS,
        quantiles=QUANTILES,
        decoder_hidden=64,
        target_index_in_extra=0,  # mv_beds is COVARS[0]
    )
    model = PinnGRUQuantileForecaster(pinn, head_cfg, train_pinn=False).to(DEVICE)

    # Ablation knobs: pin the corresponding anchor to ≈0 via a large negative
    # logit and freeze it so the trained head cannot recover the lost signal.
    if disable_level_anchor:
        with torch.no_grad():
            model.head.level_anchor.fill_(-15.0)
        model.head.level_anchor.requires_grad_(False)
    if disable_trend_anchor:
        with torch.no_grad():
            model.head.trend_anchor.fill_(-15.0)
        model.head.trend_anchor.requires_grad_(False)

    # Ablation: zero out the 4 PINN parameter features (β, γ_c, δ_c, η) in
    # the augmented features. The PINN state features (8 compartments) and
    # the covariate (z-scored mv_beds) are unchanged. Isolates the marginal
    # contribution of the learned time-varying parameters.
    if mask_param_features:
        original_augmented = model._augmented_features
        def _augmented_no_params(t_seq, x_extra):
            feats = original_augmented(t_seq, x_extra)
            # state (8) | params (4) | x_extra (1)  → zero out params slice
            feats = feats.clone()
            feats[..., 8:12] = 0.0
            return feats
        model._augmented_features = _augmented_no_params

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_state: dict | None = None
    plateau = 0
    rng = np.random.default_rng(SEED)

    Tv = T_tensor[val_idx]
    Xv = X_tensor[val_idx]
    Yv = Y_tensor[val_idx]

    q50_idx = QUANTILES.index(0.5)
    mse = torch.nn.MSELoss()
    for _epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(train_idx))
        for s in range(0, len(perm), batch_size):
            batch = train_idx[perm[s : s + batch_size]]
            y_hat = model(T_tensor[batch], X_tensor[batch])
            if use_mse_loss:
                # Decision-aware ablation: replace pinball with MSE on the
                # q50 output only. The q10 and q90 channels are unsupervised
                # and become unusable as calibrated bounds — this ablation
                # therefore reports only the point forecast.
                loss = mse(y_hat[..., q50_idx], Y_tensor[batch])
            else:
                # Pure pinball loss: pinball-q50 is already L1 (median) for
                # point accuracy, pinball-q90 is the asymmetric decision-
                # aware term.
                loss = pinball_loss_multiq(y_hat, Y_tensor[batch], QUANTILES)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            yv_hat = model(Tv, Xv)
            if use_mse_loss:
                val = float(mse(yv_hat[..., q50_idx], Yv).detach())
            else:
                val = float(pinball_loss_multiq(yv_hat, Yv, QUANTILES).detach())
        if val < best_val - 1e-6:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            plateau = 0
        else:
            plateau += 1
            if plateau >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "mu_y": mu_y,
        "sigma_y": sigma_y,
        "mu_x": mu_x,
        "sigma_x": sigma_x,
        "best_val_pinball": best_val,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "pinn_data": pinn_parts["data"],
        "pinn_ode": pinn_parts["ode"],
    }


def predict_pinn_gru_at_origins(
    models: dict[str, PinnGRUQuantileForecaster],
    scalers: dict[str, dict],
    df: pd.DataFrame,
    origins: pd.DatetimeIndex,
    date_to_norm: dict[pd.Timestamp, float],
) -> pd.DataFrame:
    """Roll the trained quantile forecaster across origins.

    Output shape per (origin, region, horizon): a single
    ``(y_hat=q50, q_lo=q10, q_hi=q90, y_true)`` record, all on the
    original (population-count) MV-bed scale.
    """
    L = LOOKBACK
    records: list[dict] = []
    q_index = {q: i for i, q in enumerate(QUANTILES)}
    for origin in origins:
        for region, model in models.items():
            sc = scalers[region]
            sub = df[df["region_name"] == region].sort_values("date")
            hist = sub[sub["date"] < origin]
            if len(hist) < L:
                continue
            window = hist.tail(L)
            x_win = ((window[list(COVARS)].to_numpy() - sc["mu_x"]) / sc["sigma_x"]).astype(np.float32)
            t_win = np.array(
                [date_to_norm[d] for d in window["date"]], dtype=np.float32
            )
            t_t = torch.tensor(t_win, dtype=torch.float32, device=DEVICE).reshape(1, L, 1)
            x_t = torch.tensor(x_win, dtype=torch.float32, device=DEVICE).reshape(1, L, x_win.shape[1])

            model.eval()
            with torch.no_grad():
                y_q = model(t_t, x_t)[0].cpu().numpy()  # (K, Q)
            for i, h in enumerate(HORIZONS):
                q10 = float(y_q[i, q_index[0.1]]) * sc["sigma_y"] + sc["mu_y"]
                q50 = float(y_q[i, q_index[0.5]]) * sc["sigma_y"] + sc["mu_y"]
                q90 = float(y_q[i, q_index[0.9]]) * sc["sigma_y"] + sc["mu_y"]
                target_date = origin + pd.Timedelta(days=int(h) - 1)
                y_rows = sub[sub["date"] == target_date]
                if y_rows.empty:
                    continue
                records.append({
                    "model": "pinn_gru",
                    "origin": origin,
                    "region": region,
                    "horizon": int(h),
                    "y_hat": q50,
                    "q_lo": q10,
                    "q_hi": q90,
                    "y_true": float(y_rows[TARGET].iloc[0]),
                })
    return pd.DataFrame(records)


def run_pinn_ablations_main() -> int:
    """Train PinnGRU ablations and save one forecast parquet per ablation."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = load_regional_csv(DEFAULT_CSV)
    test_start = pd.Timestamp(DEFAULT_SPLIT_DATES["test"][0])
    test_end = pd.Timestamp(DEFAULT_SPLIT_DATES["test"][1])
    origins = build_origins(test_start, test_end, LOOKBACK, max(HORIZONS), ORIGIN_STRIDE)

    static = pd.read_csv(ROOT / "data" / "processed" / "regional_static.csv")
    pop_by_code = dict(zip(static["region_code"], static["population"]))

    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    date_to_norm = {d: (i / max(len(all_dates) - 1, 1)) for i, d in enumerate(all_dates)}

    df_tv = df[df["date"] < test_start].copy()
    val_start = pd.Timestamp(DEFAULT_SPLIT_DATES["val"][0])
    val_end = pd.Timestamp(DEFAULT_SPLIT_DATES["val"][1])

    for ablation in ABLATIONS:
        tag = ablation["tag"]
        label = ablation["label"]
        kwargs = ablation["kwargs"]
        out_path = OUT_DIR / f"forecasts_pinn_gru__{tag}.parquet"
        if out_path.exists():
            print(f"\n=== Ablation: {label} (cached -> {out_path.name}) ===", flush=True)
            continue

        print(f"\n=== Ablation: {label} ===", flush=True)
        t_start = time.time()
        pinn_models = {}
        pinn_scalers = {}
        for region_name, group in df_tv.groupby("region_name", observed=True):
            group_sorted = group.sort_values("date")
            region_code = group_sorted["region_code"].iloc[0]
            pop = pop_by_code[region_code]
            series_y = group_sorted[TARGET].to_numpy(dtype=float)
            series_x = group_sorted[list(COVARS)].to_numpy(dtype=float)
            series_h = group_sorted["hospital_cases"].to_numpy(dtype=float)
            dates = group_sorted["date"].to_numpy()
            is_val = (dates >= val_start.to_datetime64()) & (dates <= val_end.to_datetime64())
            t_norm = np.array([date_to_norm[d] for d in group_sorted["date"]], dtype=np.float32)

            t_region = time.time()
            model, scaler = train_pinn_gru_region(
                series_y, series_x, series_h, t_norm, is_val, pop, **kwargs,
            )
            print(
                f"  {region_name:30s}  val_pinball={scaler['best_val_pinball']:.4f}  "
                f"({time.time() - t_region:.1f}s)",
                flush=True,
            )
            pinn_models[region_name] = model
            pinn_scalers[region_name] = scaler

        fc = predict_pinn_gru_at_origins(pinn_models, pinn_scalers, df, origins, date_to_norm)
        fc["model"] = f"pinn_gru__{tag}"
        fc.to_parquet(out_path, index=False)
        print(f"  -> {out_path}  ({len(fc)} forecasts, {time.time() - t_start:.1f}s)")

    print("\nAblation forecasts written. Rebuild Table 1 with:")
    print("  ukci-forecast-evaluation table1")
    return 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")
    df = load_regional_csv(DEFAULT_CSV)
    print(f"Loaded {len(df):,} rows, {df['region_name'].nunique()} regions, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    test_start = pd.Timestamp(DEFAULT_SPLIT_DATES["test"][0])
    test_end = pd.Timestamp(DEFAULT_SPLIT_DATES["test"][1])
    origins = build_origins(test_start, test_end, LOOKBACK, max(HORIZONS), ORIGIN_STRIDE)
    print(f"Rolling-origin schedule: {len(origins)} origins, "
          f"{origins[0].date()} -> {origins[-1].date()} every {ORIGIN_STRIDE}d")

    static = pd.read_csv(ROOT / "data" / "processed" / "regional_static.csv")
    pop_by_code = dict(zip(static["region_code"], static["population"]))

    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    n_total = len(all_dates)
    date_to_norm = {d: (i / max(n_total - 1, 1)) for i, d in enumerate(all_dates)}

    # ---- 1. SeasonalNaive(7) ----
    print("\n[1/5] SeasonalNaive(7) ...", flush=True)
    t0 = time.time()
    sn = fit_predict_baseline_per_origin(df, origins, SeasonalNaive, "seasonal_naive")
    sn.to_parquet(OUT_DIR / "forecasts_seasonal_naive.parquet", index=False)
    print(f"  {len(sn)} forecasts in {time.time() - t0:.1f}s")

    # ---- 2. ARIMAPerRegion ----
    print("\n[2/5] ARIMAPerRegion (AIC grid, refit per origin)...", flush=True)
    t0 = time.time()
    arima = fit_predict_baseline_per_origin(df, origins, ARIMAPerRegion, "arima_per_region")
    arima.to_parquet(OUT_DIR / "forecasts_arima_per_region.parquet", index=False)
    print(f"  {len(arima)} forecasts in {time.time() - t0:.1f}s")

    # ---- 3. XGBoostPerRegion (lag features, refit per origin) ----
    print("\n[3/5] XGBoostPerRegion (lag features, refit per origin)...", flush=True)
    t0 = time.time()
    xgb = fit_predict_baseline_per_origin(df, origins, XGBoostPerRegion, "xgboost_per_region")
    xgb.to_parquet(OUT_DIR / "forecasts_xgboost_per_region.parquet", index=False)
    print(f"  {len(xgb)} forecasts in {time.time() - t0:.1f}s")

    # ---- 4. GRUPerRegion (fit on train+val, slide on test) ----
    print("\n[4/5] GRUPerRegion (fit on train+val)...", flush=True)
    t0 = time.time()
    df_tv = df[df["date"] < test_start].copy()
    gru = fit_gru_per_region(df_tv)
    gru_forecasts = predict_gru_at_origins(gru, df, origins)
    gru_forecasts.to_parquet(OUT_DIR / "forecasts_gru_per_region.parquet", index=False)
    print(f"  {len(gru_forecasts)} forecasts in {time.time() - t0:.1f}s")

    # ---- 5. PinnGRU (per region, quantile head + Delta-validation early stop) ----
    print("\n[5/5] PinnGRU (quantile head, level-anchor, Delta val early stop)...", flush=True)
    t0 = time.time()
    val_start = pd.Timestamp(DEFAULT_SPLIT_DATES["val"][0])
    val_end = pd.Timestamp(DEFAULT_SPLIT_DATES["val"][1])
    pinn_models: dict[str, PinnGRUQuantileForecaster] = {}
    pinn_scalers: dict[str, dict] = {}
    for region_name, group in df_tv.groupby("region_name", observed=True):
        group_sorted = group.sort_values("date")
        region_code = group_sorted["region_code"].iloc[0]
        pop = pop_by_code[region_code]
        series_y = group_sorted[TARGET].to_numpy(dtype=float)
        series_x = group_sorted[list(COVARS)].to_numpy(dtype=float)
        series_h = group_sorted["hospital_cases"].to_numpy(dtype=float)
        dates = group_sorted["date"].to_numpy()
        is_val = (dates >= val_start.to_datetime64()) & (dates <= val_end.to_datetime64())
        t_norm = np.array([date_to_norm[d] for d in group_sorted["date"]], dtype=np.float32)
        t_region = time.time()
        model, scaler = train_pinn_gru_region(
            series_y, series_x, series_h, t_norm, is_val, pop
        )
        print(f"    {region_name:30s} N={pop / 1e6:.2f}M  "
              f"val_pinball={scaler['best_val_pinball']:.4f}  "
              f"(n_tr={scaler['n_train']}, n_va={scaler['n_val']}, "
              f"pinn_data={scaler['pinn_data']:.1e}, pinn_ode={scaler['pinn_ode']:.1e})  "
              f"({time.time() - t_region:.1f}s)", flush=True)
        pinn_models[region_name] = model
        pinn_scalers[region_name] = scaler
    pinn_forecasts = predict_pinn_gru_at_origins(
        pinn_models, pinn_scalers, df, origins, date_to_norm
    )
    pinn_forecasts.to_parquet(OUT_DIR / "forecasts_pinn_gru.parquet", index=False)
    print(f"  {len(pinn_forecasts)} forecasts in {time.time() - t0:.1f}s total")

    # ---- Combine + tabulate ----
    all_forecasts = pd.concat(
        [sn, arima, xgb, gru_forecasts, pinn_forecasts], ignore_index=True
    )
    all_forecasts.to_parquet(OUT_DIR / "forecasts.parquet", index=False)
    print(f"\nCombined forecasts -> {OUT_DIR / 'forecasts.parquet'}  ({len(all_forecasts)} rows)")

    print("\nForecasts written. Compute metric panel with:")
    print("  ukci-forecast-evaluation metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
