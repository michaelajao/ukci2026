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

import os
# Allow torch's bundled libiomp5md.dll to coexist with the one statsmodels/MKL
# loads via numpy on Windows; without this, SARIMAX aborts with OMP error #15.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.loader import DEFAULT_CSV, DEFAULT_SPLIT_DATES, load_regional_csv
from forecasting.baselines import ARIMAPerRegion, GRUPerRegion, SeasonalNaive
from forecasting.composite_loss import (
    CompositeLoss,
    CompositeLossConfig,
    PinnGRUForecaster,
    TemporalHeadConfig,
    mc_dropout_quantiles,
)
from forecasting.pinn_seird import RegionalPINN

# ---------------------------------------------------------------------------
# Hyperparameters / paths
# ---------------------------------------------------------------------------

HORIZONS: tuple[int, ...] = (7, 14, 21, 28)
LOOKBACK = 28
ORIGIN_STRIDE = 14
TARGET = "mv_beds"
# Features fed to the PinnGRU lookback window. ``mv_beds`` (the target itself)
# is included so the GRU has an autoregressive anchor on the recent level;
# without it the PINN state+param features alone (sigmoid-bounded in [0, 1])
# cannot ground the predicted level and the model regresses to the training
# mean, breaking under distribution shift between Alpha+Delta and Omicron.
COVARS: tuple[str, ...] = ("mv_beds", "admissions", "hospital_cases", "occupied_beds")
SEED = 0
MC_K = 100

OUT_DIR = ROOT / "results" / "forecasting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train_pinn_gru_region(
    series_y: np.ndarray,
    series_x: np.ndarray,
    t_norm: np.ndarray,
    population: float,
    epochs: int = 200,
    patience: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[PinnGRUForecaster, dict]:
    """Fit one PinnGRUForecaster on one region's series.

    Args:
        series_y: (T,) target values on the original scale.
        series_x: (T, F_cov) covariate values on the original scale.
        t_norm:   (T,) absolute-day-normalised time index in [0, 1].
        population: regional population for the PINN's state denormalisation.

    Returns:
        (model, scalers) where scalers is
        ``{"mu_y", "sigma_y", "mu_x", "sigma_x"}``.
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
    for t0 in range(0, n - L - max_h + 1):
        wins_t.append(t_norm[t0 : t0 + L].astype(np.float32)[:, None])
        wins_x.append(x_scaled[t0 : t0 + L])
        wins_y.append([float(y_scaled[t0 + L + h - 1]) for h in HORIZONS])

    T_tensor = torch.tensor(np.stack(wins_t), dtype=torch.float32, device=DEVICE)
    X_tensor = torch.tensor(np.stack(wins_x), dtype=torch.float32, device=DEVICE)
    Y_tensor = torch.tensor(np.stack(wins_y), dtype=torch.float32, device=DEVICE)

    pinn = RegionalPINN(population=float(population), t_min=0.0, t_max=1.0)
    head_cfg = TemporalHeadConfig(
        input_dim=8 + 4 + series_x.shape[1],
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        horizons=HORIZONS,
    )
    model = PinnGRUForecaster(pinn, head_cfg, train_pinn=True).to(DEVICE)

    composite = CompositeLoss(
        CompositeLossConfig(
            lambda_phys=0.0,
            lambda_under=0.5,
            lambda_smooth=0.01,
            huber_delta=1.0,
            use_forecast=True,
            use_phys=False,
            use_under=True,
            use_smooth=True,
        )
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state: dict | None = None
    plateau = 0
    n_samples = len(T_tensor)
    rng = np.random.default_rng(SEED)
    for _epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_samples)
        running = 0.0
        count = 0
        for s in range(0, n_samples, batch_size):
            idx = perm[s : s + batch_size]
            t_b = T_tensor[idx]
            x_b = X_tensor[idx]
            y_b = Y_tensor[idx]
            y_hat = model(t_b, x_b)
            loss, _ = composite(y_hat, y_b, pinn_residual=None)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach()) * len(idx)
            count += len(idx)
        avg = running / max(count, 1)
        if avg < best_loss - 1e-6:
            best_loss = avg
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
        "best_loss": best_loss,
    }


def predict_pinn_gru_at_origins(
    models: dict[str, PinnGRUForecaster],
    scalers: dict[str, dict],
    df: pd.DataFrame,
    origins: pd.DatetimeIndex,
    date_to_norm: dict[pd.Timestamp, float],
) -> pd.DataFrame:
    L = LOOKBACK
    records: list[dict] = []
    for origin in origins:
        for region, model in models.items():
            sc = scalers[region]
            sub = df[df["region_name"] == region].sort_values("date")
            hist = sub[sub["date"] < origin]
            if len(hist) < L:
                continue
            window = hist.tail(L)
            y_win = ((window[TARGET].to_numpy() - sc["mu_y"]) / sc["sigma_y"]).astype(np.float32)
            x_win = ((window[list(COVARS)].to_numpy() - sc["mu_x"]) / sc["sigma_x"]).astype(np.float32)
            t_win = np.array(
                [date_to_norm[d] for d in window["date"]], dtype=np.float32
            )

            t_t = torch.tensor(t_win, dtype=torch.float32, device=DEVICE).reshape(1, L, 1)
            x_t = torch.tensor(x_win, dtype=torch.float32, device=DEVICE).reshape(1, L, x_win.shape[1])

            q = mc_dropout_quantiles(model, t_t, x_t, k=MC_K, quantiles=(0.1, 0.5, 0.9))
            for i, h in enumerate(HORIZONS):
                q10 = float(q[0.1][0, i].item()) * sc["sigma_y"] + sc["mu_y"]
                q50 = float(q[0.5][0, i].item()) * sc["sigma_y"] + sc["mu_y"]
                q90 = float(q[0.9][0, i].item()) * sc["sigma_y"] + sc["mu_y"]
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
    print("\n[1/4] SeasonalNaive(7) ...", flush=True)
    t0 = time.time()
    sn = fit_predict_baseline_per_origin(df, origins, SeasonalNaive, "seasonal_naive")
    sn.to_parquet(OUT_DIR / "forecasts_seasonal_naive.parquet", index=False)
    print(f"  {len(sn)} forecasts in {time.time() - t0:.1f}s")

    # ---- 2. ARIMAPerRegion ----
    print("\n[2/4] ARIMAPerRegion (AIC grid, refit per origin)...", flush=True)
    t0 = time.time()
    arima = fit_predict_baseline_per_origin(df, origins, ARIMAPerRegion, "arima_per_region")
    arima.to_parquet(OUT_DIR / "forecasts_arima_per_region.parquet", index=False)
    print(f"  {len(arima)} forecasts in {time.time() - t0:.1f}s")

    # ---- 3. GRUPerRegion (fit on train+val, slide on test) ----
    print("\n[3/4] GRUPerRegion (fit on train+val)...", flush=True)
    t0 = time.time()
    df_tv = df[df["date"] < test_start].copy()
    gru = fit_gru_per_region(df_tv)
    gru_forecasts = predict_gru_at_origins(gru, df, origins)
    gru_forecasts.to_parquet(OUT_DIR / "forecasts_gru_per_region.parquet", index=False)
    print(f"  {len(gru_forecasts)} forecasts in {time.time() - t0:.1f}s")

    # ---- 4. PinnGRU (per region, composite loss) ----
    print("\n[4/4] PinnGRU (per-region, composite loss, MC Dropout K=100)...", flush=True)
    t0 = time.time()
    pinn_models: dict[str, PinnGRUForecaster] = {}
    pinn_scalers: dict[str, dict] = {}
    for region_name, group in df_tv.groupby("region_name", observed=True):
        group_sorted = group.sort_values("date")
        region_code = group_sorted["region_code"].iloc[0]
        pop = pop_by_code[region_code]
        series_y = group_sorted[TARGET].to_numpy(dtype=float)
        series_x = group_sorted[list(COVARS)].to_numpy(dtype=float)
        t_norm = np.array([date_to_norm[d] for d in group_sorted["date"]], dtype=np.float32)
        t_region = time.time()
        model, scaler = train_pinn_gru_region(series_y, series_x, t_norm, pop)
        print(f"    {region_name:30s} N={pop / 1e6:.2f}M  "
              f"final_loss={scaler['best_loss']:.4f}  "
              f"({time.time() - t_region:.1f}s)", flush=True)
        pinn_models[region_name] = model
        pinn_scalers[region_name] = scaler
    pinn_forecasts = predict_pinn_gru_at_origins(
        pinn_models, pinn_scalers, df, origins, date_to_norm
    )
    pinn_forecasts.to_parquet(OUT_DIR / "forecasts_pinn_gru.parquet", index=False)
    print(f"  {len(pinn_forecasts)} forecasts in {time.time() - t0:.1f}s total")

    # ---- Combine + tabulate ----
    all_forecasts = pd.concat([sn, arima, gru_forecasts, pinn_forecasts], ignore_index=True)
    all_forecasts.to_parquet(OUT_DIR / "forecasts.parquet", index=False)
    print(f"\nCombined forecasts -> {OUT_DIR / 'forecasts.parquet'}  ({len(all_forecasts)} rows)")

    print("\nForecasts written. Compute metric panel with:")
    print("  python scripts/build_metric_table.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
