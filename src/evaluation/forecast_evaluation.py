"""Forecast evaluation artifacts for the UKCI 2026 paper.

This module owns the paper-facing forecast evaluation pipeline:

- metric CSVs from rolling-origin forecasts;
- paper Table 1 CSV from saved model/ablation forecasts;
- the regional forecast panel figure.

The primitive metric definitions remain in :mod:`evaluation.forecast_metrics`.
This file only assembles outputs from trained artifacts under
``results/forecasting/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import configure_utf8_stdout, repo_root, results_dir

configure_utf8_stdout()

ROOT = repo_root()
OUT_DIR = results_dir("forecasting")
FORECASTS = OUT_DIR / "forecasts.parquet"
HORIZONS = (7, 14, 21, 28)
REGION_ORDER = ("Y56", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63")

BASELINE_ROWS = [
    ("seasonal_naive", "Seasonal-naive(7)"),
    ("arima_per_region", "ARIMA"),
    ("xgboost_per_region", "XGBoost (lag features)"),
    ("gru_per_region", "GRU per region"),
    ("pinn_gru", "PinnGRU (proposed)"),
]
ABLATION_ROWS = [
    ("pinn_gru__no_decision_aware", "w/o decision-aware loss (MSE)"),
    ("pinn_gru__no_pretrain", "w/o PINN pre-training"),
    ("pinn_gru__no_params", "w/o PINN parameter features"),
    ("pinn_gru__no_level", "w/o level anchor"),
    ("pinn_gru__no_trend", "w/o trend anchor"),
]

FORECAST_FILES = {
    "seasonal_naive": "forecasts_seasonal_naive.parquet",
    "arima_per_region": "forecasts_arima_per_region.parquet",
    "xgboost_per_region": "forecasts_xgboost_per_region.parquet",
    "gru_per_region": "forecasts_gru_per_region.parquet",
    "pinn_gru": "forecasts_pinn_gru.parquet",
    "pinn_gru__no_decision_aware": "forecasts_pinn_gru__no_decision_aware.parquet",
    "pinn_gru__no_pretrain": "forecasts_pinn_gru__no_pretrain.parquet",
    "pinn_gru__no_params": "forecasts_pinn_gru__no_params.parquet",
    "pinn_gru__no_level": "forecasts_pinn_gru__no_level.parquet",
    "pinn_gru__no_trend": "forecasts_pinn_gru__no_trend.parquet",
}

BOOTSTRAP_SEED = 20260515
BOOTSTRAP_REPS = 2000

def per_region_panel(
    y_true,
    y_hat,
    q_lo,
    q_hi,
) -> dict[str, float]:
    """Compute all metrics for one model-region-horizon trajectory."""
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_pinball_loss,
        root_mean_squared_error,
    )

    from evaluation.forecast_metrics import (
        expected_shortage,
        peak_error,
        peak_timing_error,
        underestimation_rate,
        wis,
    )

    out = {
        "mae": float(mean_absolute_error(y_true, y_hat)),
        "rmse": float(root_mean_squared_error(y_true, y_hat)),
        "mape": float(mean_absolute_percentage_error(y_true, y_hat)),
        "underestimation_rate": underestimation_rate(y_true, y_hat),
        "expected_shortage": expected_shortage(y_true, y_hat),
        "peak_error": peak_error(y_true, y_hat),
        "peak_timing_error": peak_timing_error(y_true, y_hat),
    }
    if q_lo is not None and q_hi is not None:
        quantiles = np.stack([np.stack([q_lo, q_hi], axis=0)], axis=0)
        out["wis_80"] = wis(y_true, y_hat, [0.10], quantiles)
        out["pinball_q10"] = float(mean_pinball_loss(y_true, q_lo, alpha=0.10))
        out["pinball_q90"] = float(mean_pinball_loss(y_true, q_hi, alpha=0.90))
    else:
        out["wis_80"] = float("nan")
        out["pinball_q10"] = float("nan")
        out["pinball_q90"] = float("nan")
    return out


def metric_detail_from_forecasts(forecasts):
    """Return per-model, per-region, per-horizon metric rows."""
    records: list[dict] = []
    for (model_name, horizon, region), sub in forecasts.groupby(["model", "horizon", "region"]):
        sub = sub.sort_values("origin")
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_hat = sub["y_hat"].to_numpy(dtype=float)
        has_q = sub["q_lo"].notna().all() and sub["q_hi"].notna().all()
        q_lo = sub["q_lo"].to_numpy(dtype=float) if has_q else None
        q_hi = sub["q_hi"].to_numpy(dtype=float) if has_q else None
        records.append(
            {
                "model": model_name,
                "horizon": int(horizon),
                "region": region,
                "n": len(sub),
                **per_region_panel(y_true, y_hat, q_lo, q_hi),
            }
        )
    import pandas as pd

    return pd.DataFrame(records)


def metric_table_from_detail(detail):
    """Macro-average detail metrics into the headline CSV table."""
    agg_funcs = {
        "mae": "mean",
        "rmse": "mean",
        "mape": "mean",
        "underestimation_rate": "mean",
        "expected_shortage": "sum",
        "peak_error": "mean",
        "peak_timing_error": "mean",
        "wis_80": "mean",
        "pinball_q10": "mean",
        "pinball_q90": "mean",
    }
    by_horizon = detail.groupby(["model", "horizon"]).agg(agg_funcs).reset_index()
    overall = detail.groupby(["model"]).agg(agg_funcs).reset_index().assign(horizon="all")
    import pandas as pd

    return pd.concat([by_horizon, overall[by_horizon.columns]], ignore_index=True)


def build_metric_tables():
    """Write ``table_metrics_detail.csv`` and ``table_metrics.csv``."""
    if not FORECASTS.exists():
        raise FileNotFoundError(f"{FORECASTS} not found. Run ukci-train-forecasters first.")
    import pandas as pd

    forecasts = pd.read_parquet(FORECASTS)
    detail = metric_detail_from_forecasts(forecasts)
    table = metric_table_from_detail(detail)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT_DIR / "table_metrics_detail.csv", index=False)
    table.to_csv(OUT_DIR / "table_metrics.csv", index=False)
    return detail, table


def build_metric_tables_main() -> int:
    detail, table = build_metric_tables()
    print(f"Wrote {OUT_DIR / 'table_metrics_detail.csv'} ({len(detail):,} rows)")
    print(f"Wrote {OUT_DIR / 'table_metrics.csv'} ({len(table):,} rows)")
    print()
    print(table[table["horizon"] == "all"].set_index("model").round(3).to_string())
    return 0


def load_all_model_forecasts():
    """Load saved per-model forecast parquet files, including ablations."""
    parts = []
    missing: list[str] = []
    import pandas as pd

    for model_key, filename in FORECAST_FILES.items():
        path = OUT_DIR / filename
        if not path.exists():
            missing.append(filename)
            continue
        df = pd.read_parquet(path)
        df["model"] = model_key
        parts.append(df)
    if missing:
        print("WARN: missing forecast files; omitted from paper table:")
        for filename in missing:
            print(f"  - {filename}")
    if not parts:
        raise FileNotFoundError(f"No forecast parquet files found in {OUT_DIR}")
    return pd.concat(parts, ignore_index=True)


def build_paper_table(detail):
    """Build paper Table 1 from metric detail rows."""
    agg = (
        detail.groupby(["model", "horizon"])
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), under_rate=("underestimation_rate", "mean"))
        .reset_index()
    )
    rmse = agg.pivot(index="model", columns="horizon", values="rmse")
    mae = agg.pivot(index="model", columns="horizon", values="mae")
    under_h28 = agg[agg["horizon"] == 28].set_index("model")["under_rate"]

    import pandas as pd

    rows: list[dict] = []
    for model_key, label in BASELINE_ROWS + ABLATION_ROWS:
        if model_key not in rmse.index:
            print(f"WARN: no metrics for {model_key}; row skipped.")
            continue
        rows.append(
            {
                "model_key": model_key,
                "Model": label,
                "RMSE h=7": rmse.loc[model_key, 7],
                "RMSE h=14": rmse.loc[model_key, 14],
                "RMSE h=21": rmse.loc[model_key, 21],
                "RMSE h=28": rmse.loc[model_key, 28],
                "MAE h=14": mae.loc[model_key, 14],
                "Under (%) h=28": 100.0 * under_h28.loc[model_key],
            }
        )
    return pd.DataFrame(rows)


def format_paper_table_csv(table):
    """Round and stringify the paper table for direct manuscript reading."""
    out = table.copy()
    numeric_cols = ["RMSE h=7", "RMSE h=14", "RMSE h=21", "RMSE h=28", "MAE h=14"]
    for col in numeric_cols:
        out[col] = out[col].map(lambda value: f"{value:.2f}")
    out["Under (%) h=28"] = out["Under (%) h=28"].map(lambda value: f"{value:.1f}")
    return out


def build_paper_table1():
    """Write the paper-ready forecast CSV from saved model outputs."""
    forecasts = load_all_model_forecasts()
    detail = metric_detail_from_forecasts(forecasts)
    table = build_paper_table(detail)
    formatted = format_paper_table_csv(table).drop(columns=["model_key"])
    formatted.to_csv(OUT_DIR / "table1_paper.csv", index=False)
    return formatted


def build_paper_table1_main() -> int:
    table = build_paper_table1()
    print(f"Wrote {OUT_DIR / 'table1_paper.csv'}")
    print()
    print(table.to_string(index=False))
    return 0


def _metric_for_origin_sample(sub, sample_origins, metric: str) -> float:
    """Compute a paper-table metric after resampling rolling origins.

    Origins are the independent resampling unit. For RMSE and MAE we first
    compute each region's metric across the sampled origins, then macro-average
    across regions, matching the paper table's region-level aggregation.
    Underestimation is averaged across all sampled region-origin pairs.
    """
    import numpy as np
    import pandas as pd

    sampled = pd.concat([sub[sub["origin"] == origin] for origin in sample_origins], ignore_index=True)
    if sampled.empty:
        return float("nan")
    if metric == "underestimation_rate":
        return float(np.mean(sampled["y_hat"].to_numpy(dtype=float) < sampled["y_true"].to_numpy(dtype=float)))

    region_values = []
    for _, region_sub in sampled.groupby("region"):
        error = region_sub["y_hat"].to_numpy(dtype=float) - region_sub["y_true"].to_numpy(dtype=float)
        if metric == "rmse":
            region_values.append(float(np.sqrt(np.mean(error ** 2))))
        elif metric == "mae":
            region_values.append(float(np.mean(np.abs(error))))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return float(np.mean(region_values))


def _bootstrap_ci(sub, metric: str, reps: int = BOOTSTRAP_REPS) -> tuple[float, float, float]:
    """Return mean and percentile CI for one model-horizon metric."""
    import numpy as np

    work = sub.copy()
    work["origin"] = work["origin"].astype("datetime64[ns]")
    origins = np.array(sorted(work["origin"].unique()))
    if metric == "rmse":
        work["metric_value"] = (work["y_hat"].astype(float) - work["y_true"].astype(float)) ** 2
        values = (
            work.pivot(index="origin", columns="region", values="metric_value")
            .reindex(origins)
            .to_numpy(dtype=float)
        )
        mean_value = float(np.nanmean(np.sqrt(np.nanmean(values, axis=0))))
    elif metric == "mae":
        work["metric_value"] = np.abs(work["y_hat"].astype(float) - work["y_true"].astype(float))
        values = (
            work.pivot(index="origin", columns="region", values="metric_value")
            .reindex(origins)
            .to_numpy(dtype=float)
        )
        mean_value = float(np.nanmean(np.nanmean(values, axis=0)))
    elif metric == "underestimation_rate":
        work["metric_value"] = (work["y_hat"].astype(float) < work["y_true"].astype(float)).astype(float)
        values = (
            work.pivot(index="origin", columns="region", values="metric_value")
            .reindex(origins)
            .to_numpy(dtype=float)
        )
        mean_value = float(np.nanmean(values))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if len(origins) < 2:
        return mean_value, float("nan"), float("nan")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_idx = rng.integers(0, len(origins), size=(reps, len(origins)))
    sampled = values[sample_idx, :]
    if metric == "rmse":
        draws = np.nanmean(np.sqrt(np.nanmean(sampled, axis=1)), axis=1)
    elif metric == "mae":
        draws = np.nanmean(np.nanmean(sampled, axis=1), axis=1)
    else:
        draws = np.nanmean(sampled, axis=(1, 2))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return mean_value, float(lo), float(hi)


def _paired_origin_series(forecasts, model: str, horizon: int):
    """Per-origin RMSE series for paired model comparison."""
    import numpy as np

    sub = forecasts[(forecasts["model"] == model) & (forecasts["horizon"] == horizon)]
    values = {}
    for origin, origin_sub in sub.groupby("origin"):
        error = origin_sub["y_hat"].to_numpy(dtype=float) - origin_sub["y_true"].to_numpy(dtype=float)
        values[origin] = float(np.sqrt(np.mean(error ** 2)))
    return values


def _paired_tests_against_comparators(forecasts):
    """Paired origin-level tests comparing PinnGRU with key comparators."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    rows: list[dict] = []
    comparators = [
        ("arima_per_region", "ARIMA"),
        ("gru_per_region", "GRU per region"),
    ]
    for comparator_key, comparator_label in comparators:
        for horizon in HORIZONS:
            pinn = _paired_origin_series(forecasts, "pinn_gru", horizon)
            comparator = _paired_origin_series(forecasts, comparator_key, horizon)
            origins = sorted(set(pinn).intersection(comparator))
            if not origins:
                continue
            pinn_values = np.array([pinn[origin] for origin in origins], dtype=float)
            comparator_values = np.array([comparator[origin] for origin in origins], dtype=float)
            diff = pinn_values - comparator_values
            rng = np.random.default_rng(BOOTSTRAP_SEED + int(horizon) + len(rows))
            boot_diff = np.empty(BOOTSTRAP_REPS, dtype=float)
            for i in range(BOOTSTRAP_REPS):
                idx = rng.choice(len(origins), size=len(origins), replace=True)
                boot_diff[i] = float(np.mean(diff[idx]))
            diff_lo, diff_hi = np.nanpercentile(boot_diff, [2.5, 97.5])
            t_stat, t_p = stats.ttest_rel(pinn_values, comparator_values, nan_policy="omit")
            try:
                w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox")
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            rows.append({
                "comparison": f"PinnGRU - {comparator_label}",
                "comparator_key": comparator_key,
                "horizon": horizon,
                "n_origins": len(origins),
                "mean_origin_rmse_pinn_gru": float(np.mean(pinn_values)),
                "mean_origin_rmse_comparator": float(np.mean(comparator_values)),
                "mean_origin_rmse_diff": float(np.mean(diff)),
                "diff_ci_low": float(diff_lo),
                "diff_ci_high": float(diff_hi),
                "paired_t_stat": float(t_stat),
                "paired_t_p": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
            })
    return pd.DataFrame(rows)


def build_uncertainty_tables():
    """Write origin-bootstrap CIs and paired forecast-comparison tests."""
    import pandas as pd

    forecasts = load_all_model_forecasts()
    rows: list[dict] = []
    for model_key, label in BASELINE_ROWS + ABLATION_ROWS:
        model_forecasts = forecasts[forecasts["model"] == model_key]
        if model_forecasts.empty:
            continue
        for horizon in HORIZONS:
            sub = model_forecasts[model_forecasts["horizon"] == horizon]
            if sub.empty:
                continue
            rmse, rmse_lo, rmse_hi = _bootstrap_ci(sub, "rmse")
            mae, mae_lo, mae_hi = _bootstrap_ci(sub, "mae")
            under, under_lo, under_hi = _bootstrap_ci(sub, "underestimation_rate")
            rows.append({
                "model_key": model_key,
                "Model": label,
                "horizon": horizon,
                "n_origins": int(sub["origin"].nunique()),
                "rmse": rmse,
                "rmse_ci_low": rmse_lo,
                "rmse_ci_high": rmse_hi,
                "mae": mae,
                "mae_ci_low": mae_lo,
                "mae_ci_high": mae_hi,
                "underestimation_rate": under,
                "under_ci_low": under_lo,
                "under_ci_high": under_hi,
            })
    uncertainty = pd.DataFrame(rows)
    tests = _paired_tests_against_comparators(forecasts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uncertainty.to_csv(OUT_DIR / "table1_uncertainty.csv", index=False)
    tests.to_csv(OUT_DIR / "table1_paired_tests.csv", index=False)
    return uncertainty, tests


def build_uncertainty_tables_main() -> int:
    uncertainty, tests = build_uncertainty_tables()
    print(f"Wrote {OUT_DIR / 'table1_uncertainty.csv'} ({len(uncertainty):,} rows)")
    print(f"Wrote {OUT_DIR / 'table1_paired_tests.csv'} ({len(tests):,} rows)")
    print()
    print(tests.round(4).to_string(index=False))
    return 0


def build_forecast_panel_figure(horizon: int = 14) -> Path:
    """Write the regional rolling-origin forecast panel figure."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    from evaluation.figures import (
        FORECASTER_STYLES,
        FULL_WIDTH_IN,
        TRUTH_COLOUR,
        apply_paper_style,
        save_figure,
    )

    apply_paper_style()
    forecasts_path = OUT_DIR / "forecasts.parquet"
    daily_path = ROOT / "data" / "processed" / "regional_daily.csv"

    fc = pd.read_parquet(forecasts_path)
    fc = fc[fc["horizon"] == horizon].copy()
    fc["target_date"] = fc["origin"] + pd.Timedelta(days=horizon - 1)

    daily = pd.read_csv(daily_path, parse_dates=["date"])
    test_start = pd.Timestamp("2021-12-01")
    test_end = pd.Timestamp("2022-08-31")
    daily = daily[(daily["date"] >= test_start) & (daily["date"] <= test_end)]
    code_to_name = (
        daily[["region_code", "region_name"]]
        .drop_duplicates()
        .set_index("region_code")["region_name"]
        .to_dict()
    )

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 6.0))
    grid = GridSpec(2, 8, figure=fig)
    axes = [fig.add_subplot(grid[0, 2 * k : 2 * k + 2]) for k in range(4)]
    axes.extend(fig.add_subplot(grid[1, 2 * k + 1 : 2 * k + 3]) for k in range(3))

    for ax, code in zip(axes, REGION_ORDER, strict=False):
        region_name = code_to_name[code]
        truth = daily[daily["region_code"] == code].sort_values("date")
        sub = fc[fc["region"] == region_name].sort_values("target_date")
        ax.plot(truth["date"], truth["mv_beds"], color=TRUTH_COLOUR, linewidth=1.5, zorder=2)

        for i, style in enumerate(FORECASTER_STYLES):
            model_series = sub[sub["model"] == style["model"]].sort_values("target_date")
            if model_series.empty:
                continue
            ax.plot(
                model_series["target_date"],
                model_series["y_hat"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=3 + i,
            )

        truth_max = truth["mv_beds"].max() if not truth.empty else 1.0
        ax.set_ylim(0, truth_max * 1.4)
        ax.set_title(region_name, fontsize=9, pad=2)
        ax.set_xlim(test_start, test_end)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="both", labelsize=7.5)
        ax.grid(True, alpha=0.22, linewidth=0.4)

    axes[0].set_ylabel("MV beds occupied", fontsize=9)
    axes[4].set_ylabel("MV beds occupied", fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0], color=TRUTH_COLOUR, linewidth=1.5, label="Realised MV beds")
    ]
    for style in FORECASTER_STYLES:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=max(style["linewidth"], 1.1),
                label=style["label"],
            )
        )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95), h_pad=1.0, w_pad=0.8)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.8,
        columnspacing=1.5,
    )
    fig.suptitle(
        f"Rolling-origin h = {horizon}-day forecasts on the Omicron test period",
        fontsize=10.5,
        y=0.985,
    )
    return save_figure(fig, "fig_forecast_panel")


def build_pinn_arima_ci_figure() -> Path:
    """Write a focused PinnGRU comparison with ARIMA and GRU uncertainty intervals."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from evaluation.figures import (
        FORECASTER_STYLES,
        FULL_WIDTH_IN,
        apply_paper_style,
        save_figure,
    )

    uncertainty_path = OUT_DIR / "table1_uncertainty.csv"
    tests_path = OUT_DIR / "table1_paired_tests.csv"
    if not uncertainty_path.exists() or not tests_path.exists():
        build_uncertainty_tables()
    else:
        existing_tests = pd.read_csv(tests_path)
        if "comparator_key" not in existing_tests.columns or len(existing_tests) < 8:
            build_uncertainty_tables()

    uncertainty = pd.read_csv(uncertainty_path)
    tests = pd.read_csv(tests_path)
    style_by_model = {style["model"]: style for style in FORECASTER_STYLES}
    models = [
        ("arima_per_region", "ARIMA", style_by_model["arima_per_region"]),
        ("gru_per_region", "GRU per region", style_by_model["gru_per_region"]),
        ("pinn_gru", "PinnGRU", style_by_model["pinn_gru"]),
    ]

    apply_paper_style()
    fig, (ax_rmse, ax_diff) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH_IN, 3.2), layout="constrained",
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )

    offsets = {"arima_per_region": -1.0, "gru_per_region": 0.0, "pinn_gru": 1.0}
    for model_key, label, style in models:
        sub = uncertainty[uncertainty["model_key"] == model_key].sort_values("horizon")
        x = sub["horizon"].to_numpy(dtype=float) + offsets[model_key]
        y = sub["rmse"].to_numpy(dtype=float)
        yerr = np.vstack([
            y - sub["rmse_ci_low"].to_numpy(dtype=float),
            sub["rmse_ci_high"].to_numpy(dtype=float) - y,
        ])
        ax_rmse.errorbar(
            x, y, yerr=yerr, marker="o", capsize=3.0,
            color=style["color"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], label=label,
        )

    ax_rmse.set_xticks(HORIZONS)
    ax_rmse.set_xlabel("Forecast horizon (days)")
    ax_rmse.set_ylabel("RMSE (MV beds)")
    ax_rmse.set_title("RMSE with 95% origin-bootstrap CI", pad=4)
    ax_rmse.legend(frameon=False, loc="upper left")

    ax_diff.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
    diff_specs = [
        ("arima_per_region", "PinnGRU - ARIMA", -0.45),
        ("gru_per_region", "PinnGRU - neural GRU", 0.45),
    ]
    for comparator_key, label, x_offset in diff_specs:
        sub = tests[tests["comparator_key"] == comparator_key].sort_values("horizon")
        style = style_by_model[comparator_key]
        x = sub["horizon"].to_numpy(dtype=float) + x_offset
        diff = sub["mean_origin_rmse_diff"].to_numpy(dtype=float)
        yerr = np.vstack([
            diff - sub["diff_ci_low"].to_numpy(dtype=float),
            sub["diff_ci_high"].to_numpy(dtype=float) - diff,
        ])
        ax_diff.errorbar(
            x, diff, yerr=yerr, marker="o", capsize=3.0,
            color=style["color"], linestyle=style["linestyle"],
            linewidth=max(float(style["linewidth"]), 1.2), label=label,
        )
    ax_diff.set_xticks(HORIZONS)
    ax_diff.set_xlabel("Forecast horizon (days)")
    ax_diff.set_ylabel("Paired RMSE difference (MV beds)")
    ax_diff.set_title("Paired differences across origins", pad=4)
    ax_diff.legend(frameon=False, loc="upper left")
    ax_diff.annotate(
        "negative favours PinnGRU",
        xy=(0.03, 0.08), xycoords="axes fraction",
        fontsize=7.5, color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.2},
    )

    return save_figure(fig, "fig_pinn_arima_gru_ci", close=True)


def build_forecast_figure_main() -> int:
    out = build_forecast_panel_figure()
    print(f"Wrote {out}")
    return 0


def build_pinn_arima_ci_figure_main() -> int:
    out = build_pinn_arima_ci_figure()
    print(f"Wrote {out}")
    return 0


def print_paper_sources() -> int:
    """Print the CSVs that should be treated as manuscript source tables."""
    sources = [
        OUT_DIR / "table1_paper.csv",
        OUT_DIR / "table_metrics.csv",
        OUT_DIR / "table_metrics_detail.csv",
        OUT_DIR / "table1_uncertainty.csv",
        OUT_DIR / "table1_paired_tests.csv",
        ROOT / "results" / "allocation" / "table2_allocation.csv",
    ]
    for path in sources:
        status = "exists" if path.exists() else "missing"
        print(f"{status:7s} {path.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="?",
        default="all",
        choices=(
            "all", "metrics", "table1", "uncertainty", "figure",
            "pinn-arima-ci", "sources",
        ),
        help="Forecast evaluation artifact to build or inspect.",
    )
    args = parser.parse_args(argv)

    if args.artifact == "metrics":
        return build_metric_tables_main()
    if args.artifact == "table1":
        return build_paper_table1_main()
    if args.artifact == "figure":
        return build_forecast_figure_main()
    if args.artifact == "pinn-arima-ci":
        return build_pinn_arima_ci_figure_main()
    if args.artifact == "uncertainty":
        return build_uncertainty_tables_main()
    if args.artifact == "sources":
        return print_paper_sources()

    build_metric_tables_main()
    build_paper_table1_main()
    build_uncertainty_tables_main()
    build_pinn_arima_ci_figure_main()
    build_forecast_figure_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
