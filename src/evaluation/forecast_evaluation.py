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


def build_forecast_figure_main() -> int:
    out = build_forecast_panel_figure()
    print(f"Wrote {out}")
    return 0


def print_paper_sources() -> int:
    """Print the CSVs that should be treated as manuscript source tables."""
    sources = [
        OUT_DIR / "table1_paper.csv",
        OUT_DIR / "table_metrics.csv",
        OUT_DIR / "table_metrics_detail.csv",
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
        choices=("all", "metrics", "table1", "figure", "sources"),
        help="Forecast evaluation artifact to build or inspect.",
    )
    args = parser.parse_args(argv)

    if args.artifact == "metrics":
        return build_metric_tables_main()
    if args.artifact == "table1":
        return build_paper_table1_main()
    if args.artifact == "figure":
        return build_forecast_figure_main()
    if args.artifact == "sources":
        return print_paper_sources()

    build_metric_tables_main()
    build_paper_table1_main()
    build_forecast_figure_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
