"""Build the forecasting metric table from saved rolling-origin forecasts.

Reads ``results/forecasting/forecasts.parquet`` (produced by
``scripts/train_forecasters.py``) and emits two CSVs:

  table_metrics_detail.csv     per (model, region, horizon) — full panel
  table_metrics.csv            macro-averaged per (model, horizon) and overall

Metric panel (operationally-aligned — see ``02_METHODOLOGY.md`` Sec 4.1):

These are standard metrics from forecasting and operations-research literature.
None of them are novel; the methodological choice is which subset to report,
chosen to *align with the decision-aware composite loss* the paper proposes.

  Point accuracy (sklearn):
    - MAE    sklearn.metrics.mean_absolute_error
    - RMSE   sklearn.metrics.root_mean_squared_error
    - MAPE   sklearn.metrics.mean_absolute_percentage_error
             (safe because MV-bed occupancy is strictly positive)
  Decision-asymmetric (custom — measures what the asymmetric-under-prediction
  term of the composite loss targets):
    - Underestimation rate    fraction of forecasts strictly below truth
    - Expected shortage       sum of max(0, y_true - y_hat) in bed-days
                              (= total under-provisioned demand if the
                              forecast were used directly for capacity)
  Trajectory (custom — standard in epidemic-forecasting hubs):
    - Peak error              |max(y_true) - max(y_hat)| at each fixed horizon
    - Peak timing error       |argmax(y_true) - argmax(y_hat)| in days
  Probabilistic calibration (only models that emit q10/q90):
    - WIS at the 80% interval (Bracher et al. 2021)
    - Pinball loss at q10 and q90 via sklearn.metrics.mean_pinball_loss

Deliberately omitted:
  - sMAPE: information overlap with MAE+MAPE; not in sklearn.
  - MASE: useful when comparing across datasets with different scales;
          we have one dataset and report per-region MAE+MAPE already.
  - R^2:  uninformative on heavy-tailed surge data.

Macro-averaging rule: per-region metrics are averaged with equal weights so a
large-population region (London) does not dominate the headline panel. The
exception is expected shortage, which is the planner's *total* under-provision
and is therefore summed across regions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    root_mean_squared_error,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.forecast_metrics import (
    expected_shortage,
    peak_error,
    peak_timing_error,
    underestimation_rate,
    wis,
)

OUT_DIR = ROOT / "results" / "forecasting"
FORECASTS = OUT_DIR / "forecasts.parquet"


def per_region_panel(y_true: np.ndarray, y_hat: np.ndarray,
                     q_lo: np.ndarray | None,
                     q_hi: np.ndarray | None) -> dict[str, float]:
    """All metrics for a single (model, region, horizon) trajectory."""
    out: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_hat)),
        "rmse": float(root_mean_squared_error(y_true, y_hat)),
        "mape": float(mean_absolute_percentage_error(y_true, y_hat)),
        "underestimation_rate": underestimation_rate(y_true, y_hat),
        "expected_shortage": expected_shortage(y_true, y_hat),
        "peak_error": peak_error(y_true, y_hat),
        "peak_timing_error": peak_timing_error(y_true, y_hat),
    }
    if q_lo is not None and q_hi is not None:
        # 80% prediction interval -> alpha=0.20 -> level=0.10 (single pair).
        qp = np.stack([np.stack([q_lo, q_hi], axis=0)], axis=0)  # (K=1, 2, N)
        out["wis_80"] = wis(y_true, y_hat, [0.10], qp)
        out["pinball_q10"] = float(mean_pinball_loss(y_true, q_lo, alpha=0.10))
        out["pinball_q90"] = float(mean_pinball_loss(y_true, q_hi, alpha=0.90))
    else:
        out["wis_80"] = float("nan")
        out["pinball_q10"] = float("nan")
        out["pinball_q90"] = float("nan")
    return out


def main() -> int:
    if not FORECASTS.exists():
        raise FileNotFoundError(
            f"{FORECASTS} not found. Run scripts/train_forecasters.py first."
        )
    forecasts = pd.read_parquet(FORECASTS)
    print(f"Loaded {len(forecasts):,} forecasts from {FORECASTS.name}")
    print(f"Models: {forecasts['model'].unique().tolist()}")
    print(f"Horizons: {sorted(forecasts['horizon'].unique().tolist())}")
    print(f"Regions: {forecasts['region'].nunique()}")
    print(f"Origins: {forecasts['origin'].nunique()}\n")

    detail_records: list[dict] = []
    for (model_name, h, region), sub in forecasts.groupby(
        ["model", "horizon", "region"]
    ):
        sub = sub.sort_values("origin")
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_hat = sub["y_hat"].to_numpy(dtype=float)
        has_q = sub["q_lo"].notna().all() and sub["q_hi"].notna().all()
        q_lo = sub["q_lo"].to_numpy(dtype=float) if has_q else None
        q_hi = sub["q_hi"].to_numpy(dtype=float) if has_q else None
        panel = per_region_panel(y_true, y_hat, q_lo, q_hi)
        detail_records.append({
            "model": model_name, "horizon": int(h), "region": region,
            "n": len(sub), **panel,
        })
    detail = pd.DataFrame(detail_records)
    detail.to_csv(OUT_DIR / "table_metrics_detail.csv", index=False)
    print(f"Wrote per-region detail -> {OUT_DIR / 'table_metrics_detail.csv'}")

    # Macro-aggregation across regions. Mean for all metrics except
    # expected_shortage which is summed (total under-provisioned bed-days).
    agg_funcs = {
        "mae": "mean", "rmse": "mean", "mape": "mean",
        "underestimation_rate": "mean", "expected_shortage": "sum",
        "peak_error": "mean", "peak_timing_error": "mean",
        "wis_80": "mean", "pinball_q10": "mean", "pinball_q90": "mean",
    }
    by_h = detail.groupby(["model", "horizon"]).agg(agg_funcs).reset_index()
    by_all = (
        detail.groupby(["model"]).agg(agg_funcs).reset_index().assign(horizon="all")
    )
    table = pd.concat([by_h, by_all[by_h.columns]], ignore_index=True)
    table.to_csv(OUT_DIR / "table_metrics.csv", index=False)
    print(f"Wrote headline table -> {OUT_DIR / 'table_metrics.csv'}\n")

    overall = (
        table[table["horizon"] == "all"]
        .set_index("model")[
            ["mae", "rmse", "mape", "underestimation_rate",
             "expected_shortage", "peak_error", "peak_timing_error", "wis_80"]
        ]
        .round(3)
    )
    print("=== Overall (macro-avg across regions, avg over horizons) ===")
    print(overall.to_string())

    per_h_mae = (
        table[table["horizon"] != "all"]
        .pivot(index="model", columns="horizon", values="mae")
        .round(3)
    )
    print("\n=== MAE per horizon (macro-avg across regions) ===")
    print(per_h_mae.to_string())

    per_h_under = (
        table[table["horizon"] != "all"]
        .pivot(index="model", columns="horizon", values="underestimation_rate")
        .round(3)
    )
    print("\n=== Underestimation rate per horizon (macro-avg) ===")
    print(per_h_under.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
