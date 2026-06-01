"""Per-origin bias recalibration + horizon-adaptive persistence blend for the
PinnGRU point forecast.

The raw PinnGRU point forecast systematically over-predicts, and the bias grows
with horizon (a regime-shift effect: trained on high-demand Alpha/Delta, applied
frozen to mild Omicron). Two leakage-free corrections fix this:

  1. Per-origin bias recalibration. At each rolling origin ``t`` and horizon
     ``h``, subtract the mean of the model's last ``K`` h-step errors whose
     target date is already observed (``< t``). Uses only information available
     at ``t`` -- the adaptation ARIMA gets for free from per-origin refitting.

  2. Horizon-adaptive persistence blend. Blend the recalibrated point forecast
     with seasonal-naive(7), with a per-horizon weight chosen by an expanding
     window over realised past origins (pooled across regions). Long horizons
     lean on persistence, which is the strongest simple predictor there.

Both steps are strictly leakage-free (no test-set tuning). Only the point
forecast (``y_hat``) is changed; the ``q_lo``/``q_hi`` quantile interval is left
as produced by the quantile heads, so the downstream allocation is unaffected.

Output: ``results/forecasting/forecasts_pinn_gru_cal.parquet`` (model key
``pinn_gru_cal``), registered as the proposed model in
``evaluation.forecast_evaluation``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import results_dir

K = 3
ALPHAS = np.linspace(0.0, 1.0, 11)
MIN_REALISED = 7  # min realised origin-region pairs before fitting a blend weight
OUT = results_dir("forecasting")


def recalibrate(pinn: pd.DataFrame, snaive: pd.DataFrame) -> pd.DataFrame:
    """Return a ``pinn_gru_cal`` forecast frame: recalibrated + blended y_hat."""
    pinn = pinn.copy()
    pinn["origin"] = pd.to_datetime(pinn["origin"])
    sn = snaive[["origin", "region", "horizon", "y_hat"]].copy()
    sn["origin"] = pd.to_datetime(sn["origin"])
    sn = sn.rename(columns={"y_hat": "sn"})

    # --- Step 1: per-(region, horizon) rolling bias recalibration -----------
    parts = []
    for (_region, h), g in pinn.groupby(["region", "horizon"]):
        g = g.sort_values("origin").reset_index(drop=True)
        o = g["origin"].values
        err = (g["y_hat"] - g["y_true"]).values
        yh = g["y_hat"].values.copy()
        recal = yh.copy()
        for i, t in enumerate(o):
            usable = [err[j] for j in range(i)
                      if o[j] + np.timedelta64(int(h) - 1, "D") < t]
            if usable:
                recal[i] = yh[i] - float(np.mean(usable[-K:]))
        g["recal"] = recal
        parts.append(g)
    df = pd.concat(parts).merge(sn, on=["origin", "region", "horizon"], how="left")

    # --- Step 2: per-horizon expanding-window blend weight ------------------
    outs = []
    for h in sorted(df["horizon"].unique()):
        d = df[df["horizon"] == h].copy()
        d["y_hat_cal"] = d["recal"]
        for t in np.sort(d["origin"].unique()):
            realised = d[(d["origin"].values + np.timedelta64(int(h) - 1, "D")) < t]
            alpha = 1.0
            if len(realised) >= MIN_REALISED:
                alpha = float(min(ALPHAS, key=lambda a: np.mean(
                    (a * realised["recal"].values
                     + (1 - a) * realised["sn"].values
                     - realised["y_true"].values) ** 2)))
            cur = d["origin"].values == t
            d.loc[cur, "y_hat_cal"] = (
                alpha * d.loc[cur, "recal"].values
                + (1 - alpha) * d.loc[cur, "sn"].values
            )
        outs.append(d)
    df = pd.concat(outs)

    out = df[["model", "origin", "region", "horizon",
              "y_hat_cal", "q_lo", "q_hi", "y_true"]].copy()
    out = out.rename(columns={"y_hat_cal": "y_hat"})
    out["model"] = "pinn_gru_cal"
    return out.sort_values(["origin", "region", "horizon"]).reset_index(drop=True)


def main() -> int:
    pinn = pd.read_parquet(OUT / "forecasts_pinn_gru.parquet")
    sn = pd.read_parquet(OUT / "forecasts_seasonal_naive.parquet")
    cal = recalibrate(pinn, sn)
    path = OUT / "forecasts_pinn_gru_cal.parquet"
    cal.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(cal)} rows)")
    for h in sorted(cal["horizon"].unique()):
        d = cal[cal["horizon"] == h]
        e = d["y_hat"].values - d["y_true"].values
        print(f"  h={h:2d}: RMSE={np.sqrt(np.mean(e**2)):6.2f}  "
              f"bias={np.mean(e):6.2f}  under%={100*np.mean(e<0):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
