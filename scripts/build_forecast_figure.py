"""Build the forecasting result figure (Figure 5 in the paper draft).

Reads ``results/forecasting/forecasts.parquet`` and renders a 7-panel
figure (one per NHS region) of the rolling-origin h=14 forecasts produced
by the four forecasters, overlaid on the realised MV-bed trajectory.

Why h=14:
    Two weeks ahead is the operational lead time for NHS surge decisions
    (mutual-aid transfer, capacity expansion). A shorter horizon (h=7)
    closes too tightly around truth for the differences between models
    to be visible; a longer horizon (h=28) widens the uncertainty band so
    much that the central tendency is washed out.

Lines per panel:
    truth (full daily series in test period)   black solid
    PinnGRU q50 at h=14 with q10/q90 band      coloured solid + filled band
    ARIMA point forecast at h=14               dashed
    GRUPerRegion point forecast at h=14        dash-dot
    SeasonalNaive(7) at h=14                   dotted

The figure is rendered for the Springer LNNS full-width column.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.figures import (
    FULL_WIDTH_IN,
    REGION_PALETTE,
    apply_paper_style,
    save_figure,
)

# ---------------------------------------------------------------------------

HORIZON = 14  # operational two-week lead time
REGION_ORDER = ("Y56", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63")

MODEL_STYLES = {
    "pinn_gru": dict(color="#0072B2", linestyle="-", linewidth=1.6, label="PinnGRU (q50)"),
    "arima_per_region": dict(color="#D55E00", linestyle="--", linewidth=1.0, label="ARIMA"),
    "gru_per_region":  dict(color="#009E73", linestyle="-.", linewidth=1.0, label="GRU"),
    "seasonal_naive":   dict(color="#666666", linestyle=":",  linewidth=1.0, label="Seasonal-naive(7)"),
}


def main() -> int:
    apply_paper_style()
    forecasts_path = ROOT / "results" / "forecasting" / "forecasts.parquet"
    daily_path = ROOT / "data" / "processed" / "regional_daily.csv"

    fc = pd.read_parquet(forecasts_path)
    fc = fc[fc["horizon"] == HORIZON].copy()
    fc["target_date"] = fc["origin"] + pd.Timedelta(days=HORIZON - 1)

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

    fig, axes = plt.subplots(
        2, 4, figsize=(FULL_WIDTH_IN, 4.6), layout="constrained",
        sharex=True, sharey=False,
    )
    axes_flat = axes.flatten()

    for ax, code in zip(axes_flat[: len(REGION_ORDER)], REGION_ORDER):
        region_name = code_to_name[code]

        truth = daily[daily["region_code"] == code].sort_values("date")
        ax.plot(truth["date"], truth["mv_beds"], color="black", linewidth=1.0,
                label="Truth", zorder=4)

        sub = fc[fc["region"] == region_name].sort_values("target_date")

        # PinnGRU uncertainty band first (so it sits behind the lines).
        pinn = sub[sub["model"] == "pinn_gru"]
        if not pinn.empty:
            ax.fill_between(
                pinn["target_date"], pinn["q_lo"], pinn["q_hi"],
                color="#0072B2", alpha=0.18, linewidth=0, zorder=1,
                label="PinnGRU 80% band",
            )

        for model_name, style in MODEL_STYLES.items():
            ms = sub[sub["model"] == model_name].sort_values("target_date")
            if ms.empty:
                continue
            ax.plot(
                ms["target_date"], ms["y_hat"],
                color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], marker="o", markersize=3,
                zorder=3,
            )

        ax.set_title(region_name, fontsize=9)
        ax.set_xlim(test_start, test_end)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # Hide the unused 8th slot; place the global legend there.
    for ax in axes_flat[len(REGION_ORDER):]:
        ax.set_axis_off()
    legend_ax = axes_flat[-1]
    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.2, label="Truth"),
        plt.Rectangle((0, 0), 1, 1, color="#0072B2", alpha=0.18,
                      label="PinnGRU 80% band"),
    ]
    for style in MODEL_STYLES.values():
        legend_handles.append(plt.Line2D(
            [0], [0], color=style["color"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], marker="o", markersize=3,
            label=style["label"],
        ))
    legend_ax.legend(handles=legend_handles, frameon=False, fontsize=8,
                     loc="center", title=f"h = {HORIZON}-day forecast",
                     title_fontsize=9)

    fig.supylabel("MV beds occupied", fontsize=9)
    fig.suptitle(
        f"Omicron-test rolling-origin {HORIZON}-day forecasts by NHS region",
        fontsize=10,
    )

    out = save_figure(fig, "fig_forecast_panel")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
