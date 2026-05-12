#!/usr/bin/env python3
"""Exploratory data analysis for the UKCI 2026 NHS critical-care surge paper.

Loads the tidy regional CSV produced by ``scripts/build_regional_dataset.py``
and emits paper-quality figures into ``figures/`` plus summary tables into
``results/eda/``.

Artefacts produced:

==================================  ============================================
Output                              Role in the paper
==================================  ============================================
fig_regional_mv_beds.{pdf,png}      §5 — headline MV-bed time series (target)
fig_metric_overview.{pdf,png}       §3 — all four observed metrics per region
fig_regional_distributions.{pdf,p}  Appendix — per-region boxplot distribution
fig_peak_alignment.{pdf,png}        §5 / discussion — cross-region peak timing
fig_data_quality.{pdf,png}          §3 / appendix — completeness summary
fig_wave_overlay.{pdf,png}          §5 — wave-stratified MV-bed trace
fig_autocorrelation.{pdf,png}       §4 — national ACF + PACF, motivates GRU window
fig_regional_acf.{pdf,png}          §4 / appendix — per-region ACF (robustness)
fig_lead_lag.{pdf,png}              §4 — admissions→MV beds lag structure
fig_weekly_seasonality.{pdf,png}    §4 — day-of-week reporting effect
fig_mobility_overlay.{pdf,png}      §3 / §4 — Mobility (lagged 21d) vs MV
fig_region_context.{pdf,png}        §3 / §5 — population + IMD per region
table_regional_summary.csv          Appendix Table A1 — region × wave statistics
table_wave_summary.csv              §5 inline numbers — per-wave aggregate stats
==================================  ============================================

Run from the repository root:

    python scripts/run_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.figures import (  # noqa: E402
    COLUMN_WIDTH_IN,
    FULL_WIDTH_IN,
    REGION_PALETTE,
    WAVE_COLOURS,
    WAVE_PERIODS,
    apply_paper_style,
    overlay_wave_bands,
    save_figure,
)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REGIONAL_CSV = REPO_ROOT / "data" / "processed" / "regional_daily.csv"
MOBILITY_CSV = REPO_ROOT / "data" / "processed" / "regional_mobility.csv"
STATIC_CSV = REPO_ROOT / "data" / "processed" / "regional_static.csv"
EDA_OUT = REPO_ROOT / "results" / "eda"

#: Stable region-code ordering for the colour palette and legend.
REGION_ORDER: tuple[str, ...] = ("Y56", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tidy_csv(path: Path = REGIONAL_CSV) -> pd.DataFrame:
    """Load the tidy regional CSV.

    Args:
        path: Path to the regional-daily CSV.

    Returns:
        DataFrame with parsed ``date`` and a categorical ``region_name``
        ordered by ``REGION_ORDER``.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    region_lookup = (
        df[["region_code", "region_name"]].drop_duplicates().set_index("region_code")[
            "region_name"
        ]
    )
    region_name_order = [region_lookup[code] for code in REGION_ORDER]
    df["region_name"] = pd.Categorical(
        df["region_name"], categories=region_name_order, ordered=True
    )
    df = df.sort_values(["region_name", "date"]).reset_index(drop=True)
    return df


def assign_wave(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``wave`` column using the canonical wave boundaries.

    Args:
        df: DataFrame with a ``date`` column.

    Returns:
        A copy of ``df`` with a categorical ``wave`` column.
    """
    out = df.copy()
    out["wave"] = pd.Categorical(
        ["unassigned"] * len(out), categories=list(WAVE_PERIODS) + ["unassigned"]
    )
    for label, (start, end) in WAVE_PERIODS.items():
        mask = (out["date"] >= pd.Timestamp(start)) & (out["date"] < pd.Timestamp(end))
        out.loc[mask, "wave"] = label
    return out


# ---------------------------------------------------------------------------
# Figure: regional MV-bed time series (headline §5 figure)
# ---------------------------------------------------------------------------


def figure_regional_mv_beds(df: pd.DataFrame) -> Path:
    """Draw the headline §5 figure: MV-bed occupancy by NHS region over the
    full daily-publication period, with wave bands behind the traces.

    Uses a two-panel layout (linear-scale upper, log-scale lower) so the
    London-dominated wave 1 and the smaller wave 2/3 dynamics in other
    regions are both readable. Legend sits below to keep the data clear.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(FULL_WIDTH_IN, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
        layout="constrained",
    )
    region_names = list(df["region_name"].cat.categories)
    for ax in (ax_top, ax_bot):
        overlay_wave_bands(ax)
    for region_name, colour in zip(region_names, REGION_PALETTE, strict=False):
        sub = df[df["region_name"] == region_name]
        ax_top.plot(sub["date"], sub["mv_beds"], label=region_name, color=colour, linewidth=1.0)
        # On log scale, mask zeros so the trace does not drop to ylim=1 artefactually.
        mv_log = sub["mv_beds"].where(sub["mv_beds"] > 0)
        ax_bot.plot(sub["date"], mv_log, label=region_name, color=colour, linewidth=1.0)
    ax_top.set_ylabel("MV beds (linear)")
    ax_bot.set_ylabel("MV beds (log)")
    ax_bot.set_yscale("log")
    ax_bot.set_ylim(bottom=1)
    ax_bot.set_xlabel("Date")
    ax_top.set_title(
        "NHS England mechanical-ventilation occupancy by region, 1 Aug 2020 – 31 Aug 2022"
    )
    ax_bot.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    handles, labels = ax_top.get_legend_handles_labels()
    wave_idx = [i for i, lab in enumerate(labels) if lab in WAVE_PERIODS]
    region_idx = [i for i in range(len(labels)) if i not in wave_idx]
    ordered = [handles[i] for i in region_idx + wave_idx]
    ordered_labels = [labels[i] for i in region_idx + wave_idx]
    fig.legend(
        ordered,
        ordered_labels,
        ncol=5,
        loc="outside lower center",
        frameon=False,
        fontsize=8,
    )
    return save_figure(fig, "fig_regional_mv_beds")


# ---------------------------------------------------------------------------
# Figure: per-region distributions (boxplot)
# ---------------------------------------------------------------------------


def figure_regional_distributions(df: pd.DataFrame) -> Path:
    """Per-region MV-bed distribution rendered as horizontal boxplots.

    Horizontal layout avoids x-axis label collisions and gives the reader
    a clear per-region magnitude ordering for §5.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 3.2))
    region_names = list(df["region_name"].cat.categories)
    grouped = [df.loc[df["region_name"] == r, "mv_beds"].dropna().values for r in region_names]
    positions = list(range(len(region_names), 0, -1))
    bp = ax.boxplot(
        grouped,
        positions=positions,
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    for patch, colour in zip(bp["boxes"], REGION_PALETTE, strict=False):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.1)
    ax.set_yticks(positions)
    ax.set_yticklabels(region_names)
    ax.set_xlabel("MV beds occupied")
    ax.set_title("Distribution of MV occupancy by NHS region")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return save_figure(fig, "fig_regional_distributions")


# ---------------------------------------------------------------------------
# Figure: peak alignment across regions
# ---------------------------------------------------------------------------


def figure_peak_alignment(df: pd.DataFrame) -> Path:
    """For each region, plot the date of the rolling-7-day-mean peak per
    wave. Reveals whether waves swept across regions simultaneously or with
    lag. Useful for the §5 discussion of whether per-region GRU is enough.
    """
    enriched = assign_wave(df)
    records: list[dict] = []
    for (region_name, wave), sub in enriched.groupby(
        ["region_name", "wave"], observed=True
    ):
        if wave == "unassigned" or sub.empty:
            continue
        smoothed = sub.set_index("date")["mv_beds"].rolling(7, min_periods=4).mean()
        peak_date = smoothed.idxmax()
        peak_value = float(smoothed.max())
        records.append(
            {
                "region_name": region_name,
                "wave": wave,
                "peak_date": peak_date,
                "peak_value": peak_value,
            }
        )
    peaks = pd.DataFrame.from_records(records)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.4))
    overlay_wave_bands(ax)
    region_names = list(df["region_name"].cat.categories)
    max_peak = float(peaks["peak_value"].max())
    ax.set_ylim(-0.7, len(region_names) - 0.3)
    for region_name, colour in zip(region_names, REGION_PALETTE, strict=False):
        sub = peaks[peaks["region_name"] == region_name]
        sizes = 60 + 540 * np.sqrt(sub["peak_value"].values / max_peak)
        ax.scatter(
            sub["peak_date"],
            [region_name] * len(sub),
            s=sizes,
            color=colour,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        for date, name, value in zip(sub["peak_date"], [region_name] * len(sub), sub["peak_value"]):
            ax.annotate(
                f"{int(value)}",
                xy=(date, name),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )
    ax.set_xlabel("Date of regional wave peak (7-day rolling mean)")
    ax.set_yticks(range(len(region_names)))
    ax.set_yticklabels(region_names)
    ax.set_xlim(pd.Timestamp("2020-08-01"), pd.Timestamp("2022-09-01"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Per-region wave peaks (marker area ∝ peak MV beds; label = bed count)")
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return save_figure(fig, "fig_peak_alignment")


# ---------------------------------------------------------------------------
# Figure: missingness heatmap
# ---------------------------------------------------------------------------


def figure_data_quality(df: pd.DataFrame) -> Path:
    """Data-quality summary across the four numeric metrics.

    Left panel: completeness per region × metric (% non-missing). Right
    panel: count of missing days per metric across the full panel. Both
    are designed to surface real problems quickly rather than flood the
    page with near-zero heatmap cells (every column is well above 99%
    complete; a uniform heatmap is uninformative).
    """
    metrics = ("admissions", "hospital_cases", "occupied_beds", "mv_beds")
    region_names = list(df["region_name"].cat.categories)
    completeness = np.zeros((len(region_names), len(metrics)), dtype=float)
    for i, region in enumerate(region_names):
        sub = df[df["region_name"] == region]
        for j, metric in enumerate(metrics):
            completeness[i, j] = 1.0 - float(sub[metric].isna().mean())
    missing_total = {m: int(df[m].isna().sum()) for m in metrics}

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH_IN, 3.0),
        gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.05},
        layout="constrained",
    )
    vmin = max(0.0, float(completeness.min()) - 0.005)
    im = ax_l.imshow(
        completeness * 100.0,
        aspect="auto",
        cmap="Greens",
        vmin=vmin * 100.0,
        vmax=100.0,
    )
    ax_l.set_xticks(range(len(metrics)))
    ax_l.set_xticklabels(metrics, rotation=20, ha="right", fontsize=8)
    ax_l.set_yticks(range(len(region_names)))
    ax_l.set_yticklabels(region_names, fontsize=8)
    ax_l.grid(False)
    for i in range(len(region_names)):
        for j in range(len(metrics)):
            ax_l.text(
                j, i, f"{completeness[i, j] * 100:.1f}",
                ha="center", va="center", fontsize=7, color="black",
            )
    ax_l.set_title("Completeness (% non-missing) by region x metric", fontsize=9)
    fig.colorbar(im, ax=ax_l, label="% non-missing", pad=0.02, location="bottom",
                 shrink=0.7, aspect=30)

    positions = list(range(len(metrics), 0, -1))
    bars = ax_r.barh(
        positions,
        [missing_total[m] for m in metrics],
        color="#c0504d",
        edgecolor="black",
        height=0.55,
    )
    ax_r.set_yticks(positions)
    ax_r.set_yticklabels(metrics, fontsize=8)
    ax_r.set_xlabel("Missing rows (out of 5,327)")
    ax_r.set_title("Missing-row count by metric", fontsize=9)
    for bar, m in zip(bars, metrics):
        ax_r.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(missing_total[m]), va="center", fontsize=7,
        )
    return save_figure(fig, "fig_data_quality")


# ---------------------------------------------------------------------------
# Figure: wave overlay (single region, comparative wave profile)
# ---------------------------------------------------------------------------


def figure_wave_overlay(df: pd.DataFrame) -> Path:
    """Per-wave overlay of MV-bed occupancy along ``days since wave start``.

    Aggregates the daily national total across all 7 NHS regions and aligns
    each wave on a common day-zero axis. Marker shows the wave's *interior*
    peak (computed on the 7-day rolling mean to avoid noise-driven local
    maxima at the boundary, where the prior wave's tail can dominate).
    """
    enriched = assign_wave(df)
    line_palette = {
        "Alpha + early vaccination": "#1f77b4",
        "Delta": "#d95f02",
        "Omicron and beyond": "#2ca02c",
    }
    edge_buffer = 14
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.0))
    for label in WAVE_PERIODS:
        sub = enriched[enriched["wave"] == label]
        if sub.empty:
            continue
        daily_total = (
            sub.groupby("date", observed=True)["mv_beds"].sum().sort_index()
        )
        days = np.arange(len(daily_total))
        colour = line_palette[label]
        ax.plot(days, daily_total.values, color=colour, label=label, linewidth=1.6)
        smoothed = daily_total.rolling(7, center=True, min_periods=4).mean().values
        interior = smoothed.copy()
        if len(interior) > 2 * edge_buffer:
            interior[:edge_buffer] = np.nan
            interior[-edge_buffer:] = np.nan
        peak_idx = int(np.nanargmax(interior))
        peak_value = int(daily_total.values[peak_idx])
        ax.scatter(
            [days[peak_idx]], [peak_value],
            color=colour, edgecolor="black", linewidth=0.5, s=45, zorder=3,
        )
        ax.annotate(
            f"peak {peak_value}",
            xy=(days[peak_idx], peak_value),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=7,
            color=colour,
        )
    ax.set_xlabel("Days since wave start")
    ax.set_ylabel("National MV beds occupied")
    ax.set_title("Wave profile comparison (sum across 7 NHS regions, interior peaks marked)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    fig.tight_layout()
    return save_figure(fig, "fig_wave_overlay")


# ---------------------------------------------------------------------------
# Figure: all four observed metrics per region (§3 Data Description)
# ---------------------------------------------------------------------------


def figure_metric_overview(df: pd.DataFrame) -> Path:
    """2x2 panel showing every observed metric the model consumes.

    The PINN-SEIRD model uses ``admissions``, ``hospital_cases``,
    ``occupied_beds``, and ``mv_beds`` (the forecasting target). This figure
    gives the reader the full input set at a glance.
    """
    metrics = [
        ("admissions", "Daily COVID-19 admissions"),
        ("hospital_cases", "General hospitalisations (H)"),
        ("occupied_beds", "Occupied COVID beds (H + C)"),
        ("mv_beds", "Mechanical-ventilation beds (C, target)"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(FULL_WIDTH_IN, 4.8), sharex=True,
        layout="constrained",
    )
    axes_flat = axes.flatten()
    region_names = list(df["region_name"].cat.categories)
    for ax, (col, title) in zip(axes_flat, metrics, strict=False):
        overlay_wave_bands(ax, alpha=0.35)
        for region_name, colour in zip(region_names, REGION_PALETTE, strict=False):
            sub = df[df["region_name"] == region_name]
            ax.plot(sub["date"], sub[col], color=colour, linewidth=0.9, label=region_name)
        ax.set_title(title, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0, 0].set_ylabel("Patients / day")
    axes[1, 0].set_ylabel("Beds occupied")
    axes[1, 0].set_xlabel("Date")
    axes[1, 1].set_xlabel("Date")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    wave_idx = [i for i, lab in enumerate(labels) if lab in WAVE_PERIODS]
    region_idx = [i for i in range(len(labels)) if i not in wave_idx]
    ordered = [handles[i] for i in region_idx + wave_idx]
    ordered_labels = [labels[i] for i in region_idx + wave_idx]
    fig.legend(
        ordered, ordered_labels, ncol=5, loc="outside lower center",
        frameon=False, fontsize=8,
    )
    fig.suptitle(
        "NHS England COVID-19 observed metrics per region, 1 Aug 2020 – 31 Aug 2022",
        fontsize=10,
    )
    return save_figure(fig, "fig_metric_overview")


# ---------------------------------------------------------------------------
# Figure: ACF + PACF (§4 model-order motivation)
# ---------------------------------------------------------------------------


def figure_autocorrelation(df: pd.DataFrame) -> Path:
    """ACF and PACF of national MV-bed occupancy (2-panel).

    Drives two modelling choices: the GRU lookback window (where ACF stays
    above the white-noise threshold) and the ARIMA(p, d, q) grid (PACF cut-off
    gives p, ACF cut-off gives q).
    """
    from statsmodels.tsa.stattools import acf, pacf

    max_lag = 35
    national = (
        df.groupby("date", observed=True)["mv_beds"].sum().sort_index()
    )
    national_diff = national.diff().dropna()
    acf_vals = acf(national_diff, nlags=max_lag, fft=True)
    pacf_vals = pacf(national_diff, nlags=max_lag, method="yw")
    ci = 1.96 / np.sqrt(len(national_diff))

    fig, (ax_a, ax_p) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH_IN, 3.0), layout="constrained",
    )
    lags = np.arange(max_lag + 1)
    ax_a.stem(lags, acf_vals, basefmt=" ", linefmt="C0-", markerfmt="C0o")
    ax_a.axhspan(-ci, ci, color="grey", alpha=0.2)
    ax_a.set_title("ACF of $\\Delta$MV beds (national)", fontsize=9)
    ax_a.set_xlabel("Lag (days)")
    ax_a.set_ylabel("ACF")
    ax_p.stem(lags, pacf_vals, basefmt=" ", linefmt="C1-", markerfmt="C1o")
    ax_p.axhspan(-ci, ci, color="grey", alpha=0.2)
    ax_p.set_title("PACF of $\\Delta$MV beds (national)", fontsize=9)
    ax_p.set_xlabel("Lag (days)")
    ax_p.set_ylabel("PACF")
    return save_figure(fig, "fig_autocorrelation")


def figure_regional_acf(df: pd.DataFrame) -> Path:
    """Per-region ACF of $\\Delta$MV beds.

    Companion to :func:`figure_autocorrelation`. Confirms that the lag
    structure motivating the GRU window length is consistent across all
    seven NHS regions, not driven by London alone.
    """
    from statsmodels.tsa.stattools import acf

    max_lag = 35
    region_names = list(df["region_name"].cat.categories)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.2), layout="constrained")
    n_per_region = []
    for region_name, colour in zip(region_names, REGION_PALETTE, strict=False):
        sub_series = (
            df[df["region_name"] == region_name].set_index("date")["mv_beds"]
        ).diff().dropna()
        n_per_region.append(len(sub_series))
        ax.plot(
            np.arange(max_lag + 1),
            acf(sub_series, nlags=max_lag, fft=True),
            color=colour, linewidth=1.2, label=region_name,
        )
    ci = 1.96 / np.sqrt(float(np.mean(n_per_region)))
    ax.axhspan(-ci, ci, color="grey", alpha=0.2,
               label=f"95% white-noise band (n~{int(np.mean(n_per_region))})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Per-region ACF of $\\Delta$MV beds")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    return save_figure(fig, "fig_regional_acf")


# ---------------------------------------------------------------------------
# Figure: lead-lag cross-correlation between covariates and the target
# ---------------------------------------------------------------------------


def figure_lead_lag(df: pd.DataFrame) -> Path:
    """Cross-correlation of admissions / hospital_cases / occupied_beds
    against MV beds at lags 0–28 days.

    A positive lag ``k`` means the covariate at time ``t-k`` correlates with
    MV beds at time ``t``. The peak lag tells us the typical pipeline delay
    through the SEI_aI_sHCRD compartments (admissions → ward → ventilation).
    """
    max_lag = 28
    enriched = df.copy()
    enriched["mv_beds_diff"] = enriched.groupby("region_name", observed=True)[
        "mv_beds"
    ].diff()

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.0), layout="constrained")
    covariate_colours = {
        "admissions": "#0072B2",
        "hospital_cases": "#D55E00",
        "occupied_beds": "#009E73",
    }
    lags = np.arange(0, max_lag + 1)
    for covariate, colour in covariate_colours.items():
        per_region: list[np.ndarray] = []
        for region_name in df["region_name"].cat.categories:
            sub = enriched[enriched["region_name"] == region_name].dropna(
                subset=[covariate, "mv_beds_diff"]
            )
            cov_diff = sub[covariate].diff().dropna()
            tgt_diff = sub["mv_beds_diff"].iloc[1:]
            n = min(len(cov_diff), len(tgt_diff))
            x = cov_diff.iloc[:n].to_numpy()
            y = tgt_diff.iloc[:n].to_numpy()
            corrs = []
            for k in lags:
                if k == 0:
                    a, b = x, y
                else:
                    a, b = x[:-k], y[k:]
                if len(a) < 10 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
                    corrs.append(np.nan)
                else:
                    corrs.append(float(np.corrcoef(a, b)[0, 1]))
            per_region.append(np.array(corrs))
        matrix = np.vstack(per_region)
        mean = np.nanmean(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        q75 = np.nanpercentile(matrix, 75, axis=0)
        ax.fill_between(lags, q25, q75, color=colour, alpha=0.18, linewidth=0)
        ax.plot(lags, mean, color=colour, label=covariate, linewidth=1.4)
        peak_idx = int(np.nanargmax(mean))
        ax.scatter([lags[peak_idx]], [mean[peak_idx]], color=colour, edgecolor="black",
                   linewidth=0.5, s=35, zorder=3)
        ax.annotate(
            f"lag {lags[peak_idx]}d",
            xy=(lags[peak_idx], mean[peak_idx]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color=colour,
        )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Lag k (days), covariate(t-k) vs $\\Delta$MV(t)")
    ax.set_ylabel("Cross-correlation\n(mean; band = IQR over 7 regions)")
    ax.set_title("Lead-lag of covariates to MV-bed change")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_ylim(-0.05, max(ax.get_ylim()[1], 0.6))
    return save_figure(fig, "fig_lead_lag")


# ---------------------------------------------------------------------------
# Figure: day-of-week reporting effect
# ---------------------------------------------------------------------------


def figure_weekly_seasonality(df: pd.DataFrame) -> Path:
    """Day-of-week effect on the four metrics, after removing the 7-day
    rolling mean (i.e. residual relative to the local trend).

    NHS England daily reports are known to dip on Saturdays and Sundays.
    This figure shows the magnitude of that effect — which justifies adding
    a weekly seasonal feature (or de-seasonalisation step) to the forecaster.
    """
    metrics = ("admissions", "hospital_cases", "occupied_beds", "mv_beds")
    enriched = df.copy()
    enriched = enriched.sort_values(["region_name", "date"])
    for metric in metrics:
        enriched[f"{metric}_residual"] = (
            enriched.groupby("region_name", observed=True)[metric]
            .transform(lambda s: s - s.rolling(7, center=True, min_periods=4).mean())
        )
    enriched["dow"] = enriched["date"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.8), layout="constrained")
    metric_colours = {
        "admissions": "#0072B2",
        "hospital_cases": "#D55E00",
        "occupied_beds": "#009E73",
        "mv_beds": "#CC79A7",
    }
    positions = np.arange(len(dow_order))
    bar_width = 0.2
    for i, metric in enumerate(metrics):
        means = [
            float(enriched.loc[enriched["dow"] == d, f"{metric}_residual"].mean())
            for d in dow_order
        ]
        ax.bar(
            positions + (i - 1.5) * bar_width,
            means,
            width=bar_width,
            color=metric_colours[metric],
            label=metric,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([d[:3] for d in dow_order])
    ax.set_ylabel("Residual vs 7-day rolling mean")
    ax.set_title("Day-of-week reporting effect (NHS daily data)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    return save_figure(fig, "fig_weekly_seasonality")


# ---------------------------------------------------------------------------
# Figure: mobility overlay vs MV beds (§3 / §4)
# ---------------------------------------------------------------------------


def figure_mobility_overlay(df: pd.DataFrame) -> Path:
    """Two-panel overlay of Google Mobility (already +21d shifted) and
    MV-bed occupancy per NHS region, on a shared time axis.

    Shows that mobility dips on lockdowns precede MV-bed surges, and
    motivates including mobility as a forecasting covariate.
    """
    mob = pd.read_csv(MOBILITY_CSV, parse_dates=["date"])
    mob = mob[(mob["date"] >= df["date"].min()) & (mob["date"] <= df["date"].max())]
    region_codes = ("Y56", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63")
    code_to_name = (
        df[["region_code", "region_name"]].drop_duplicates().set_index("region_code")[
            "region_name"
        ]
        .to_dict()
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(FULL_WIDTH_IN, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1},
        layout="constrained",
    )
    for ax in (ax_top, ax_bot):
        overlay_wave_bands(ax, alpha=0.35)
    for code, colour in zip(region_codes, REGION_PALETTE, strict=False):
        sub_mv = df[df["region_code"] == code]
        ax_top.plot(sub_mv["date"], sub_mv["mv_beds"], color=colour, linewidth=0.9,
                    label=code_to_name[code])
        sub_mob = mob[mob["nhs_code"] == code].sort_values("date").copy()
        # Smooth out weekday/weekend oscillations so the trend is readable.
        sub_mob["workplaces_smooth"] = (
            sub_mob["workplaces"].rolling(7, center=True, min_periods=4).mean()
        )
        ax_bot.plot(sub_mob["date"], sub_mob["workplaces_smooth"],
                    color=colour, linewidth=1.1)
    ax_top.set_ylabel("MV beds occupied")
    ax_bot.set_ylabel("Workplaces mobility (% vs baseline)")
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.set_xlabel("Date")
    ax_top.set_title(
        "Workplaces mobility (Google, +21d forward shift) vs MV-bed occupancy by NHS region"
    )
    ax_bot.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    handles, labels = ax_top.get_legend_handles_labels()
    wave_idx = [i for i, lab in enumerate(labels) if lab in WAVE_PERIODS]
    region_idx = [i for i in range(len(labels)) if i not in wave_idx]
    ordered = [handles[i] for i in region_idx + wave_idx]
    ordered_labels = [labels[i] for i in region_idx + wave_idx]
    fig.legend(
        ordered, ordered_labels, ncol=5, loc="outside lower center",
        frameon=False, fontsize=8,
    )
    return save_figure(fig, "fig_mobility_overlay")


# ---------------------------------------------------------------------------
# Figure: per-region context (population + IMD)
# ---------------------------------------------------------------------------


def figure_region_context(df: pd.DataFrame) -> Path:
    """Per-region static context: ONS population, IMD score, peak MV beds.

    Sets up the operational comparison: London is large, deprived, and
    surged hardest in Alpha; the North West and N.E.+Yorkshire are the
    most deprived; the South East / East are larger but less deprived.
    """
    static = pd.read_csv(STATIC_CSV)
    region_order = ("Y56", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63")
    static = static.set_index("region_code").loc[list(region_order)].reset_index()
    region_peaks = df.groupby("region_code", observed=True)["mv_beds"].max()
    static["peak_mv_beds"] = static["region_code"].map(region_peaks)
    static["population_millions"] = static["population"] / 1e6

    fig, (ax_pop, ax_imd, ax_peak) = plt.subplots(
        1, 3, figsize=(FULL_WIDTH_IN, 3.6), layout="constrained",
        gridspec_kw={"wspace": 0.15},
    )
    positions = list(range(len(static), 0, -1))
    colours = list(REGION_PALETTE[: len(static)])
    bars_pop = ax_pop.barh(
        positions, static["population_millions"], color=colours,
        edgecolor="black", linewidth=0.6, height=0.6,
    )
    ax_pop.set_yticks(positions)
    ax_pop.set_yticklabels(static["region_name"])
    ax_pop.set_xlabel("Population (millions)")
    ax_pop.set_title("ONS MYE 2021", fontsize=9)
    for bar, v in zip(bars_pop, static["population_millions"]):
        ax_pop.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}M", va="center", fontsize=7)
    ax_imd.barh(
        positions, static["imd_pop_weighted_score"], color=colours,
        edgecolor="black", linewidth=0.6, height=0.6,
    )
    ax_imd.set_yticks(positions)
    ax_imd.set_yticklabels([""] * len(positions))
    ax_imd.set_xlabel("IMD score (higher = more deprived)")
    ax_imd.set_title("English IMD 2019", fontsize=9)
    for pos, v in zip(positions, static["imd_pop_weighted_score"]):
        ax_imd.text(v + 0.3, pos, f"{v:.1f}", va="center", fontsize=7)
    ax_peak.barh(
        positions, static["peak_mv_beds"], color=colours,
        edgecolor="black", linewidth=0.6, height=0.6,
    )
    ax_peak.set_yticks(positions)
    ax_peak.set_yticklabels([""] * len(positions))
    ax_peak.set_xlabel("Peak MV beds (observed)")
    ax_peak.set_title("Observed Alpha-wave peak", fontsize=9)
    for pos, v in zip(positions, static["peak_mv_beds"]):
        ax_peak.text(v + 15, pos, f"{int(v)}", va="center", fontsize=7)
    fig.suptitle(
        "Per-region context: population, deprivation, and observed peak load",
        fontsize=10,
    )
    return save_figure(fig, "fig_region_context")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table_regional_summary(df: pd.DataFrame) -> Path:
    """Per-region × per-wave summary statistics for MV-bed occupancy."""
    enriched = assign_wave(df)
    enriched = enriched[enriched["wave"] != "unassigned"]
    summary = (
        enriched.groupby(["region_name", "wave"], observed=True)["mv_beds"]
        .agg(
            n_days="size",
            mean="mean",
            std="std",
            median="median",
            peak="max",
            integrated_bed_days="sum",
        )
        .round(2)
        .reset_index()
    )
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    out_path = EDA_OUT / "table_regional_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def table_wave_summary(df: pd.DataFrame) -> Path:
    """National-aggregate summary statistics per wave."""
    enriched = assign_wave(df)
    enriched = enriched[enriched["wave"] != "unassigned"]
    daily = (
        enriched.groupby(["date", "wave"], observed=True)["mv_beds"]
        .sum()
        .reset_index()
    )
    summary = (
        daily.groupby("wave", observed=True)["mv_beds"]
        .agg(
            n_days="size",
            national_mean="mean",
            national_peak="max",
            national_total_bed_days="sum",
        )
        .round(2)
        .reset_index()
    )
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    out_path = EDA_OUT / "table_wave_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    if not REGIONAL_CSV.exists():
        raise FileNotFoundError(
            f"Tidy regional CSV not found at {REGIONAL_CSV}. "
            f"Run scripts/build_regional_dataset.py first."
        )
    apply_paper_style()
    df = load_tidy_csv()
    print(f"Loaded {len(df):,} rows from {REGIONAL_CSV.relative_to(REPO_ROOT)}")

    outputs: list[Path] = []
    outputs.append(figure_regional_mv_beds(df))
    outputs.append(figure_metric_overview(df))
    outputs.append(figure_regional_distributions(df))
    outputs.append(figure_peak_alignment(df))
    outputs.append(figure_data_quality(df))
    outputs.append(figure_wave_overlay(df))
    outputs.append(figure_autocorrelation(df))
    outputs.append(figure_regional_acf(df))
    outputs.append(figure_lead_lag(df))
    outputs.append(figure_weekly_seasonality(df))
    if MOBILITY_CSV.exists():
        outputs.append(figure_mobility_overlay(df))
    if STATIC_CSV.exists():
        outputs.append(figure_region_context(df))
    outputs.append(table_regional_summary(df))
    outputs.append(table_wave_summary(df))

    print("\nGenerated EDA artefacts:")
    for path in outputs:
        rel = path.relative_to(REPO_ROOT)
        print(f"  - {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
