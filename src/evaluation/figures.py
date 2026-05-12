"""Paper matplotlib settings and helpers.

This is the single plotting style module for the UKCI 2026 paper. Keep plot
logic in the analysis modules, but keep typography, colours, figure sizes,
line styles, and export conventions here so every figure has the same visual
grammar.

Conventions:

- All figures are sized for the Springer LNNS two-column layout
  (max width 117 mm = 4.6 in for a single column, 240 mm = 9.4 in for
  full width).
- Wave colour bands (Alpha / Delta / Omicron) use a fixed palette so
  figures across the paper line up.
- Saving writes PNG by default; request PDF only for manuscript assets that
  benefit from vector output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from utils import figures_dir

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

#: Springer LNNS column widths in inches.
COLUMN_WIDTH_IN: float = 4.6
"""Single-column figure width (mm 117 / in 4.6)."""

FULL_WIDTH_IN: float = 9.4
"""Full-width figure width (mm 240 / in 9.4)."""

HALF_WIDTH_IN: float = 7.0
"""Intermediate width for dense one-row panels that need more room."""

PAPER_DPI: int = 300
"""Default raster export resolution."""

# Keep this dictionary close to the working settings already used by the
# figures. Additions are conservative and mostly make defaults explicit.
PAPER_RC_PARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.0,
    "patch.linewidth": 0.8,
    "figure.dpi": PAPER_DPI,
    "savefig.dpi": PAPER_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

#: Wave boundaries (inclusive of the start date, exclusive of the end).
#: Matches the chronological-by-wave split in ``src/data/loader.py``.
WAVE_PERIODS: dict[str, tuple[str, str]] = {
    "Alpha + early vaccination": ("2020-08-01", "2021-06-01"),
    "Delta": ("2021-06-01", "2021-12-01"),
    "Omicron and beyond": ("2021-12-01", "2022-09-01"),
}

#: Colours for the wave background bands. Soft pastel tones so the bands
#: sit behind the lines without competing for attention.
WAVE_COLOURS: dict[str, str] = {
    "Alpha + early vaccination": "#e7f0fa",
    "Delta": "#fbeee0",
    "Omicron and beyond": "#eaf5e9",
}

#: Categorical palette for NHS regions (Okabe-Ito, colourblind-safe).
REGION_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (last so it falls on the lightest line)
)

TRUTH_COLOUR = "black"
"""Colour used for observed/realised series in forecast figures."""

FORECASTER_STYLES: tuple[dict[str, object], ...] = (
    {
        "model": "seasonal_naive",
        "label": "Seasonal-naive(7)",
        "color": "#888888",
        "linestyle": (0, (1, 1.5)),
        "linewidth": 0.9,
    },
    {
        "model": "xgboost_per_region",
        "label": "XGBoost (lag features)",
        "color": "#CC79A7",
        "linestyle": (0, (3, 1, 1, 1)),
        "linewidth": 0.9,
    },
    {
        "model": "gru_per_region",
        "label": "GRU per region",
        "color": "#009E73",
        "linestyle": (0, (4, 1.5)),
        "linewidth": 0.9,
    },
    {
        "model": "arima_per_region",
        "label": "ARIMA (refit per origin)",
        "color": "#D55E00",
        "linestyle": (0, (5, 2)),
        "linewidth": 1.0,
    },
    {
        "model": "pinn_gru",
        "label": "PinnGRU (proposed)",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.6,
    },
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_paper_style() -> None:
    """Apply a single matplotlib style suitable for the LNNS manuscript.

    Call once near the top of a figure script. Subsequent calls are
    idempotent.
    """
    mpl.rcParams.update(PAPER_RC_PARAMS)


def overlay_wave_bands(ax: plt.Axes, *, alpha: float = 0.45) -> None:
    """Shade the background of ``ax`` with the three wave periods.

    Args:
        ax: The axes to shade. Must already have a date x-axis.
        alpha: Alpha for the shading rectangles.
    """
    for label, (start, end) in WAVE_PERIODS.items():
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=WAVE_COLOURS[label],
            alpha=alpha,
            zorder=0,
            label=label,
        )


def save_figure(
    fig: plt.Figure,
    basename: str | Path,
    *,
    pdf: bool = False,
    close: bool = False,
) -> Path:
    """Save ``fig`` as PNG (and optionally PDF) under ``figures/``.

    The paper renders fine from PNG at 300 dpi; PDFs are only produced when
    explicitly requested (``pdf=True``).

    Args:
        fig: A matplotlib Figure.
        basename: Output basename (with or without extension). Stored under
            ``<repo>/figures/`` unless ``basename`` is already absolute.
        pdf: Also write a PDF copy.
        close: Close the figure after saving to release memory in batch runs.

    Returns:
        Path to the PNG file.
    """
    base = Path(basename).with_suffix("")
    if not base.is_absolute():
        base = figures_dir(str(base))
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    fig.savefig(png_path, dpi=PAPER_DPI)
    if pdf:
        fig.savefig(base.with_suffix(".pdf"))
    if close:
        plt.close(fig)
    return png_path
