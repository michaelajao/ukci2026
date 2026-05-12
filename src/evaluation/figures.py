"""Reusable matplotlib helpers for paper-quality figures.

Centralises styling, colour palette, output sizing, and serialisation
conventions so every figure produced for the UKCI 2026 paper has a
consistent look. Importing this module side-effect-free; call
:func:`apply_paper_style` once at the start of a figure script.

Conventions:

- All figures are sized for the Springer LNNS two-column layout
  (max width 117 mm = 4.6 in for a single column, 240 mm = 9.4 in for
  full width). Defaults assume single-column.
- Wave colour bands (Alpha / Delta / Omicron) use a fixed palette so
  figures across the paper line up.
- Saving writes both PDF (for the manuscript) and PNG (for slides /
  quick preview) to the same basename.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

#: Springer LNNS column widths in inches.
COLUMN_WIDTH_IN: float = 4.6
"""Single-column figure width (mm 117 / in 4.6)."""

FULL_WIDTH_IN: float = 9.4
"""Full-width figure width (mm 240 / in 9.4)."""

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_paper_style() -> None:
    """Apply a single matplotlib style suitable for the LNNS manuscript.

    Call once near the top of a figure script. Subsequent calls are
    idempotent.
    """
    mpl.rcParams.update(
        {
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
            "lines.linewidth": 1.2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def save_figure(fig: plt.Figure, basename: str | Path, *, pdf: bool = False) -> Path:
    """Save ``fig`` as PNG (and optionally PDF) under ``figures/``.

    The paper renders fine from PNG at 300 dpi; PDFs are only produced when
    explicitly requested (``pdf=True``).

    Args:
        fig: A matplotlib Figure.
        basename: Output basename (with or without extension). Stored under
            ``<repo>/figures/`` unless ``basename`` is already absolute.
        pdf: Also write a PDF copy.

    Returns:
        Path to the PNG file.
    """
    base = Path(basename).with_suffix("")
    if not base.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        base = repo_root / "figures" / base
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    fig.savefig(png_path, dpi=300)
    if pdf:
        fig.savefig(base.with_suffix(".pdf"))
    return png_path
