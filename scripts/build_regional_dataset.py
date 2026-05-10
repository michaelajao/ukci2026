#!/usr/bin/env python3
"""
Harmonise the NHS England COVID-19 Hospital Activity XLSX archives into one
tidy regional daily CSV at ``data/processed/regional_daily.csv``.

Each archive's ``Daily publication`` sheet contains seven stacked tables, one
per metric. We extract the three load-bearing metrics named in the methodology
(MV beds, occupied beds, admissions) plus one useful covariate (estimated new
hospital cases). For each metric the layout is:

    row  N    "<n>. <metric title>"
    row N+1   (description text, ignored)
    row N+2   "Name", date_1, date_2, ... date_K
    row N+3   "ENGLAND", v_1, v_2, ... v_K
    row N+4+  region_name, v_1, ...
    ...
    row N+11  blank / next section title

Output schema (long → pivoted to wide):

    date          datetime  daily, 1 Aug 2020 – 31 Aug 2022
    region_code   str       Y56..Y63
    region_name   str       canonical NHS region name
    admissions    float     section 1
    hospital_cases float    section 2
    occupied_beds float     section 6
    mv_beds       float     section 7  (PRIMARY FORECASTING TARGET)

Overlapping dates across archives are resolved by trusting the later archive
(NHS England's official corrected figures). A provenance and quality report
is written to ``data/processed/data_quality_report.md``.

Usage:
    python scripts/build_regional_dataset.py

Run from the repository root.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DAILY_SHEET = "Daily publication"

# NHS England region codes and canonical names (per the NHS Digital ODS table).
NHS_REGIONS = {
    "Y56": "London",
    "Y58": "South West",
    "Y59": "South East",
    "Y60": "Midlands",
    "Y61": "East of England",
    "Y62": "North West",
    "Y63": "North East and Yorkshire",
}
REGION_NAME_TO_CODE = {name: code for code, name in NHS_REGIONS.items()}

# Section-number → output column-name mapping. Sections we ignore:
#   3 estimated new admissions from community (subset of 1)
#   4 estimated new admissions community 3-7 day lagged (subset of 1)
#   5 hospital admissions from a care home (subset of 1)
SECTIONS_TO_EXTRACT: dict[int, str] = {
    1: "admissions",         # Total reported admissions to hospital and diagnoses
    2: "hospital_cases",     # Estimated new hospital cases
    6: "occupied_beds",      # Total beds occupied by confirmed COVID-19 patients
    7: "mv_beds",            # Mechanical Ventilation beds occupied
}

SECTION_TITLE_RE = re.compile(r"^(\d+)\.\s+(.+)$")


# ---------------------------------------------------------------------------
# Parsing one XLSX archive
# ---------------------------------------------------------------------------

def find_section_rows(df: pd.DataFrame) -> dict[int, int]:
    """Return mapping of section number → row index of its title."""
    rows: dict[int, int] = {}
    for i in range(len(df)):
        cell = df.iat[i, 0]
        if isinstance(cell, str):
            m = SECTION_TITLE_RE.match(cell.strip())
            if m:
                rows[int(m.group(1))] = i
    return rows


def extract_section(
    df: pd.DataFrame, section_num: int, title_row: int, end_row: int
) -> Iterable[dict]:
    """Yield long-form records for one stacked metric table.

    Layout (column 0 is always NaN; the table starts at column 1):
        col 0   col 1                      col 2..N
        ----    -----                      --------
                'Name'                     date_1, date_2, ...
                'ENGLAND'                  v_1, v_2, ...     (aggregate, skipped)
                <region_name>              v_1, v_2, ...
                ...
    """
    header_row = title_row + 2
    if header_row >= len(df):
        return

    # Confirm the header row's region-label cell (col 1) is the literal "Name".
    label_cell = df.iat[header_row, 1]
    if not isinstance(label_cell, str) or label_cell.strip() != "Name":
        return

    # Dates live in columns 2..N of the header row.
    date_row = df.iloc[header_row]
    dates = pd.to_datetime(date_row.iloc[2:], errors="coerce")
    valid_date_cols: list[int] = [
        col for col in range(2, len(date_row)) if not pd.isna(dates.iloc[col - 2])
    ]
    valid_dates = {col: dates.iloc[col - 2] for col in valid_date_cols}

    metric = SECTIONS_TO_EXTRACT[section_num]

    # Data rows: header_row+1 up to (but not including) end_row.
    # Region name lives in column 1; values in columns 2..N.
    for r in range(header_row + 1, end_row):
        region = df.iat[r, 1]
        if not isinstance(region, str):
            continue
        region = region.strip()
        if not region or region.upper() == "ENGLAND":
            continue
        if region not in REGION_NAME_TO_CODE:
            continue

        for col in valid_date_cols:
            val = df.iat[r, col]
            if pd.isna(val):
                continue
            try:
                value = float(val)
            except (TypeError, ValueError):
                continue
            yield {
                "date": valid_dates[col],
                "region_code": REGION_NAME_TO_CODE[region],
                "region_name": region,
                "metric": metric,
                "value": value,
            }


def parse_archive(path: Path) -> pd.DataFrame:
    """Parse one NHS XLSX archive into a long-form DataFrame."""
    df = pd.read_excel(path, sheet_name=DAILY_SHEET, header=None)
    section_rows = find_section_rows(df)

    records: list[dict] = []
    sorted_section_rows = sorted(section_rows.values())
    for section_num, title_row in section_rows.items():
        if section_num not in SECTIONS_TO_EXTRACT:
            continue
        # Section ends at the next section title or end of sheet.
        next_titles = [r for r in sorted_section_rows if r > title_row]
        end_row = next_titles[0] if next_titles else len(df)
        records.extend(extract_section(df, section_num, title_row, end_row))

    long = pd.DataFrame(records)
    if long.empty:
        return long
    long["source_archive"] = path.name
    return long


# ---------------------------------------------------------------------------
# Combining + validation + output
# ---------------------------------------------------------------------------

def archive_period_key(filename: str) -> str:
    """Extract a sortable date-key from the XLSX filename so later archives
    win in deduplication. Filenames embed the period-end date e.g. ``...20210406...``."""
    digits = re.findall(r"\d{8}", filename)
    return digits[-1] if digits else filename


def combine_archives(per_archive: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, deduplicate (later archive wins), pivot to wide."""
    long = pd.concat(per_archive, ignore_index=True)
    long["_priority"] = long["source_archive"].map(archive_period_key)
    long = long.sort_values("_priority")
    long = long.drop_duplicates(
        subset=["date", "region_code", "metric"], keep="last"
    )

    wide = long.pivot_table(
        index=["date", "region_code", "region_name"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Ensure all expected metric columns exist even if absent from every archive.
    for m in SECTIONS_TO_EXTRACT.values():
        if m not in wide.columns:
            wide[m] = pd.NA

    cols = ["date", "region_code", "region_name", *SECTIONS_TO_EXTRACT.values()]
    wide = wide[cols].sort_values(["date", "region_code"]).reset_index(drop=True)
    return wide


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of issue strings; empty list = healthy dataset."""
    issues: list[str] = []
    n_regions = df["region_code"].nunique()
    if n_regions != 7:
        issues.append(f"expected 7 regions, found {n_regions}")

    expected_codes = set(NHS_REGIONS.keys())
    actual_codes = set(df["region_code"].unique())
    missing = expected_codes - actual_codes
    if missing:
        issues.append(f"missing region codes: {sorted(missing)}")
    extra = actual_codes - expected_codes
    if extra:
        issues.append(f"unexpected region codes: {sorted(extra)}")

    # Date contiguity: expect every calendar day per region from min..max.
    for code, sub in df.groupby("region_code"):
        dates = pd.to_datetime(sub["date"]).sort_values().drop_duplicates()
        full = pd.date_range(dates.min(), dates.max(), freq="D")
        gaps = full.difference(dates)
        if len(gaps) > 0:
            issues.append(
                f"region {code}: {len(gaps)} missing days between "
                f"{dates.min().date()} and {dates.max().date()}"
            )

    # Missingness audit on primary target.
    nan_mv = int(df["mv_beds"].isna().sum())
    if nan_mv > 0:
        issues.append(f"mv_beds has {nan_mv} NaN values")

    return issues


def write_quality_report(
    df: pd.DataFrame,
    per_archive: list[pd.DataFrame],
    issues: list[str],
    out_path: Path,
) -> None:
    """Write a human-readable provenance + quality report."""
    lines: list[str] = []
    lines.append("# Regional daily dataset — data quality report")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M UTC}")
    lines.append(f"Source files: {len(per_archive)} XLSX archive(s) in `data/raw/`")
    lines.append("")
    lines.append("## Archive coverage")
    lines.append("")
    lines.append("| Archive | rows extracted | min date | max date |")
    lines.append("|---|---|---|---|")
    for sub in per_archive:
        if sub.empty:
            continue
        name = sub["source_archive"].iloc[0]
        lines.append(
            f"| `{name}` | {len(sub):,} | {sub['date'].min().date()} | "
            f"{sub['date'].max().date()} |"
        )
    lines.append("")
    lines.append("## Output dataset")
    lines.append("")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Date range: {df['date'].min().date()} – {df['date'].max().date()}")
    lines.append(f"- Distinct regions: {df['region_code'].nunique()}")
    lines.append(f"- Distinct dates: {df['date'].nunique()}")
    lines.append("")
    lines.append("### Per-metric summary statistics")
    lines.append("")
    lines.append("| metric | non-null | mean | std | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for m in SECTIONS_TO_EXTRACT.values():
        if m not in df.columns:
            continue
        s = df[m].dropna()
        if s.empty:
            lines.append(f"| {m} | 0 | – | – | – | – |")
            continue
        lines.append(
            f"| {m} | {len(s):,} | {s.mean():.1f} | {s.std():.1f} | "
            f"{s.min():.1f} | {s.max():.1f} |"
        )
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    if issues:
        lines.append("Issues detected:")
        lines.append("")
        for it in issues:
            lines.append(f"- {it}")
    else:
        lines.append("All checks passed: 7 regions, no date gaps, no missing target values.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    if not xlsx_files:
        sys.exit(
            f"No XLSX files in {RAW_DIR}. "
            f"Run scripts/download_nhs_data.py first."
        )

    print(f"Parsing {len(xlsx_files)} archive(s)...")
    per_archive: list[pd.DataFrame] = []
    for f in xlsx_files:
        sub = parse_archive(f)
        if sub.empty:
            print(f"  [warn] {f.name}: 0 records extracted")
        else:
            print(
                f"  [ok]   {f.name}: {len(sub):,} records, "
                f"{sub['date'].min().date()}..{sub['date'].max().date()}"
            )
        per_archive.append(sub)

    non_empty = [s for s in per_archive if not s.empty]
    if not non_empty:
        sys.exit("No records extracted from any archive. Aborting.")

    df = combine_archives(non_empty)
    issues = validate(df)

    out_csv = PROCESSED_DIR / "regional_daily.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df):,} rows)")

    out_md = PROCESSED_DIR / "data_quality_report.md"
    write_quality_report(df, per_archive, issues, out_md)
    print(f"Wrote {out_md}")

    if issues:
        print(f"\n{len(issues)} validation issue(s):")
        for it in issues:
            print(f"  - {it}")
        return 1
    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
