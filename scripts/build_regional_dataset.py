#!/usr/bin/env python3
"""
Harmonise the three NHS England COVID-19 Hospital Activity XLSX archives into
one tidy regional daily CSV: data/processed/regional_daily.csv

Output schema:
    date          (datetime, daily)
    region_code   (str, NHS England region code: Y56, Y61, ...)
    region_name   (str, human-readable region name)
    admissions    (float, COVID-19 hospital admissions)
    occupied_beds (float, occupied beds with COVID-19 patients)
    mv_beds       (float, mechanical ventilation beds occupied by COVID patients)

NHS England published these as Excel files with multiple sheets per file.
The relevant sheets are typically named like "Daily admissions", "Beds occupied
by COVID-19 patients", "MV beds occupied", with regions as columns and dates
as rows.

Implementation notes:
- Each XLSX archive may have slightly different sheet names and column orders.
  This script auto-detects sheets by content rather than name.
- Region naming has been consistent across all three files since 1 April 2019,
  but we still verify and reconcile.
- Duplicate dates across overlapping archives are resolved by trusting the
  later archive (which is the official corrected version).
- A data quality report is written to data/processed/data_quality_report.md.

Usage:
    python scripts/build_regional_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


# NHS England Region codes and canonical names
NHS_REGIONS = {
    "Y56": "London",
    "Y58": "South West",
    "Y59": "South East",
    "Y60": "Midlands",
    "Y61": "East of England",
    "Y62": "North West",
    "Y63": "North East and Yorkshire",
}


def main() -> int:
    """Stub: this is filled in once the downloads succeed and we can inspect
    the actual XLSX structure. The plan is in docs/03_TIMELINE.md (Wed 7 May)."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not any(RAW_DIR.glob("*.xlsx")):
        sys.exit(
            f"No XLSX files in {RAW_DIR}. "
            f"Run scripts/download_nhs_data.py first."
        )

    print("This script is a stub. Implementation steps:")
    print("  1. Use openpyxl or pandas to load each XLSX archive")
    print("  2. Auto-detect sheets by content (admissions, beds, MV beds)")
    print("  3. Reshape from wide (regions as columns) to long (one row per "
          "region-day)")
    print("  4. Concatenate the three archives, resolving date overlaps")
    print("  5. Validate region naming, date contiguity, and missingness")
    print("  6. Write data/processed/regional_daily.csv")
    print("  7. Write data/processed/data_quality_report.md")
    print("")
    print("See docs/03_TIMELINE.md, Thursday 7 May entry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
