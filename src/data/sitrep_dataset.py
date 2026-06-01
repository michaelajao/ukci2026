"""Parse NHS UEC SitRep monthly files into a monthly adult-critical-care series.

DATA SOURCE (download step, kept separate from this preprocess step):
  NHS England UEC "Critical Care and General & Acute Beds" daily SitRep,
  per-month workbooks. Direct URLs follow
    https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2022/05/
        YYYYMM-Month[-YYYY]-sitrep-data-FINAL.xlsx
  saved to data/raw/uec_sitrep_YYYYMM.xlsx . Each workbook is a MONTHLY AVERAGE
  ("Average daily number of available and occupied beds"), not daily.

Each 'type 1 acute trusts' sheet has, after ~14 metadata rows:
  (a) a top block: ENGLAND + the 7 NHS regions with the full bed-metric set;
  (b) a per-trust block headed 'Region | Trust Name | Adult critical care beds
      available | Adult critical care beds occupied' (~125 acute trusts).
Trusts are identified by NAME only (no ODS code) -> a name->ICB mapping is a
later step before ICB aggregation.

Outputs (preprocess step, entry point ``ukci-build-sitrep-dataset``):
  data/processed/sitrep_trust_monthly.csv   month, region, trust_name, cc_available, cc_occupied
  data/processed/sitrep_region_monthly.csv  month, region, cc_available, cc_occupied
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from utils import repo_root

ROOT = repo_root()
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

AVAIL = "Adult critical care beds available"
OCC = "Adult critical care beds occupied"
REGIONS = {
    "East of England", "London", "Midlands", "North East and Yorkshire",
    "North West", "South East", "South West",
}


def _cell(raw, r, c):
    return str(raw.iat[r, c]).strip() if not pd.isna(raw.iat[r, c]) else ""


def parse_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trusts, regions) frames for one monthly workbook."""
    month = pd.Timestamp(f"{path.stem[-6:-2]}-{path.stem[-2:]}-01")
    xl = pd.ExcelFile(path)
    sheet = next((s for s in xl.sheet_names if "type 1" in s.lower()),
                 xl.sheet_names[0])
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    # locate the trust-block header row (the cell reading 'Trust Name')
    th = tname_col = None
    for r in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            if _cell(raw, r, c) == "Trust Name":
                th, tname_col = r, c
                break
        if th is not None:
            break
    if th is None:
        raise RuntimeError(f"No 'Trust Name' header in {path.name}")

    hdr = [_cell(raw, th, c) for c in range(raw.shape[1])]
    avail_col = hdr.index(AVAIL)
    occ_col = hdr.index(OCC)
    region_col = tname_col - 1

    trusts = []
    for r in range(th + 1, raw.shape[0]):
        name = _cell(raw, r, tname_col)
        if not name:
            continue
        trusts.append({
            "month": month,
            "region": _cell(raw, r, region_col),
            "trust_name": name,
            "cc_available": pd.to_numeric(raw.iat[r, avail_col], errors="coerce"),
            "cc_occupied": pd.to_numeric(raw.iat[r, occ_col], errors="coerce"),
        })
    trusts = pd.DataFrame(trusts)

    # region block: rows above the trust header whose geography col is a region
    regions = []
    for r in range(th):
        g = _cell(raw, r, 2)
        if g in REGIONS:
            regions.append({
                "month": month,
                "region": g,
                "cc_available": pd.to_numeric(raw.iat[r, avail_col], errors="coerce"),
                "cc_occupied": pd.to_numeric(raw.iat[r, occ_col], errors="coerce"),
            })
    regions = pd.DataFrame(regions)
    return trusts, regions


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(RAW.glob("uec_sitrep_2*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No uec_sitrep_*.xlsx in {RAW}")
    trust_parts, region_parts = [], []
    for f in files:
        t, r = parse_file(f)
        trust_parts.append(t)
        region_parts.append(r)
    trusts = pd.concat(trust_parts, ignore_index=True).sort_values(["month", "region", "trust_name"])
    regions = pd.concat(region_parts, ignore_index=True).sort_values(["month", "region"])
    PROC.mkdir(parents=True, exist_ok=True)
    trusts.to_csv(PROC / "sitrep_trust_monthly.csv", index=False)
    regions.to_csv(PROC / "sitrep_region_monthly.csv", index=False)
    return trusts, regions


def main() -> int:
    trusts, regions = build()
    print(f"Wrote {PROC / 'sitrep_trust_monthly.csv'} "
          f"({len(trusts)} rows, {trusts['trust_name'].nunique()} trusts, "
          f"{trusts['month'].nunique()} months)")
    print(f"Wrote {PROC / 'sitrep_region_monthly.csv'} ({len(regions)} rows)")
    # peak-month stress reference: total occupied adult CC beds per month (England)
    by_month = regions.groupby("month").agg(
        cc_available=("cc_available", "sum"),
        cc_occupied=("cc_occupied", "sum")).round(0)
    by_month["occ_rate"] = (by_month["cc_occupied"] / by_month["cc_available"]).round(3)
    print("\nEngland adult critical-care beds by month (sum over 7 regions):")
    print(by_month.to_string())
    peak = by_month["cc_occupied"].idxmax()
    print(f"\nPeak occupancy month (stress scenario): {peak.date()} "
          f"({by_month.loc[peak, 'cc_occupied']:.0f} occupied of "
          f"{by_month.loc[peak, 'cc_available']:.0f} available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
