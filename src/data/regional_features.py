#!/usr/bin/env python3
"""Build per-NHS-region static features from ONS MYE 2021.

Reads ``data/raw/supporting/ukpopestimatesmid2021on2021geographyfinal.xls``
(downloaded by ``ukci-download-supporting-data``) and writes
``data/processed/regional_static.csv`` — one row per NHS England region
with the ONS Mid-Year Population Estimate.

The 7 NHS England commissioning regions are exact unions of NUTS-1 ITL1
regions, so the cross-walk is straightforward:

    NHS region                NUTS-1 (ITL1)
    -----------------------   ----------------------------------------
    Y56 London                E12000007
    Y58 South West            E12000009
    Y59 South East            E12000008
    Y60 Midlands              E12000004 (East Mid.) + E12000005 (West Mid.)
    Y61 East of England       E12000006
    Y62 North West            E12000002
    Y63 N. East and Yorkshire E12000001 (N.E.) + E12000003 (Yorkshire)

The ONS MYE2-Persons sheet is hierarchical — a NUTS-1 "Region" row is
followed by its constituent LA rows — so the LA-to-region mapping is
derived by walking the sheet top-to-bottom and inheriting the most recent
Region.

Run from the repository root:

    ukci-build-regional-features
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from utils import repo_root

REPO_ROOT = repo_root()
SUPPORTING_DIR = REPO_ROOT / "data" / "raw" / "supporting"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ONS_FILE = SUPPORTING_DIR / "ukpopestimatesmid2021on2021geographyfinal.xls"

#: NHS region code -> list of NUTS-1 (ITL1) codes whose union equals it.
NUTS_TO_NHS: dict[str, list[str]] = {
    "Y56": ["E12000007"],
    "Y58": ["E12000009"],
    "Y59": ["E12000008"],
    "Y60": ["E12000004", "E12000005"],
    "Y61": ["E12000006"],
    "Y62": ["E12000002"],
    "Y63": ["E12000001", "E12000003"],
}

NHS_REGION_NAMES: dict[str, str] = {
    "Y56": "London",
    "Y58": "South West",
    "Y59": "South East",
    "Y60": "Midlands",
    "Y61": "East of England",
    "Y62": "North West",
    "Y63": "North East and Yorkshire",
}

def _nuts_to_nhs(nuts_code: str) -> str | None:
    for nhs_code, nuts_list in NUTS_TO_NHS.items():
        if nuts_code in nuts_list:
            return nhs_code
    return None


# ---------------------------------------------------------------------------
# ONS MYE 2021 -> LA-to-NHS-region lookup + region population totals
# ---------------------------------------------------------------------------


def build_la_lookup() -> pd.DataFrame:
    """Walk the ONS MYE2-Persons sheet to build an LA-District lookup.

    Returns:
        DataFrame with columns ``la_code``, ``la_name``, ``nuts_code``,
        ``nuts_name``, ``nhs_code``, ``population``.
    """
    df = pd.read_excel(ONS_FILE, sheet_name="MYE2 - Persons", skiprows=7)
    df = df[["Code", "Name", "Geography", "All ages"]].dropna(subset=["Code"])

    la_geos = {
        "Non-metropolitan District",
        "Unitary Authority",
        "Metropolitan District",
        "London Borough",
    }

    records: list[dict] = []
    current_nuts_code: str | None = None
    current_nuts_name: str | None = None
    for _, row in df.iterrows():
        geo = row["Geography"]
        if geo == "Region":
            current_nuts_code = row["Code"]
            current_nuts_name = row["Name"]
            continue
        if geo == "Country":
            current_nuts_code = None
            current_nuts_name = None
            continue
        if geo in la_geos and current_nuts_code is not None:
            nhs_code = _nuts_to_nhs(current_nuts_code)
            if nhs_code is None:
                continue
            records.append(
                {
                    "la_code": row["Code"],
                    "la_name": row["Name"],
                    "nuts_code": current_nuts_code,
                    "nuts_name": current_nuts_name,
                    "nhs_code": nhs_code,
                    "population": int(row["All ages"]),
                }
            )
    return pd.DataFrame.from_records(records)


def regional_population(la_lookup: pd.DataFrame) -> pd.Series:
    """Population total per NHS region (sum across constituent LAs).

    Args:
        la_lookup: LA-level lookup from :func:`build_la_lookup`.

    Returns:
        Series indexed by NHS region code with the total population.
    """
    return la_lookup.groupby("nhs_code")["population"].sum()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not ONS_FILE.exists():
        raise FileNotFoundError(
            f"ONS file not found: {ONS_FILE}. Run "
            f"ukci-download-supporting-data first."
        )

    print("Building LA-District lookup from ONS MYE2-Persons sheet...")
    la_lookup = build_la_lookup()
    print(f"  {len(la_lookup):,} English LA Districts -> {la_lookup['nhs_code'].nunique()} NHS regions")

    print("Aggregating ONS population to NHS region totals...")
    pop = regional_population(la_lookup)

    static = pd.DataFrame({
        "region_code": list(NHS_REGION_NAMES),
        "region_name": [NHS_REGION_NAMES[c] for c in NHS_REGION_NAMES],
        "population": [int(pop[c]) for c in NHS_REGION_NAMES],
    })
    static_path = PROCESSED_DIR / "regional_static.csv"
    static.to_csv(static_path, index=False)
    print(f"Wrote {static_path}")
    print(static.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
