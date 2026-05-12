#!/usr/bin/env python3
"""Build per-NHS-region auxiliary features from the supporting datasets.

Reads from ``data/raw/supporting/`` (downloaded by
``scripts/download_external_data.py``) and writes to ``data/processed/``:

- ``regional_static.csv`` — one row per NHS England region with population
  (ONS MYE 2021) and population-weighted mean IMD 2019 score.
- ``regional_mobility.csv`` — daily Google Community Mobility series per
  NHS region for the 6 mobility categories, with a +21-day forward shift
  applied (Cartení et al. 2020 finding, reused in Valente et al. 2022).

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

    python scripts/build_regional_features.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTING_DIR = REPO_ROOT / "data" / "raw" / "supporting"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ONS_FILE = SUPPORTING_DIR / "ukpopestimatesmid2021on2021geographyfinal.xls"
IMD_FILE = SUPPORTING_DIR / "IoD2019_File_7_Scores_Ranks_Deciles.csv"
MOBILITY_FILES = (
    SUPPORTING_DIR / "2020_GB_Region_Mobility_Report.csv",
    SUPPORTING_DIR / "2021_GB_Region_Mobility_Report.csv",
    SUPPORTING_DIR / "2022_GB_Region_Mobility_Report.csv",
)

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

#: Google Mobility category columns -> short names.
MOBILITY_COLS: dict[str, str] = {
    "retail_and_recreation_percent_change_from_baseline": "retail_recreation",
    "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy",
    "parks_percent_change_from_baseline": "parks",
    "transit_stations_percent_change_from_baseline": "transit_stations",
    "workplaces_percent_change_from_baseline": "workplaces",
    "residential_percent_change_from_baseline": "residential",
}

#: Forward shift (days) applied to mobility before joining to clinical
#: outcomes. Cartení et al. (2020) found ~21-day mobility-to-cases lag.
MOBILITY_LAG_DAYS = 21


def _nuts_to_nhs(nuts_code: str) -> str | None:
    for nhs_code, nuts_list in NUTS_TO_NHS.items():
        if nuts_code in nuts_list:
            return nhs_code
    return None


# ---------------------------------------------------------------------------
# ONS MYE 2021 -> LA-to-NHS-region lookup + region population totals
# ---------------------------------------------------------------------------


def _canonical_name(name: str) -> str:
    """Lowercase + strip common suffixes so ONS and Google names align."""
    s = name.lower().strip()
    for suffix in (
        " council area",
        " council",
        " principal area",
        " county borough",
        " borough",
        " district",
        " (met county)",
        " (b)",
        " city",
        " city of",
        ", city of",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = s.replace("&", "and")
    return s


def build_la_lookup() -> pd.DataFrame:
    """Walk the ONS MYE2-Persons sheet to build an LA-District lookup.

    This is the fine-grained (309 English LA Districts) view — kept so the
    IMD aggregation (which carries LA-District codes per LSOA) can join
    through it.

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


def build_utla_lookup() -> pd.DataFrame:
    """Walk the ONS MYE2-Persons sheet to build a UTLA-level lookup.

    Google Community Mobility reports at the UTLA level — counties (24),
    metropolitan counties (6), unitary authorities (81), and a single
    "Greater London" UTLA covering all 33 London Boroughs. We collect
    each of these from the ONS sheet, derive the NHS region from the
    surrounding NUTS-1 Region row, and add a synthetic Greater London
    row (whose population is the sum of the 33 London Boroughs).

    Returns:
        DataFrame with columns ``utla_name``, ``utla_canonical``,
        ``nuts_code``, ``nhs_code``, ``population``.
    """
    df = pd.read_excel(ONS_FILE, sheet_name="MYE2 - Persons", skiprows=7)
    df = df[["Code", "Name", "Geography", "All ages"]].dropna(subset=["Code"])

    utla_geos = {"County", "Metropolitan County", "Unitary Authority"}
    records: list[dict] = []
    current_nuts_code: str | None = None
    london_boroughs: list[int] = []
    for _, row in df.iterrows():
        geo = row["Geography"]
        if geo == "Region":
            current_nuts_code = row["Code"]
            continue
        if geo == "Country":
            current_nuts_code = None
            continue
        if geo in utla_geos and current_nuts_code is not None:
            nhs_code = _nuts_to_nhs(current_nuts_code)
            if nhs_code is None:
                continue
            records.append(
                {
                    "utla_name": row["Name"],
                    "utla_canonical": _canonical_name(row["Name"]),
                    "nuts_code": current_nuts_code,
                    "nhs_code": nhs_code,
                    "population": int(row["All ages"]),
                }
            )
        elif geo == "London Borough" and current_nuts_code == "E12000007":
            london_boroughs.append(int(row["All ages"]))

    # Synthesise a "Greater London" UTLA = sum of London Boroughs.
    records.append(
        {
            "utla_name": "Greater London",
            "utla_canonical": "greater london",
            "nuts_code": "E12000007",
            "nhs_code": "Y56",
            "population": sum(london_boroughs),
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
# IMD 2019 -> population-weighted regional mean score
# ---------------------------------------------------------------------------


def regional_imd(la_lookup: pd.DataFrame) -> pd.DataFrame:
    """Compute the population-weighted mean IMD score per NHS region.

    The IMD is published at LSOA level; each LSOA carries its parent LA
    code, so we join through the LA -> NHS-region table.

    Args:
        la_lookup: LA-level lookup from :func:`build_la_lookup`.

    Returns:
        DataFrame indexed by NHS code with ``imd_mean_score`` (LSOA-mean)
        and ``imd_pop_weighted_score`` (population-weighted).
    """
    imd = pd.read_csv(
        IMD_FILE,
        usecols=[
            "LSOA code (2011)",
            "Local Authority District code (2019)",
            "Index of Multiple Deprivation (IMD) Score",
        ],
    )
    imd.columns = ["lsoa_code", "la_code", "imd_score"]
    la_to_nhs = la_lookup.set_index("la_code")[["nhs_code", "population"]]
    imd = imd.merge(la_to_nhs, left_on="la_code", right_index=True, how="left")
    imd = imd.dropna(subset=["nhs_code"])

    # LSOA-level IMD is unweighted within the LA; we use the LA population
    # divided by the LA's LSOA count as a per-LSOA weight (effectively a
    # uniform weight per LSOA within an LA, scaled by LA size).
    lsoa_counts = imd.groupby("la_code")["lsoa_code"].count()
    imd["lsoa_weight"] = imd["population"] / imd["la_code"].map(lsoa_counts)

    def _agg(sub: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "imd_mean_score": float(sub["imd_score"].mean()),
                "imd_pop_weighted_score": float(
                    (sub["imd_score"] * sub["lsoa_weight"]).sum()
                    / sub["lsoa_weight"].sum()
                ),
            }
        )

    return imd.groupby("nhs_code").apply(_agg)


# ---------------------------------------------------------------------------
# Google Mobility -> daily series per NHS region
# ---------------------------------------------------------------------------


def regional_mobility(utla_lookup: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Google Mobility from UTLA-level to NHS region, daily.

    Google reports at UTLA granularity (151 GB UTLAs including "Greater
    London" as one entity and the six Metropolitan Counties as single
    entities). We canonicalise names on both sides and inner-join, then
    take the population-weighted mean per (NHS region, date). Mobility
    is forward-shifted by ``MOBILITY_LAG_DAYS`` so the row for date
    ``t`` is aligned with hospital outcomes at ``t + lag``.

    Args:
        utla_lookup: UTLA-level lookup from :func:`build_utla_lookup`.

    Returns:
        Long-form DataFrame with columns
        ``date, nhs_code, retail_recreation, grocery_pharmacy, parks,
        transit_stations, workplaces, residential``.
    """
    frames: list[pd.DataFrame] = []
    cols_needed = ["sub_region_1", "sub_region_2", "date", *MOBILITY_COLS]
    for f in MOBILITY_FILES:
        frames.append(pd.read_csv(f, usecols=cols_needed))
    mob = pd.concat(frames, ignore_index=True)

    # UTLA-level rows: sub_region_1 set, sub_region_2 NaN.
    mob = mob[mob["sub_region_1"].notna() & mob["sub_region_2"].isna()].copy()
    mob = mob.drop(columns=["sub_region_2"])
    mob = mob.rename(columns={"sub_region_1": "utla_name", **MOBILITY_COLS})
    mob["utla_canonical"] = mob["utla_name"].map(_canonical_name)

    mob = mob.merge(
        utla_lookup[["utla_canonical", "nhs_code", "population"]],
        on="utla_canonical",
        how="inner",
    )

    mob["date"] = pd.to_datetime(mob["date"])
    value_cols = list(MOBILITY_COLS.values())

    def _weighted(group: pd.DataFrame) -> pd.Series:
        w = group["population"].astype(float)
        out = {}
        for col in value_cols:
            v = group[col]
            mask = v.notna()
            if mask.sum() == 0 or w[mask].sum() == 0:
                out[col] = float("nan")
            else:
                out[col] = float((v[mask] * w[mask]).sum() / w[mask].sum())
        return pd.Series(out)

    regional = (
        mob.groupby(["nhs_code", "date"]).apply(_weighted).reset_index()
    )

    # Apply +21 day forward shift: the mobility observed at date `t`
    # aligns with hospital outcomes at `t + 21`. We shift the index forward,
    # which is equivalent to subtracting the lag from the recorded date.
    regional["date"] = regional["date"] + pd.Timedelta(days=MOBILITY_LAG_DAYS)
    regional = regional.sort_values(["nhs_code", "date"]).reset_index(drop=True)
    return regional


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not ONS_FILE.exists():
        raise FileNotFoundError(
            f"ONS file not found: {ONS_FILE}. Run download_external_data.py."
        )

    print("Building LA-District lookup from ONS MYE2-Persons sheet...")
    la_lookup = build_la_lookup()
    print(f"  {len(la_lookup):,} English LA Districts -> {la_lookup['nhs_code'].nunique()} NHS regions")

    print("Building UTLA lookup from ONS MYE2-Persons sheet...")
    utla_lookup = build_utla_lookup()
    print(f"  {len(utla_lookup):,} English UTLAs -> {utla_lookup['nhs_code'].nunique()} NHS regions")

    print("Aggregating ONS population to NHS region totals...")
    pop = regional_population(la_lookup)

    print("Aggregating IMD 2019 (LSOA) to NHS region (population-weighted)...")
    imd = regional_imd(la_lookup)

    static = pd.DataFrame({
        "region_code": list(NHS_REGION_NAMES),
        "region_name": [NHS_REGION_NAMES[c] for c in NHS_REGION_NAMES],
        "population": [int(pop[c]) for c in NHS_REGION_NAMES],
        "imd_mean_score": [float(imd.loc[c, "imd_mean_score"]) for c in NHS_REGION_NAMES],
        "imd_pop_weighted_score": [
            float(imd.loc[c, "imd_pop_weighted_score"]) for c in NHS_REGION_NAMES
        ],
    })
    static_path = PROCESSED_DIR / "regional_static.csv"
    static.to_csv(static_path, index=False)
    print(f"Wrote {static_path}")
    print(static.to_string(index=False))

    print("\nAggregating Google Mobility (UTLA) to NHS region (population-weighted)...")
    mobility = regional_mobility(utla_lookup)
    n_unique_regions = mobility["nhs_code"].nunique()
    print(
        f"  {len(mobility):,} region-days covered, {n_unique_regions} regions, "
        f"date range {mobility['date'].min().date()} -> {mobility['date'].max().date()}"
    )
    if n_unique_regions != 7:
        raise RuntimeError(
            f"Expected 7 NHS regions in mobility output; got {n_unique_regions}. "
            f"Check the UTLA-name join coverage."
        )
    mobility_path = PROCESSED_DIR / "regional_mobility.csv"
    mobility.to_csv(mobility_path, index=False)
    print(f"Wrote {mobility_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
