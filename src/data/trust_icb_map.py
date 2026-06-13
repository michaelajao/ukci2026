"""Build trust-name -> ICB mapping and aggregate the SitRep series to ICBs.

Source: NHS England "Trust-ICB-Attribution-File.xls" (sheet 'ICB Mapping'),
a direct provider->ICB attribution (ODS code, provider name, ICB code/name,
region). We match the 130 SitRep acute-trust NAMES to it (the SitRep has names
only, no ODS code), then aggregate sitrep_trust_monthly.csv to 42 ICBs.

Outputs:
  data/processed/trust_icb_map.csv      trust_name, icb_name, match_type, score
  data/processed/sitrep_icb_monthly.csv month, icb_name, cc_available, cc_occupied
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

from utils import repo_root

ROOT = repo_root()
ATTR = ROOT / "data" / "raw" / "trust_icb_attribution.xls"
TRUST_CSV = ROOT / "data" / "processed" / "sitrep_trust_monthly.csv"
PROC = ROOT / "data" / "processed"

# Trusts whose SitRep name does not equal the attribution-file name (mergers /
# renames), with the host ICB verified against authoritative sources (ICB/trust
# websites, NHS England, NHS ODS). These override name matching.
OVERRIDE = {
    "Homerton University Hospital NHS Foundation Trust": "NHS North East London Integrated Care Board",
    "Northern Care Alliance NHS Ft": "NHS Greater Manchester Integrated Care Board",
    "Salford Royal NHS Foundation Trust": "NHS Greater Manchester Integrated Care Board",
    "South Warwickshire NHS Foundation Trust": "NHS Coventry And Warwickshire Integrated Care Board",
    "West Hertfordshire Hospitals NHS Trust": "NHS Hertfordshire And West Essex Integrated Care Board",
    "York Teaching Hospital NHS Foundation Trust": "NHS Humber And North Yorkshire Integrated Care Board",
    "Northern Devon Healthcare NHS Trust": "NHS Devon Integrated Care Board",
    "Pennine Acute Hospitals NHS Trust": "NHS Greater Manchester Integrated Care Board",
    "Royal Devon and Exeter NHS Foundation Trust": "NHS Devon Integrated Care Board",
}


def norm(s: str) -> str:
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_attribution() -> dict[str, str]:
    a = pd.read_excel(ATTR, sheet_name="ICB Mapping", header=7, engine="xlrd")
    a = a.dropna(axis=1, how="all")
    # identify name column (most 'trust' values) and ICB column ('integrated care board')
    def frac(col, kw):
        v = a[col].astype(str).str.lower()
        return v.str.contains(kw, na=False).mean()
    name_col = max(a.columns, key=lambda c: frac(c, "trust"))
    icb_col = max(a.columns, key=lambda c: frac(c, "integrated care board"))
    print(f"attribution: name_col={name_col!r}  icb_col={icb_col!r}  rows={len(a)}")
    out = {}
    for _, row in a.iterrows():
        nm, icb = row[name_col], row[icb_col]
        if pd.isna(nm) or pd.isna(icb):
            continue
        out[norm(nm)] = str(icb).strip()
    return out


def main() -> int:
    attr = load_attribution()
    keys = list(attr.keys())
    trusts = sorted(pd.read_csv(TRUST_CSV)["trust_name"].unique())

    rows, unmatched = [], []
    for t in trusts:
        if t in OVERRIDE:
            rows.append({"trust_name": t, "icb_name": OVERRIDE[t], "match_type": "verified", "score": 1.0})
            continue
        nt = norm(t)
        if nt in attr:
            rows.append({"trust_name": t, "icb_name": attr[nt], "match_type": "exact", "score": 1.0})
            continue
        cand = difflib.get_close_matches(nt, keys, n=1, cutoff=0.80)
        if cand:
            score = difflib.SequenceMatcher(None, nt, cand[0]).ratio()
            rows.append({"trust_name": t, "icb_name": attr[cand[0]],
                         "match_type": "fuzzy", "score": round(score, 3)})
        else:
            unmatched.append(t)
            rows.append({"trust_name": t, "icb_name": pd.NA, "match_type": "none", "score": 0.0})

    m = pd.DataFrame(rows)
    m.to_csv(PROC / "trust_icb_map.csv", index=False)
    n_exact = (m.match_type == "exact").sum()
    n_verified = (m.match_type == "verified").sum()
    n_fuzzy = (m.match_type == "fuzzy").sum()
    print(f"\nMatched: exact={n_exact}, verified={n_verified}, fuzzy={n_fuzzy}, "
          f"none={len(unmatched)} (of {len(trusts)})")
    print(f"Distinct ICBs covered: {m['icb_name'].nunique(dropna=True)}")
    if n_fuzzy:
        print("\nFUZZY matches (verify these):")
        print(m[m.match_type == "fuzzy"][["trust_name", "icb_name", "score"]].to_string(index=False))
    if unmatched:
        print("\nUNMATCHED (need manual/verification):")
        for u in unmatched:
            print("  -", u)

    # aggregate monthly series to ICB
    tm = pd.read_csv(TRUST_CSV, parse_dates=["month"])
    tm = tm.merge(m[["trust_name", "icb_name"]], on="trust_name", how="left")
    icb = (tm.dropna(subset=["icb_name"])
           .groupby(["month", "icb_name"], as_index=False)
           .agg(cc_available=("cc_available", "sum"), cc_occupied=("cc_occupied", "sum"),
                n_trusts=("trust_name", "nunique")))
    icb.to_csv(PROC / "sitrep_icb_monthly.csv", index=False)
    print(f"\nWrote {PROC / 'sitrep_icb_monthly.csv'} "
          f"({len(icb)} rows, {icb['icb_name'].nunique()} ICBs x {icb['month'].nunique()} months)")
    # ICB peak month for stress scenario
    tot = icb.groupby("month")["cc_occupied"].sum()
    print(f"Peak ICB-summed occupancy month: {tot.idxmax().date()} ({tot.max():.0f} beds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
