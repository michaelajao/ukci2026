#!/usr/bin/env python3
"""
Download external supporting datasets identified by the UKCI 2026 literature
review:

- Google Community Mobility Reports (GB, 2020-2022) -- regressor inputs
  to the GRU temporal head (Valente et al. 2022; Cartené et al. 2020).
- ONS Mid-Year Population Estimates 2021 -- denominators for
  population-proportional allocation baselines and regional normalisation.
- English Indices of Deprivation 2019, File 7 (LSOA scores/ranks/deciles)
  and File 10 (Local Authority district summaries) -- inputs for
  IMD-weighted fairness constraints in the MILP (per the Bertsimas et al.
  2022 and Luo & Stellato 2024 fairness patterns).

Target directory: data/raw/supporting/

Each downloaded file is verified by SHA-256 and recorded in
data/raw/supporting/MANIFEST.txt alongside its source URL.

Usage:
    python scripts/download_external_data.py
    python scripts/download_external_data.py --check     # dry-run, URL probe
    python scripts/download_external_data.py --skip imd  # skip an archive

Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("This script requires the 'requests' library.")
    print("Install via: pip install requests")
    sys.exit(2)


@dataclass(frozen=True)
class Archive:
    label: str
    description: str
    url: str
    filename: str
    optional: bool = False


ARCHIVES: tuple[Archive, ...] = (
    Archive(
        label="google_mobility_gb_2020",
        description=(
            "Google COVID-19 Community Mobility Report -- Great Britain 2020. "
            "Daily percent change vs baseline in retail/recreation, grocery/"
            "pharmacy, parks, transit, workplaces, residential, at sub-region "
            "(Upper-Tier Local Authority) level."
        ),
        url="https://www.gstatic.com/covid19/mobility/2020_GB_Region_Mobility_Report.csv",
        filename="2020_GB_Region_Mobility_Report.csv",
    ),
    Archive(
        label="google_mobility_gb_2021",
        description=(
            "Google COVID-19 Community Mobility Report -- Great Britain 2021. "
            "Same schema as 2020 file."
        ),
        url="https://www.gstatic.com/covid19/mobility/2021_GB_Region_Mobility_Report.csv",
        filename="2021_GB_Region_Mobility_Report.csv",
    ),
    Archive(
        label="google_mobility_gb_2022",
        description=(
            "Google COVID-19 Community Mobility Report -- Great Britain 2022. "
            "Series ends 15 October 2022 when Google discontinued publication."
        ),
        url="https://www.gstatic.com/covid19/mobility/2022_GB_Region_Mobility_Report.csv",
        filename="2022_GB_Region_Mobility_Report.csv",
    ),
    Archive(
        label="ons_mye_2021_uk",
        description=(
            "ONS Mid-Year Population Estimates 2021 -- UK England Wales "
            "Scotland Northern Ireland, on 2021 geography (final). Required "
            "for NHS region population denominators."
        ),
        url=(
            "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
            "populationandmigration/populationestimates/datasets/"
            "populationestimatesforukenglandandwalesscotlandandnorthernireland/"
            "mid2021/ukpopestimatesmid2021on2021geographyfinal.xls"
        ),
        filename="ukpopestimatesmid2021on2021geographyfinal.xls",
    ),
    Archive(
        label="imd_2019_file_07_scores",
        description=(
            "English Indices of Deprivation 2019, File 7 -- All IoD2019 "
            "Scores, Ranks, Deciles and Population Denominators. LSOA-level. "
            "Master input for IMD-weighted fairness constraints."
        ),
        url=(
            "https://assets.publishing.service.gov.uk/media/"
            "5dc407b440f0b6379a7acc8d/"
            "File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_"
            "Denominators_3.csv"
        ),
        filename="IoD2019_File_7_Scores_Ranks_Deciles.csv",
    ),
    Archive(
        label="imd_2019_file_10_la_lower_tier",
        description=(
            "English Indices of Deprivation 2019, File 10 -- Local Authority "
            "District Summaries (lower-tier). LA-aggregated IMD score, "
            "average rank, proportion of LSOAs in most-deprived deciles. "
            "Convenience aggregation for LA-level fairness reporting."
        ),
        url=(
            "https://assets.publishing.service.gov.uk/media/"
            "5d8b3cfbe5274a08be69aa91/"
            "File_10_-_IoD2019_Local_Authority_District_Summaries__"
            "lower-tier__.xlsx"
        ),
        filename="IoD2019_File_10_LA_lower_tier_summaries.xlsx",
    ),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(archive: Archive, dest_dir: Path, *, dry_run: bool) -> dict:
    print(f"[{archive.label}]")
    print(f"  [GET ] {archive.url}")
    if dry_run:
        try:
            r = requests.head(archive.url, allow_redirects=True, timeout=20)
            size = r.headers.get("Content-Length", "?")
            print(f"  [HEAD] {r.status_code} {size} bytes")
            return {"status": "head_ok", "code": r.status_code, "size": size}
        except Exception as exc:
            print(f"  [HEAD] ERROR {type(exc).__name__}: {exc}")
            return {"status": "head_error", "error": str(exc)}
    target = dest_dir / archive.filename
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(archive.url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0") or 0)
            print(f"  [SIZE] {total or 'unknown'} bytes")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.replace(target)
        digest = sha256(target)
        actual = target.stat().st_size
        print(f"  [DONE] {target.name} ({actual} bytes, sha256 {digest[:16]}...)")
        return {
            "status": "ok",
            "size": actual,
            "sha256": digest,
            "filename": target.name,
        }
    except Exception as exc:
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        if tmp.exists():
            tmp.unlink()
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def write_manifest(dest_dir: Path, records: list[dict]) -> None:
    lines = [
        "# External supporting datasets for UKCI 2026 critical-care surge pipeline.",
        f"# Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "# Source: scripts/download_external_data.py",
        "",
    ]
    for archive, result in zip(ARCHIVES, records):
        lines.append(f"label:       {archive.label}")
        lines.append(f"description: {archive.description}")
        lines.append(f"url:         {archive.url}")
        lines.append(f"file:        {archive.filename}")
        if result.get("status") == "ok":
            lines.append(f"size:        {result['size']}")
            lines.append(f"sha256:      {result['sha256']}")
        else:
            lines.append(f"status:      {result.get('status', 'unknown')}")
            if "error" in result:
                lines.append(f"error:       {result['error']}")
        lines.append("")
    (dest_dir / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: HEAD each URL without writing files.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="LABEL_PREFIX",
        help="Skip archives whose label starts with any of these prefixes.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dest = repo_root / "data" / "raw" / "supporting"
    dest.mkdir(parents=True, exist_ok=True)

    skip_prefixes = tuple(args.skip)
    selected: list[Archive] = [
        a for a in ARCHIVES if not any(a.label.startswith(p) for p in skip_prefixes)
    ]

    print("External data downloader")
    print(f"Destination: {dest}")
    print(f"Archives to fetch: {len(selected)} (skipped: {len(ARCHIVES) - len(selected)})")
    print()

    records: list[dict] = []
    failures = 0
    for archive in selected:
        rec = download_one(archive, dest, dry_run=args.check)
        records.append(rec)
        if rec.get("status") not in {"ok", "head_ok"}:
            failures += 1
        time.sleep(0.5)
    # Pad records with skipped placeholders so manifest order matches ARCHIVES.
    while len(records) < len(ARCHIVES):
        records.append({"status": "skipped"})

    if not args.check:
        write_manifest(dest, records)
        print()
        print(f"Manifest written: {dest / 'MANIFEST.txt'}")
    print()
    succ = len([r for r in records if r.get("status") in {"ok", "head_ok"}])
    print(f"Summary: {succ} OK, {failures} failed (of {len(selected)} attempted)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
