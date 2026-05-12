#!/usr/bin/env python3
"""
Download NHS England COVID-19 Hospital Activity archive XLSX files.

The data lives at:
    https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-hospital-activity/

NHS England published daily admissions and beds data at regional and national level
from 1 August 2020 to 31 August 2022. The data is split across three archive files
plus weekly continuation files. This script downloads them to data/raw/nhs/.

Because URL paths on england.nhs.uk include the upload year/month and have changed
historically, the script supports two modes:

    1. KNOWN_URLS mode (default): a curated list of canonical URLs.
       If any URL has changed since this script was last updated, the user is told
       and given clear remediation steps.

    2. MANUAL mode: the user pastes URLs into a config file and the script
       downloads them in order.

Usage:
    python scripts/download_nhs_data.py
    python scripts/download_nhs_data.py --manual scripts/manual_urls.txt
    python scripts/download_nhs_data.py --check  # dry run

Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:
    sys.exit(
        "This script requires the 'requests' library.\n"
        "Install via: pip install requests"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repository root (script lives in repo_root/scripts/)
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "nhs"
PORTAL_URL = (
    "https://www.england.nhs.uk/statistics/statistical-work-areas/"
    "covid-19-hospital-activity/"
)


@dataclass
class Archive:
    """One NHS England archive file to download."""
    label: str
    description: str
    canonical_filename: str
    url: str
    expected_sheets: tuple[str, ...] = ()


# Canonical URLs as published on the NHS England portal.
# These are the historical "Daily Admissions and Beds" archive files that
# cover 1 August 2020 to 31 August 2022 at regional and national level.
#
# IMPORTANT: NHS England occasionally revises historical files. If any URL
# below 404s, visit PORTAL_URL, locate the linked XLSX file, and either:
#   (a) update the url= field below, or
#   (b) run this script with --manual to paste the URL at runtime.
KNOWN_ARCHIVES: list[Archive] = [
    Archive(
        label="daily_admissions_beds_2020-08_to_2021-04",
        description=(
            "Daily Admissions and Beds, 1 August 2020 up to 6 April 2021. "
            "Regional and national level."
        ),
        canonical_filename="COVID-19-daily-admissions-and-beds-20210406-DQnotes.xlsx",
        url=(
            "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/"
            "2022/02/COVID-19-daily-admissions-and-beds-20210406-DQnotes.xlsx"
        ),
    ),
    Archive(
        label="daily_admissions_beds_2021-04_to_2021-09",
        description=(
            "Daily Admissions and Beds, 7 April 2021 up to 30 September 2021. "
            "Regional and national level."
        ),
        canonical_filename="COVID-19-daily-admissions-and-beds-20211207-20210407-20210930-DQnotes.xlsx",
        url=(
            "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/"
            "2022/02/COVID-19-daily-admissions-and-beds-20211207-20210407-20210930-DQnotes.xlsx"
        ),
    ),
    Archive(
        label="daily_admissions_beds_2021-10_to_2022-03",
        description=(
            "Daily Admissions and Beds, 1 October 2021 up to 31 March 2022. "
            "Regional and national level."
        ),
        canonical_filename="COVID-19-daily-admissions-and-beds-20220512-211001-220331-v2.xlsx",
        url=(
            "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/"
            "2022/05/COVID-19-daily-admissions-and-beds-20220512-211001-220331-v2.xlsx"
        ),
    ),
    Archive(
        label="daily_admissions_beds_2022-04_to_2022-08",
        description=(
            "Daily Admissions and Beds, 1 April 2022 up to 31 August 2022. "
            "Regional and national level. Daily publication ceased after this date; "
            "NHS England moved to weekly reporting."
        ),
        canonical_filename="COVID-19-daily-admissions-and-beds-20220831-v2_DQnotes.xlsx",
        url=(
            "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/"
            "2022/11/COVID-19-daily-admissions-and-beds-20220831-v2_DQnotes.xlsx"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return SHA-256 hex digest of file at path."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(archive: Archive, dest_dir: Path, timeout: int = 60) -> Path | None:
    """Download a single archive. Returns destination path or None on failure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / archive.canonical_filename

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {archive.canonical_filename} already exists "
              f"({dest.stat().st_size:,} bytes, sha256 {sha256sum(dest)[:16]}...)")
        return dest

    print(f"  [GET ] {archive.url}")
    try:
        r = requests.get(archive.url, stream=True, timeout=timeout, headers={
            "User-Agent": "ukci2026-research-pipeline/1.0 "
                          "(Coventry University CSM; academic use)"
        })
    except requests.RequestException as e:
        print(f"  [FAIL] network error: {e}")
        return None

    if r.status_code != 200:
        print(f"  [FAIL] HTTP {r.status_code}: {archive.url}")
        print(f"         Visit {PORTAL_URL} to find the current URL.")
        return None

    content_length = r.headers.get("Content-Length")
    if content_length:
        print(f"  [SIZE] {int(content_length):,} bytes")

    bytes_written = 0
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    if bytes_written == 0:
        print(f"  [FAIL] empty download")
        dest.unlink(missing_ok=True)
        return None

    digest = sha256sum(dest)
    print(f"  [DONE] {dest.name} ({bytes_written:,} bytes, sha256 {digest[:16]}...)")
    return dest


def parse_manual_urls(path: Path) -> list[Archive]:
    """Parse a manual-URLs file. Format: one URL per line, blank lines and # ignored."""
    archives: list[Archive] = []
    if not path.exists():
        sys.exit(f"Manual URL file not found: {path}")

    with path.open() as f:
        urls = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    for i, url in enumerate(urls, start=1):
        filename = url.rsplit("/", 1)[-1]
        archives.append(Archive(
            label=f"manual_archive_{i}",
            description=f"Manual archive {i}",
            canonical_filename=filename,
            url=url,
        ))
    return archives


def write_manifest(downloaded: Iterable[tuple[Archive, Path]]) -> None:
    """Write a manifest of downloaded files with hashes for reproducibility."""
    manifest_path = RAW_DIR / "MANIFEST.txt"
    with manifest_path.open("w") as f:
        f.write("# NHS England COVID-19 Hospital Activity download manifest\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write(f"# Source portal: {PORTAL_URL}\n\n")
        for archive, path in downloaded:
            f.write(f"label:       {archive.label}\n")
            f.write(f"description: {archive.description}\n")
            f.write(f"url:         {archive.url}\n")
            f.write(f"file:        {path.name}\n")
            f.write(f"size:        {path.stat().st_size}\n")
            f.write(f"sha256:      {sha256sum(path)}\n")
            f.write("\n")
    print(f"\nManifest written: {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual", type=Path, default=None,
        help="Path to a text file with one URL per line (overrides KNOWN_ARCHIVES).",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Dry run: print URLs without downloading.",
    )
    args = parser.parse_args()

    archives = parse_manual_urls(args.manual) if args.manual else KNOWN_ARCHIVES

    print(f"NHS England Hospital Activity downloader")
    print(f"Portal: {PORTAL_URL}")
    print(f"Destination: {RAW_DIR}")
    print(f"Archives to fetch: {len(archives)}")
    print()

    if args.check:
        for a in archives:
            print(f"  {a.label}")
            print(f"    {a.description}")
            print(f"    URL: {a.url}")
            print()
        return 0

    downloaded: list[tuple[Archive, Path]] = []
    failed: list[Archive] = []

    for archive in archives:
        print(f"[{archive.label}]")
        result = download_one(archive, RAW_DIR)
        if result is not None:
            downloaded.append((archive, result))
        else:
            failed.append(archive)
        print()

    if downloaded:
        write_manifest(downloaded)

    print(f"Summary: {len(downloaded)} downloaded, {len(failed)} failed")

    if failed:
        print("\nFAILED archives:")
        for a in failed:
            print(f"  {a.label}: {a.url}")
        print(
            f"\nRemediation: visit {PORTAL_URL}, locate the correct URL for each "
            f"failed archive, then either:\n"
            f"  (a) edit KNOWN_ARCHIVES in {Path(__file__).name}, or\n"
            f"  (b) save the URLs in a text file and run with --manual <path>"
        )
        return 1

    print("\nNext step: run scripts/build_regional_dataset.py to harmonise into a "
          "tidy CSV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
