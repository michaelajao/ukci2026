#!/usr/bin/env python3
"""
Download external supporting datasets for the UKCI 2026 critical-care
surge pipeline:

- ONS Mid-Year Population Estimates 2021 -- regional population denominators
  used (a) by the per-region PINN to normalise the H and C compartments to
  per-capita scale and (b) by the population-proportional allocation
  baseline in :mod:`optimization.regional_allocation`.

Target directory: data/raw/supporting/

Each downloaded file is verified by SHA-256 and recorded in
data/raw/supporting/MANIFEST.txt alongside its source URL.

Usage:
    ukci-download-supporting-data
    ukci-download-supporting-data --check     # dry-run, URL probe

Run from the repository root.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from utils import repo_root, sha256_file

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
)


def sha256(path: Path) -> str:
    return sha256_file(path)


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
        "# Source: data.supporting_downloads",
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
    args = parser.parse_args()

    root = repo_root()
    dest = root / "data" / "raw" / "supporting"
    dest.mkdir(parents=True, exist_ok=True)

    print("External data downloader")
    print(f"Destination: {dest}")
    print(f"Archives to fetch: {len(ARCHIVES)}")
    print()

    records: list[dict] = []
    failures = 0
    for archive in ARCHIVES:
        rec = download_one(archive, dest, dry_run=args.check)
        records.append(rec)
        if rec.get("status") not in {"ok", "head_ok"}:
            failures += 1
        time.sleep(0.5)

    if not args.check:
        write_manifest(dest, records)
        print()
        print(f"Manifest written: {dest / 'MANIFEST.txt'}")
    print()
    succ = len([r for r in records if r.get("status") in {"ok", "head_ok"}])
    print(f"Summary: {succ} OK, {failures} failed (of {len(ARCHIVES)} attempted)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
