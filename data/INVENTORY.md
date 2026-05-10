# Data Inventory

This file documents the contents of `data/` and the rationale for each file. The repository combines:

1. Raw NHS England XLSX archives (downloaded by `scripts/download_nhs_data.py`).
2. Pre-processed NHS regional time series at the root of `data/` (carried over from prior MSAGAT-Net work — same NHS England source, but already aggregated and 7-day rolling-averaged).
3. International epidemic datasets in `data/external/` retained as benchmarking material for stretch generalisation experiments.
4. Empty subdirectories (`processed/`, `graphs/`, `legacy/`) reserved for derived artefacts.

Last updated: 2026-05-10.

---

## `data/raw/` — NHS England XLSX archives

Fetched by `scripts/download_nhs_data.py` on 2026-05-10. SHA-256 hashes are recorded in [data/raw/MANIFEST.txt](raw/MANIFEST.txt).

| File | Period | Size |
|---|---|---|
| `COVID-19-daily-admissions-and-beds-20210406-DQnotes.xlsx` | 1 Aug 2020 – 6 Apr 2021 | 195 KB |
| `COVID-19-daily-admissions-and-beds-20211207-20210407-20210930-DQnotes.xlsx` | 7 Apr 2021 – 30 Sep 2021 | 73 KB |
| `COVID-19-daily-admissions-and-beds-20220512-211001-220331-v2.xlsx` | 1 Oct 2021 – 31 Mar 2022 | 81 KB |
| `COVID-19-daily-admissions-and-beds-20220831-v2_DQnotes.xlsx` | 1 Apr 2022 – 31 Aug 2022 | 97 KB |

Together these cover the full daily-publication period (1 Aug 2020 – 31 Aug 2022) at NHS England regional and national level. NHS England switched to weekly publication after 31 Aug 2022, so this is the natural cap for daily-resolution analysis.

**Source:** <https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-hospital-activity/>

---

## NHS-related pre-processed files (root of `data/`)

| File | Description | Status |
|---|---|---|
| [data/nhs_timeseries.txt](nhs_timeseries.txt) | NHS ICU / mechanical-ventilation bed occupancy, 895 days × 7 NHS England regions, 7-day rolling-averaged. The same `NHS-ICUBeds` benchmark used in MSAGAT-Net. | Keep. Validated headline forecasting target. May be superseded by a daily-resolution version derived from `data/raw/` and written to `data/processed/regional_daily.csv`. |
| [data/nhs-adj.txt](nhs-adj.txt) | 7×7 NHS England region adjacency matrix. | Keep. To be copied / canonicalised into `data/graphs/nhs_region_adj.txt`. |
| [data/ltla_timeseries.txt](ltla_timeseries.txt) | UK Local Tier Local Authority time series (372 LTLAs, ~5 MB). | Keep. Candidate finer-grained generalisation dataset for Experiment E3 if the official acute-trust list proves slow to scrape. |
| [data/ltla-adj.txt](ltla-adj.txt) | LTLA adjacency matrix. | Keep alongside `ltla_timeseries.txt`. |

**AIIM overlap watch.** `nhs_timeseries.txt` is the dataset MSAGAT-Net uses. To produce observationally-distinct inputs for the UKCI paper, derive a daily-resolution version from the freshly-downloaded XLSX archives in `data/raw/` and use that as the proposed-model training input. Keep the rolling-averaged file only as a sanity-check fallback.

---

## `data/external/` — international epidemic datasets

Carried over from prior MSAGAT-Net experiments. **Not used in the headline NHS analysis.** Retained as stretch generalisation experiment material — if the main pipeline finishes ahead of schedule, an "international forecast generalisation" appendix table runs the proposed PINN-GRU on these datasets and reports MAE/RMSE per country, providing a defensible response to "single-country" reviewer complaints.

| File | Country / scope | Notes |
|---|---|---|
| `australia-covid.txt`, `australia-adj.txt` | Australia COVID-19, 8 states | Same source family as MSAGAT-Net's Australia benchmark. |
| `japan.txt`, `japan-adj.txt` | Japan prefecture-level influenza, 47 prefectures | MSAGAT-Net's Japan benchmark. |
| `spain-covid.txt`, `spain-adj.txt`, `spain-label.csv` | Spain COVID-19 | International benchmark candidate. |
| `region785.txt`, `region-adj.txt` | US regional, 10 regions | MSAGAT-Net's Region785 benchmark. |
| `state360.txt`, `state-adj-49.txt`, `state-adj-50.txt` | US state-level, 49/50 states | MSAGAT-Net's State360 benchmark. |

**Decision rule.** Use these only if E1–E4 are complete by 26 May (Day 17) and an extra appendix is genuinely useful. Otherwise leave them in place as repository documentation that the proposed pipeline is forecasting-backend agnostic.

---

## Empty subdirectories (output targets)

| Directory | Purpose |
|---|---|
| [data/processed/](processed/) | Output of `scripts/build_regional_dataset.py` — tidy daily-resolution NHS regional CSV. |
| [data/graphs/](graphs/) | Canonical adjacency matrices and travel-time matrices for the optimisation module (NHS region adjacency, centroid-distance × 1.3 detour `T_ij`, etc.). |
| [data/legacy/](legacy/) | Older snapshots of NHS / LTLA data superseded by fresh downloads. Currently empty. |

Each directory contains a `.gitkeep` so that an empty directory state survives `git clean`.

---

## `.gitignore` policy

- `data/raw/*.xlsx` — gitignored. Public NHS source; we link to the portal rather than redistribute the bytes.
- `data/processed/*.csv` and `data/processed/*.md` — gitignored. Derived artefacts; reproducible from `data/raw/` via `build_regional_dataset.py`.
- `data/external/` and `data/legacy/` — **not** gitignored. These are committed benchmarking artefacts.
- `data/INVENTORY.md` (this file) — committed.
- `.gitkeep` files — committed.

---

## Reproducibility

To reproduce the data state of this directory from scratch:

```powershell
# 1. Re-fetch raw NHS England archives
python scripts/download_nhs_data.py

# 2. Harmonise into a tidy daily regional CSV (TODO: implement)
python scripts/build_regional_dataset.py
```

`scripts/download_nhs_data.py` writes `data/raw/MANIFEST.txt` recording each archive's URL, size, and SHA-256 hash. If a hash differs from the one recorded in this manifest, NHS England has revised the historical data — in that case, re-derive `regional_daily.csv` and document the revision in this inventory file.
