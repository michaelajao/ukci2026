# Project Status — NHS Critical-Care Forecast-to-Decision

Living working doc. Update as decisions are made. (Cross-session memory also held
in the Claude memory dir: paper-landscape, ukci-paper-state.)

## Papers (three, kept distinct)
1. **MSAGAT-Net** — Elsevier, in review. Graph-attention forecasting; NHS-ICUBeds is 1 of 6 datasets. Forecasting only. **Do not overlap** (no GNN in the other two).
2. **UKCI 2026 conference** (`docs/paper/`) — forecast-to-decision: PINN-SEIRD + cost-asymmetric pinball + robust LP, **7 NHS regions**. 12-page Springer cap — **now 12 pp, compiles clean** (Fig 3 RMSE-CI plot removed as redundant after recalibration; conclusion/ablation/E5/regional-structure prose compressed; compartment diagram shrunk). Authors: **Ajao-Olarinoye, Babatunde, Sadeghimanesh** (Vasile Palade removed — UKCI Co-Chair, avoids COI).
3. **Journal** (`docs/journal/`) — realistic optimisation extension at **ICB scale**. Authors: **Ajao-Olarinoye, Babatunde, Palade, Sadeghimanesh, Fei He, Petra A. Wark, Matthew England**.

## Key decisions
- **Demand metric:** conference = MV beds (region-only). Journal = **adult critical-care beds occupied** (only metric public below region; comes with real measured capacity). [confirm]
- **Journal target:** Health Care Management Science (best fit — verify Springer OA agreement with oa.lib@coventry.ac.uk) OR **EJOR / Computers & OR** (Elsevier hybrid, APC fully covered by Coventry). IEEE not a fit; INFORMS/MDPI not covered.
- No GNN forecaster (overlaps MSAGAT-Net). Optimisation is the differentiator.

## Forecasting fix (feeds both papers)
- **Problem:** PinnGRU over-predicts at long horizons (bias +27 beds at h=28); beaten by ARIMA & seasonal-naive at h=21, h=28.
- **Fix:** (1) per-origin **bias recalibration** (leakage-free; de-bias + calibrate the median); (2) **blend with seasonal-naive**, weight decreasing with horizon.
- **DONE — integrated + regenerated.** Reproducible module `src/forecasting/recalibrate.py` writes `forecasts_pinn_gru_cal.parquet`, registered as the proposed model in `evaluation.forecast_evaluation`. Official region-macro RMSE h7/14/21/28: **10.56 / 13.34 / 20.79 / 28.54** (raw was 11.0/14.5/24.2/34.6).
- **Confirmed on standard metrics** (RMSE, MAE, MAPE, MASE, WIS-80): best at h7/14/21; MAPE best at all four horizons; overall **MASE 0.56 — lowest (beats ARIMA 0.61, seasonal-naive 0.63)**; at h28 ties ARIMA, seasonal-naive marginally best. Underestimation rate 10.3%→37% (calibrated median).
- **Table 1 now reports only standard metrics** (RMSE by horizon + MASE). The underestimation rate moved to prose as a calibration observation (per request: no custom/"made-up" metrics in the table). Abstract, intro contribution 1, new §3.5 (recalibration), §6.1, and conclusion all updated. Compiles clean (still 13 pp — trim pending).
- Optional later: freeze the blend weight on the Delta validation period (needs validation-period forecasts) for a marginally better h28.

## Data
- **Conference (have it):** NHS COVID-19 Hospital Activity, `mv_beds`, 7 regions, Aug 2020–Aug 2022.
- **Journal (to source):** NHS England UEC SitRep "Critical Care and General & Acute Beds" — trust-level adult critical-care **occupancy + open beds (capacity)**, daily, Mar 2020–present → aggregate to **42 ICBs**. Plus ICB centroids (ONS geoportal), ICB populations (LSOA→ICB lookup), trust→ICB mapping (ODS).
- ICB scale-up effort ~2–3 weeks; biggest blocker = demand-unit switch + SitRep parsing/harmonisation.
- **Data progress (this session):** downloaded the consolidated `Beds-publication-Timeseries-March-2020-April-2026.xlsx` (73 KB, in `data/raw/`). It is **England-aggregate only** — sheets "Timeseries type 1 acute trusts" and "Timeseries all acute trusts" (average available + occupied beds, Mar 2020–Apr 2026). Useful as a national reference, **not per-trust**. The **trust-level data lives in the per-month SitRep files** (year pages, e.g. `.../...-2021-22/`; direct URLs `.../sites/2/2022/05/YYYYMM-Month[-YYYY]-sitrep-data-FINAL.xlsx`).
- **Monthly file inspected** (`data/raw/uec_sitrep_202112.xlsx`, Dec 2021). Layout: 14 metadata rows, then (a) ENGLAND + the 7 NHS regions with the full bed-metric set, then (b) a per-trust block (~125 trusts) with columns {Region, Trust **NAME**, Adult critical care beds **available**, Adult critical care beds **occupied**}. Two CONSTRAINTS to decide on:
  1. Values are **monthly averages** ("Average daily number"), not daily — so the public trust/ICB critical-care series is **monthly** (~24 points over 2020–22) and smooths the surge peaks. Daily trust-level data is not publicly downloadable.
  2. Trusts are identified by **name only (no ODS code)** → a trust-name→ICB mapping is needed (fuzzy/name match, then ODS→ICB).
- Upside: "available" beds = **real measured capacity** (better than the conference Delta-peak proxy); 7-region adult-critical-care series is directly available too.
- **Parser DONE** — `src/data/sitrep_dataset.py` (run `python -m data.sitrep_dataset`). Stacks the monthly workbooks → `data/processed/sitrep_trust_monthly.csv` (**130 trusts × 12 months, 2021-22**) and `sitrep_region_monthly.csv` (7 regions × 12 months), with columns `cc_available` (capacity) and `cc_occupied` (demand). England adult-CC occupancy peaks **Nov 2021 (80%, 3220/4005 beds)** — the peak-month stress scenario (note: CC peak is the Delta tail, not Omicron, since Omicron was milder for critical care).
- **Trust→ICB mapping DONE + verified** — `src/data/trust_icb_map.py`. Source: NHS England `Trust-ICB-Attribution-File.xls` (direct provider→ICB; downloaded). 121 exact name matches; the 9 merger/rename trusts (Homerton, Salford Royal, York Teaching, Pennine Acute, Royal Devon & Exeter, Northern Devon, Northern Care Alliance, South Warwickshire, West Herts) verified against authoritative sources via a workflow and pinned in an `OVERRIDE` table. Result: **130/130 trusts → 42 ICBs, 0 unmatched**. Outputs `trust_icb_map.csv` + **`sitrep_icb_monthly.csv` (42 ICBs × 12 months, cc_available/cc_occupied; peak Nov 2021, 3220 beds)**.
- **42-ICB OPTIMISATION BUILT + RUNS** — `src/optimization/icb_allocation.py`. ICB centroids from ONS geoportal `returnCentroid` API (`icb_centroids.csv`, 42/42 matched); Haversine distance/travel matrix; capacity = median monthly available; demand scenarios = system low/median/high occupancy months (high = Nov 2021 peak). Reuses the dimension-agnostic `AllocationProblem` + LP solvers — confirming the solver is scale-agnostic. **Results:** at observed demand the system has slack (E[u]=0 everywhere — the Delta-tail CC peak did not exceed capacity, an honest finding); under a **×1.3 stress wave** the robust LP covers all demand with **328 surge beds and zero transfer**, vs deterministic (median-only) E[u]=12/worst=60 and status-quo worst=229. **Greedy takes 67 s at 42 ICBs (vs ~10 s at 7 regions); the LP solves in <0.2 s** — exactly the scale where the exact-vs-metaheuristic comparison becomes meaningful (the journal's point). 1304/1722 mutual-aid links within 240 min.
- **Facility-opening MILP DONE** — optional `facility_opening`/`open_cost` flag in `_build_milp`/`solve_robust` (default OFF, so conference results are unchanged); binary "open a surge unit per node", fixed opening cost competes with beds for the budget. At ×1.3 stress, raising the per-unit opening cost concentrates surge into fewer ICBs (**42 → 30 → 16 opened, all still covering demand**) — a clean location-allocation result.
- **KEY FINDING (reframes the journal's solver story): exact MILP solves in 0.3–1.6 s at 42 ICBs.** So 42 ICBs does NOT make exact solution expensive; the "exact-vs-metaheuristic" comparison the journal wanted needs **trust scale (~140 nodes)** or a harder problem (multi-period planning / integer transfers). The slow method is the *greedy heuristic* (120 s under stress), not the MILP. **Decision needed:** scale to ~140 trusts for the metaheuristic comparison, or drop it and let the facility-location + robust + equity formulation stand on its own.
- **Remaining:** real ICB populations (MYE) for equity + pop-prop baseline (currently placeholder = capacity); correlated demand scenarios + the forecaster; trust-scale instance if pursuing metaheuristics; extend SitRep to 2020-21 & 2022-23.

## Open tasks
- [ ] Integrate forecaster recalibration + leakage-free blend; regenerate Table 1.
- [ ] Source + harmonise ICB SitRep data (real capacity + demand).
- [x] Trim conference paper to 12 pages (done — no smaller refs; removed Fig 3 + compressed tail prose).
- [ ] Confirm journal target + verify Springer OA coverage.
- [ ] Confirm affiliations: Fei He, Petra A. Wark, Matthew England, AmirHosein Sadeghimanesh.
- [ ] Set journal document class once venue chosen.

## Build / compile
- Conference: `latexmk -cd -pdf -outdir=out docs/paper/main.tex` (MiKTeX). Compiles clean.
- Journal skeleton: `docs/journal/main.tex` (article class placeholder; swap to venue class).
- Python env: `conda run -n pyt_env python ...`.
