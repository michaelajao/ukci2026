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
- **Data progress (this session):** downloaded the consolidated `Beds-publication-Timeseries-March-2020-April-2026.xlsx` (73 KB, in `data/raw/`). It is **England-aggregate only** — sheets "Timeseries type 1 acute trusts" and "Timeseries all acute trusts" (average available + occupied beds, Mar 2020–Apr 2026). Useful as a national reference, **not per-trust**. The **trust-level daily data lives in the per-month SitRep files** (linked from the year pages, e.g. `.../critical-care-and-general-acute-beds-...-2021-22/`), which must be downloaded per month and aggregated trust→ICB. **Next step:** grab one monthly file, inspect the trust-level layout, then build the trust→ICB map + parser.

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
