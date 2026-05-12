# 04 — Implementation Plan (10–31 May 2026)

This document is the approved 21-day execution plan for the UKCI 2026 paper. It complements the strategy documents:

- `01_RESEARCH_PROGRAMME.md` — overall research strategy and contributions.
- `02_METHODOLOGY.md` — full mathematical formulation.
- `03_TIMELINE.md` — original day-by-day plan (drafted 6 May; this document supersedes its day-by-day schedule starting 10 May).

---

> **Data folder reorganisation (2026-05-12).** The `data/` directory was restructured after the literature review and the MSAGAT-Net legacy review. Some narrative below refers to the *prior* layout (`data/raw/`, `data/raw/external/`, `data/external/`, loose `.txt` files at the data root). Current canonical paths are:
>
> - `data/raw/nhs/` — NHS England XLSX archives (was `data/raw/`).
> - `data/raw/supporting/` — Google Mobility, ONS, IMD (was `data/raw/external/`).
> - `data/processed/` — tidy regional CSV.
> - `data/legacy/msagat_net/` — the four loose `.txt` files plus `data/graphs/` contents (not used by UKCI 2026).
> - `data/legacy/international/` — the 12 non-UK epidemic files (was `data/external/`).
>
> See [data/INVENTORY.md](../data/INVENTORY.md) for the authoritative current layout.

## Context

Development on the UKCI 2026 conference paper (*"Decision-Aware Physics-Informed Forecasting and Metaheuristic Allocation for NHS Critical-Care Surge Capacity Under Demand Uncertainty"*) takes place inside this repository.

The repository scaffold is in place with three excellent strategy documents (`01_…`, `02_…`, `03_…`), one fully-written PINN-SEIRD module (`src/forecasting/pinn_seird.py`, ~270 lines, working smoke test), a production-ready NHS data downloader (`scripts/download_nhs_data.py`), and stubs/empty `__init__.py` for everything else.

**Confirmed inputs (10 May 2026):**

- Gate G1 passed — Vasile signed off; Abiola committed to the metaheuristic optimisation half.
- Compute — local RTX 5060 Ti 16 GB.
- Scope ambition — full programme as documented (7 forecasting baselines + proposed = 8 models in E1 after 12 May trim+restore; 7 allocation policies; 7 experiments).
- Submission deadline — 31 May 2026 via Microsoft CMT (21 days remaining).
- The 16 `.txt`/`.csv` files in `data/` are MSAGAT-Net-era artefacts. They are **kept as benchmarking material**, not deleted. The NHS and LTLA timeseries in particular may be the same NHS England source the new download script targets (or an earlier version of it) — the download is run first and we compare before deciding what to do with each file.

**Revisions following the 12 May literature review (this section supersedes the original scope in three places):**

1. **Add reduced-STAN as a Tier 3 baseline (required, not optional).** Gao et al. (*JAMIA* 2021) is the closest physics-informed deep-learning competitor and reviewers will ask. Reduced means: SIR-consistency loss head over a per-region GRU — drop the graph-attention layer so the AIIM defence holds. STAN public code at <https://github.com/v1xerunt/STAN> (JHU data is public; IQVIA claims data is on-request-only — we will use only the publicly-replicable feature subset).
2. **Add augmented ε-constraint as a Phase C method (replacing SA in the headline comparison; SA retained as Appendix-only comparator).** Kargar et al. (*Transport. Res. E* 2024). Augmented ε-constraint produces *exact* non-dominated points, complementing NSGA-II's heuristic Pareto approximation. This is paradigmatically distinct, whereas SA+GA+NSGA-II are all stochastic search.
3. **Add IMD-weighted fairness as MILP constraints (not only evaluation).** Bertsimas et al. (*NRL* 2022) and Luo & Stellato (arXiv 2024) both put fairness *into* the optimisation problem rather than only measuring it. Inter-region and intra-region bounds anchored on population share and IMD score will be added to the Pyomo model.
4. **Add counterfactual policy-impact analysis (new experiment E7).** Gao et al. (*Nat. Comm.* 2023, HOIST) demonstrate the model-derived number-needed-to-treat / cost-ratio pattern. Adapted to surge: take optimised allocation, perturb total budget ±20%, compute marginal beds-per-£ by region, report a 5-row policy table. This is the operational-impact paragraph in §6.
5. **Add Google Mobility as a covariate to the GRU temporal head.** Valente et al. (*Stat. J. IAOS* 2022) demonstrate this is high-value for ICU forecasting; Cartení et al. (*Sci. Tot. Env.* 2020) justify the 21-day forward shift.
6. **Add ethical/legal limitations paragraph in §7.** Brown et al. (*NEJM* 2020) and Liddell et al. (same correspondence) bracket the discussion of allocation-under-shortage. Three sentences, not more.

These revisions are reflected in: §1.5 (new — external features), Phase 0 (added `build_regional_features.py`), Phase 1 Tier 3 baseline list (added reduced STAN), Phase 3 (added `epsilon_constraint.py`), Phase 4 (added `run_e7.py`), §3 Baselines (updated allocation list), §4 paper-writing plan §6.4 (E7 subsection) and §7 (ethics paragraph), and the D-day schedule.

**Honest expert assessment.** This is a publishable conference paper at UKCI. The decision-aware composite loss with asymmetric underestimation penalty and the ω/φ compartmental refinement are genuinely novel-in-application; the multi-objective robust MILP with equity linearisation is solid OR. The pipeline as a whole (PINN-SEIRD → MC Dropout scenarios → NSGA-II Pareto fronts on NHS data) is untraversed ground. The optimisation half is defensible-but-not-strong for a CI venue — pre-empt MOEA/D-style reviewer questions with a one-paragraph justification of NSGA-II in §4.6. AIIM overlap risk is *perception*, not fact, given the per-region GRU choice; cite MSAGAT-Net obliquely as concurrent work in Related Work and reuse no figures or text. Implementation risk is the dominant concern. Probability of complete submission at this scope: ~70% with discipline, ~20% scope-reduced, ~10% withdrawal — driven mostly by household disruption risk flagged in `01_RESEARCH_PROGRAMME.md` §9.

---

## 1. Data inventory and reconciliation — keep, don't delete

All paths under the repository root.

**Step 0 — what we already have (verified by direct inspection).**

- `data/nhs_timeseries.txt` is **the NHS ICU/MV bed occupancy data, 895 days × 7 NHS England regions**. Values appear 7-day rolling-averaged. This is the **primary forecasting target** the new paper specifies in `02_METHODOLOGY.md` §0 ("forecasting target: mechanical ventilation bed occupancy unless stated"). It is the same NHS-ICUBeds dataset benchmarked in MSAGAT-Net.
- `data/nhs-adj.txt` is the companion 7×7 NHS region adjacency.

**Implication.** The headline MV-bed forecasting target and adjacency are already prepared. The download script is needed for *enrichment* (admissions, occupied COVID beds, and potentially raw-resolution daily MV beds) and for the chronological-by-wave split documented in `01_RESEARCH_PROGRAMME.md` §6, but not to *enable* the proposed model. Day 1–2 effort drops by roughly half.

**AIIM overlap watchout (important).** Because `nhs_timeseries.txt` IS the dataset MSAGAT-Net uses, training the new paper's PINN-GRU directly on it without enrichment risks producing numerically-similar forecast tables to MSAGAT-Net's NHS-ICUBeds results. Mitigations: (i) re-derive the regional time series at *daily resolution* from the fresh download (not rolling-averaged) so the input is observationally distinct; (ii) frame Table 2 in the new paper around horizons {7, 14, 21, 28} which MSAGAT-Net does not all use; (iii) report decision-relevant metrics (Underestimation Rate, Expected Shortage) that MSAGAT-Net does not report. This keeps the forecasting numbers genuinely different even though the underlying NHS source is shared. Confirm Vasile is comfortable with this when we re-loop him post-G1.

**Step 1 (D1) — run the NHS download first.** Execute `scripts/download_nhs_data.py` into `data/raw/`. This pulls the four NHS England COVID-19 hospital activity XLSX archives (admissions, occupied beds, MV beds) and the NHS UEC critical-care SitRep — the same source family that `nhs_timeseries.txt` derives from, but at *raw daily resolution* and with covariates the existing file lacks.

**Step 2 (D1–D2) — compare existing vs fresh.** For each of the 16 existing files, classify into one of three buckets and document the result in a new file `data/INVENTORY.md`:

| Existing file | Likely status | Action |
|---|---|---|
| `data/nhs_timeseries.txt` | **Confirmed: NHS ICU/MV bed occupancy, 895 days × 7 regions, 7-day rolling-averaged.** Primary forecasting target. Same source as MSAGAT-Net's NHS-ICUBeds. | Keep. Use as the validated MV-bed target for fast-path proposed-model training while the fresh download is processed. After fresh download is parsed, diff for date coverage and produce a daily-resolution version in `data/processed/regional_daily.csv` to avoid AIIM observational overlap. |
| `data/nhs-adj.txt` | Confirmed: 7×7 NHS region adjacency. | Keep. Move/copy into `data/graphs/nhs_region_adj.txt` so the optimisation module's `T_ij` derivation has a single canonical location. |
| `data/ltla_timeseries.txt`, `data/ltla-adj.txt` | UK Local Tier Local Authority, probably from PHE/UKHSA dashboard or NHS — ~5 MB suggests a real dataset. Useful for *trust-level* generalisation experiment (E3 trust-level) since 372 LTLAs is finer-grained than 7 regions. | Keep. Validate column structure. Potentially re-use as the trust-level scope for E3 if the official acute-trust list proves hard to scrape. |
| `data/{australia,japan,spain,region785,region-adj,state360,state-adj-49,state-adj-50,spain-label,australia-adj,japan-adj,spain-adj}.{txt,csv}` (12 files) | International / non-NHS benchmarking | Keep as **stretch generalisation experiment** material. If E1–E6 finish ahead of schedule (unlikely but possible), an "international forecast generalisation" appendix table runs PINN-GRU on Japan/Spain/Australia and reports MAE/RMSE — protects against "single-country, only NHS" reviewer complaints in §7 Discussion. Move these into `data/external/` to keep `data/` itself focused on NHS. |

**Step 3 (D2) — folder reorganisation (no deletion):**

- Create `data/external/` and move all non-NHS files into it.
- Create `data/legacy/` for any older snapshots of NHS/LTLA data superseded by the fresh download.
- `data/raw/` (NHS XLSX from downloader), `data/processed/` (regional_daily.csv from `build_regional_dataset.py`), `data/graphs/` (NHS region adjacency CSV) keep their original purpose.
- Add `.gitkeep` to every subdir.

**Step 4 — `.gitignore` update.** Verify it includes: `data/raw/*.xlsx`, `data/processed/*.csv`, `data/processed/*.md`, `results/`, `figures/`, `*.pt`, `*.pkl`, `mlruns/`, `tensorboard/`, `.venv/`. **Do not gitignore `data/external/` or `data/legacy/`** — these are committed benchmarking artefacts. Raw NHS XLSX is public, so the script can re-fetch them; we link to source, don't redistribute the bytes.

**Why this matters for the paper.** The international datasets quietly upgrade contribution 6 (reproducibility package) into a soft generalisation claim — *"the pipeline is forecasting-backend agnostic and the implementation generalises to international epidemic data with minimal modification"* — which is exactly the kind of defensive-but-honest framing that helps with reviewers asking "why only NHS?". The LTLA dataset is a credible candidate for the trust-level E3 experiment if scraping the actual NHS acute-trust list proves time-consuming.

### 1.5 External UK supporting datasets (downloaded 12 May 2026 in D2 audit)

Identified by the 12 May literature review as inputs required for the revised scope (see Context revisions 3–5). Downloaded by `scripts/download_external_data.py` into `data/raw/external/`. Documented in detail in [data/INVENTORY.md](../data/INVENTORY.md). Summary:

| Dataset | File | Use |
|---|---|---|
| Google Mobility GB 2020–2022 | three CSVs, 44.6 MB total | Forecasting covariate to GRU head; 21-day forward shift per Cartení et al. (2020) |
| ONS Mid-Year 2021 estimates | `ukpopestimatesmid2021on2021geographyfinal.xls` (1.9 MB) | Population denominators for NHS region aggregation and for population-proportional allocation baseline |
| English IMD 2019 File 7 | `IoD2019_File_7_Scores_Ranks_Deciles.csv` (9.7 MB) | LSOA-level IMD master input |
| English IMD 2019 File 10 | `IoD2019_File_10_LA_lower_tier_summaries.xlsx` (255 KB) | LA-aggregated IMD summary |

**Aggregation pipeline** (new module `scripts/build_regional_features.py`, D2–D3):

1. **UTLA → NHS region cross-walk** — author `data/graphs/utla_to_nhs_region.csv` (151 GB UTLAs filtered to English UTLAs, mapped to the 7 NHS England commissioning regions). For UTLAs spanning multiple NHS regions, canonical resolution.
2. **Google Mobility → NHS region.** Filter to England, population-weighted mean across UTLAs per NHS region per date, 21-day forward shift, output to `data/processed/mobility_regional_daily.csv`.
3. **IMD 2019 → NHS region.** Aggregate LSOA-level IMD score to NHS region via LSOA → LA → NHS region cross-walk (LSOA→LA cross-walk is included in IMD File 7 column "Local Authority District code (2019)"). Output: scalar mean IMD weight per region, used as `w^{\mathrm{IMD}}_r` parameter in MILP fairness constraints.
4. **ONS MYE 2021 → NHS region.** Population denominators per NHS region for population-proportional baseline and per-region normalisation.

These outputs feed (i) Phase 2's `composite_loss.py` and `temporal_head.py` (mobility as augmented feature), (ii) Phase 3's `milp.py` (IMD weights as fairness coefficients), and (iii) Phase 3's `heuristics.py` (population-proportional and IMD-proportional baselines).

---

## 2. Modules to build (critical-path order)

Reuse `src/forecasting/pinn_seird.py` **as-is** — do not refactor. Its `__main__` smoke test is the unit test.

Day numbers count from 10 May (D1 = Sat 10 May, D21 = Sun 31 May).

### Phase 0 — Foundations (D1–D2)

| File | LoC | Purpose |
|---|---|---|
| Run `scripts/download_nhs_data.py` | — | Pulls the four NHS England XLSX archives into `data/raw/` (done D1) |
| Run `scripts/download_external_data.py` | — | Pulls Google Mobility 2020–2022, ONS MYE 2021, English IMD 2019 Files 7+10 into `data/raw/external/` (done D2; see §1.5) |
| `scripts/build_regional_dataset.py` | ~150 | Implement the body: parse XLSX → regional tidy CSV in `data/processed/regional_daily.csv`. Reconcile region naming, handle definition changes 2021–2022, validate date contiguity |
| `scripts/build_regional_features.py` | ~200 | **NEW.** Aggregate `data/raw/external/` to NHS region level: (i) UTLA→NHS region cross-walk authored at `data/graphs/utla_to_nhs_region.csv`; (ii) population-weighted Google Mobility per (region, date) with 21-day forward shift → `data/processed/mobility_regional_daily.csv`; (iii) LSOA → LA → NHS region IMD aggregation → `data/processed/imd_regional_weights.csv`; (iv) ONS MYE → `data/processed/population_regional.csv`. Join all into `regional_daily.csv` as additional feature columns. |
| `src/data/loaders.py` | ~150 | `load_regional_csv()`, sliding-window batch generator (lookback `L=28`, multi-horizon `h ∈ {7,14,21,28}`), per-region standardisation. Now reads both NHS bed columns and the new mobility/IMD/population columns. |
| `src/data/splits.py` | ~50 | Chronological splits exactly as `01_RESEARCH_PROGRAMME.md` §6: train (Aug 2020–May 2021), val (Jun–Nov 2021), test (Dec 2021–Aug 2022) |
| `src/utils/seed.py` | ~30 | Reproducible seeding for `random`, `numpy`, `torch` |
| `src/evaluation/forecast_metrics.py` | ~150 | MAE, RMSE, sMAPE, MASE, Underestimation Rate, Expected Shortage, Peak Error, Peak Timing Error, WIS (Bracher et al. 2021) |
| `data/graphs/nhs_region_adj.csv` | — | 7×7 NHS region adjacency, documented encoding in `data/graphs/README.md` |

### Phase 1 — Forecasting baselines (D3–D5)

| File | LoC | Purpose |
|---|---|---|
| `src/forecasting/baselines.py` | ~250 | `ARIMAPerRegion` (via `statsmodels.SARIMAX`), `GRUPerRegion` (non-physics control). Common `BaselineModel` interface returning point + (optional) quantile forecasts. Persistence retained as a `_PersistenceReference` helper for sanity-check footer rows but not part of E1. |
| `configs/baselines/` | 9 YAMLs | One config per baseline + proposed; Hydra-managed |

### Phase 2 — Proposed model (D6–D8) — load-bearing

| File | LoC | Purpose |
|---|---|---|
| `src/forecasting/temporal_head.py` | ~120 | 2-layer GRU (hidden=64, dropout=0.2) over augmented features `[Ũ_r(t) ‖ X̃_r(t) ‖ x_{r,t}]`; multi-horizon decoder for h ∈ {7,14,21,28} |
| `src/forecasting/composite_loss.py` | ~120 | **THE headline contribution.** `HuberForecastLoss` + `PINNResidualLoss` (calls `pinn_seird.py`) + `AsymmetricUnderestimationLoss` (one-sided hinge) + `TemporalSmoothnessLoss`. Weights `λ_phys=0.1, λ_under=0.5, λ_smooth=0.01` (per `02_METHODOLOGY.md` §1.5) |
| `src/forecasting/uncertainty.py` | ~80 | MC Dropout with K=100 forward passes; quantile extraction at p ∈ {0.1, 0.5, 0.9}; WIS calibration |
| `src/forecasting/train.py` | ~250 | Hydra entrypoint: PINN per-region pretraining → freeze parameter network → joint training of GRU head with composite loss; checkpointing, early stopping on val WIS, MLflow logging |

### Phase 3 — Scenarios + optimisation (D8–D13) — Abiola owns most of this

| File | LoC | Purpose |
|---|---|---|
| `src/data/scenarios.py` | ~120 | 3-point Low/Median/High discrete scenarios from quantiles with weights π = (0.20, 0.60, 0.20); optional Tail at q^0.95 with π=0.05 for E5 |
| `src/optimization/data_model.py` | ~180 | Sets, parameters, decision variables exactly as `02_METHODOLOGY.md` §0; loaders for `T_ij` (centroid distance × 1.3 detour), `C_{j,h}`, `F_j`, `g_j`, budgets |
| `src/optimization/milp.py` | ~400 | Pyomo deterministic MILP per `02_METHODOLOGY.md` §3.1; equity linearisation via per-region θ_i (§3.1.3); CVaR robust extension via `λ_3 W` (§3.1.4); **IMD-weighted intra-region fairness constraints (NEW)** using `w^{\mathrm{IMD}}_r` from `data/processed/imd_regional_weights.csv` and **inter-region population-share bounds** `Θ_L` per Bertsimas et al. (2022) eq. (25); Gurobi solver with CBC fallback |
| `src/optimization/heuristics.py` | ~240 | NoSurge, Population-proportional, **IMD-proportional (NEW)**, Demand-proportional, Greedy shortage-first, Historical Nightingale (best-effort lookup) |
| `src/optimization/metaheuristics.py` | ~400 | pymoo `GA` with master-slave decomposition (binary x master + LP slave for b,z,u via Pyomo); `NSGA2` with three objectives `(f_1=cost, f_2=unmet, f_3=transfer)`; custom `SimulatedAnnealing` with single-bit-flip neighbourhood (kept as Appendix-only comparator) |
| `src/optimization/epsilon_constraint.py` | ~250 | **NEW.** Augmented ε-constraint method (Mavrotas 2009; Kargar et al. 2024) over the same Pyomo MILP. Two-objective formulation: optimise cost s.t. unmet-demand ≤ ε; sweep ε across `K=10` values to expose exact Pareto front. Augmentation term `s_k` ensures non-dominated solutions only. **Methodologically complements NSGA-II** by providing exact (not heuristic) Pareto points. |
| `src/evaluation/allocation_metrics.py` | ~150 | Total Unmet Demand, Coverage Rate, Mean Travel Burden, Max Regional Shortage Ratio, Theil Index, Worst-Case Unmet, VRS, hypervolume, spread |

### Phase 4 — Experiments (D14–D17)

| File | LoC | Purpose |
|---|---|---|
| `scripts/run_e1.py` | ~120 | E1 forecasting accuracy: **8 models** × 4 horizons × 5 seeds (Seasonal Naive, ARIMA, GRU per region, PINN-GRU, reduced STAN, N-BEATS, TFT, DeepAR) → Table 1, Figure 3 |
| `scripts/run_e2.py` | ~100 | E2 regional MILP allocation: 7 policies (added IMD-proportional) × 2 forecast sources × 3 budgets × 3 travel thresholds → Table 2, Figures 4–5 |
| `scripts/run_e3.py` | ~100 | E3 trust-level metaheuristic comparison (London + Midlands subsets) → Table 3, Figure 6. NSGA-II vs augmented ε-constraint vs GA vs (SA Appendix). |
| `scripts/run_e4.py` | ~80 | E4 Pareto front (NSGA-II overlay with exact ε-constraint points) → Figure 7 |
| `scripts/run_e5.py` | ~80 | E5 robustness to forecast quality (PINN vs ARIMA vs Oracle) → Table 4 |
| `scripts/run_e6.py` | ~80 | E6 sensitivity (scenario count × λ_2 × λ_3) → Appendix |
| `scripts/run_e7.py` | ~120 | **NEW.** E7 counterfactual policy-impact analysis (HOIST-style NNT / cost-ratio, adapted to surge): take optimised allocation, perturb total budget ∈ {0.8, 0.9, 1.0, 1.1, 1.2}, compute marginal beds-per-£ by region, IMD-weighted ranking of regions by return on incremental surge investment. → Table 5, Figure 9. Drives §6.4 of the paper. |
| `notebooks/eda.ipynb` | — | Time-series, missingness audit, regional adjacency check |
| `notebooks/results_visualisation.ipynb` | — | Pareto fronts, allocation heatmaps, forecast plots |

### Phase 5 — Paper writing + polish (D15–D21)

LaTeX in `docs/paper/` using the Springer SVProc template kept in `docs/ukci_springer_template/`. Write Methodology (§3, §4) on D15–D17 in parallel with experiments running. Results section on D18 once E1+E2+E3 are done. Discussion + Intro + Abstract on D19–D20. Camera-ready and CMT submission D21.

---

## 3. Baselines — the full set, ordered by priority

### Forecasting (Experiment E1)

Three tiers — implement in this order so we can declare scope reduction at G2 (16 May) by stopping at any point.

**Tier 1 — required floors (D3–D4, ~1.5 days):**

1. **Seasonal Naive (7-day lag)** — rule floor $\hat y_{t+h} = y_{t+h-7}$ (~30 lines). Restored 12 May after author decision: keep one trivial floor for sanity-checking the metric harness.
2. **ARIMA per region** via `statsmodels.SARIMAX` with AIC-selected order (~100 lines) — non-DL statistical floor.
3. **GRU per region** without PINN coupling — the **control** for the PINN contribution; do not skip (~150 lines).

**Tier 2 — proposed model (D5–D8, load-bearing):**

4. **PINN-GRU (proposed)** with decision-aware composite loss. Per-region (no graph coupling) — see "Graph-attention SOTA" note below for the AIIM-defence rationale.

**Tier 3 — modern competitors (D9, ~2 days, AIIM-safe = no graph attention):**

5. **Reduced STAN (PINN-DL competitor, REQUIRED).** Gao et al. (*JAMIA* 2021). Per-region GRU + SIR-consistency loss head, dropping the graph-attention component to preserve the AIIM defence. Reference implementation at <https://github.com/v1xerunt/STAN> (JHU data path; IQVIA claims data omitted per data-availability statement). Standalone (~250 lines) because no library wrap is appropriate.
6. **N-BEATS / NHITS** — Oreshkin et al. ICLR 2020 (or Challu et al. AAAI 2023); via `neuralforecast`. Restored 12 May: provides a second non-physics DL row alongside TFT, useful when reviewers ask about non-transformer SOTA. Library call (~50 lines).
7. **TFT (Temporal Fusion Transformer)** — Lim et al. IJF 2021; via `pytorch-forecasting`. Interpretable attention-based probabilistic forecaster. Library call + per-region wrapper (~150 lines).
8. **DeepAR** — Salinas et al. IJF 2020; via `pytorch-forecasting` or `gluonts`. Probabilistic RNN — gives a real probabilistic baseline for the WIS calibration table. Library call (~80 lines).

**Removed from scope (12 May).** Persistence, PatchTST, Prophet, Seq2Seq attention, XGBoost, and LSTM-per-region have been dropped from the E1 sweep. Rationale per dropped model:

- *Persistence* — one-line rule floor ($\hat y_{t+h} = y_t$). Seasonal Naive covers the rule-floor slot. May still appear as a single reference row in Table 2 footer.
- *PatchTST* — overlapping transformer architecture with TFT; diminishing returns.
- *Prophet* — slow, fiddly; ARIMA already covers the statistical-baseline slot.
- *Seq2Seq attention* — already marked for drop in the original plan.
- *XGBoost* — GRU-per-region is the modern non-physics control; a separate gradient-boosting row adds little above what reviewers expect from "deep-learning vs statistical-baseline" comparisons.
- *LSTM per region* — GRU per region is structurally near-identical (Chung et al. 2014 showed near-equivalent performance on time-series tasks); a separate LSTM row is low marginal value.

The retained 6-model E1 sweep is: ARIMA, GRU per region, PINN-GRU (proposed), reduced STAN, TFT, DeepAR.

**Graph-attention SOTA — explicitly excluded.** EpiGNN (Xie et al. ECML 2022), ColaGNN (Deng et al. CIKM 2020), HOIST (Gao et al. Nat. Comm. 2023), and MSAGAT-Net would all invite direct "why not graph-coupled" comparisons that re-open the AIIM overlap question. The right move is to **cite them in §2 Related Work** as concurrent / prior graph-attention forecasters and explicitly defer to future work in §8 Conclusion. Do not implement. Note: reduced STAN is included (Tier 3) because the SIR-consistency loss head can be evaluated without the graph-attention layer; this lets us benchmark the physics-informed-DL contribution while preserving the AIIM defence.

**`pyproject.toml` deps to add for Tier 3:** `neuralforecast>=1.7`, `pytorch-forecasting>=1.0`, `gluonts>=0.14` (optional, only if `pytorch-forecasting` DeepAR is unstable on this data). All three are pure-Python wheels and install cleanly on Windows + RTX 5060 Ti.

### Allocation (Experiment E2)

1. **No-surge baseline** — pure shortage scenario, free (~30 lines)
2. **Population-proportional** — required naive comparator (~50 lines)
3. **Demand-proportional** — required naive comparator (~50 lines)
4. **Greedy shortage-first** — cheap, useful (~80 lines)
5. **Deterministic MILP** — load-bearing
6. **Robust MILP (CVaR)** — load-bearing
7. **NSGA-II** — load-bearing, earns the "metaheuristic" in the title
8. **GA single-objective with LP-slave** — load-bearing for E3
9. **Simulated Annealing** — comparator metaheuristic for E3
10. **Historical Nightingale-style** — best-effort; if archival data lookup is hard, drop to footnote

---

## 4. 12-page Springer LNNS paper — section-by-section writing plan

Page budget from `01_RESEARCH_PROGRAMME.md` §10. Write **methodology before results** so we have something to revise while experiments run.

### Abstract (0.25 pp, ~200 words) — write last (D20)

Lead: gap (forecast accuracy without operational evaluation in NHS surge planning). State pipeline. Name two headline contributions: decision-aware composite loss; multi-objective robust allocation. One numeric result (e.g., "X% reduction in unmet demand vs demand-proportional at the same budget"). Close: open code/data link.

### 1. Introduction (1.0 pp) — D19

Paragraphs: (a) NHS surge crisis, COVID-19 lessons, Nightingale precedent; (b) gap — forecasting and allocation studied separately; predict-then-optimise underused in healthcare; (c) four contribution bullets (`01_RESEARCH_PROGRAMME.md` §4); (d) paper outline.
**Anchor citations:** Elmachtoub & Grigas (*Management Science* 2022) for predict-then-optimise; Liu & Cao (*Frontiers in Public Health* 2026) and Shams Eddin & El Hajj (*Healthcare* 2025) for combined forecast+optimise in healthcare.
**Pitfall:** do not oversell forecasting accuracy. Frame forecasting as *infrastructure*, decisions as the contribution.

### 2. Related Work (1.25 pp) — D19

Three subsections:

- **Spatiotemporal epidemic and hospital demand forecasting** (~0.45 pp): Raissi et al. 2019 (*JCP*) for PINNs; the 2025 book chapter for SEIRD-PINN baseline; multi-horizon RNN review; **explicitly cite MSAGAT-Net as concurrent work using a graph-attention forecasting backbone** with a clear distinction (per-region vs graph-coupled, decisions vs forecasts) — this is the AIIM defensive move.
- **Robust optimisation and metaheuristics for healthcare allocation** (~0.45 pp): Birge & Louveaux for stochastic optimisation; Mestre et al. for hospital network design; Resende & Werneck (2004) for facility location decomposition; Deb et al. (*IEEE TEC* 2002) for NSGA-II.
- **Forecast-to-decision pipelines** (~0.35 pp): Elmachtoub & Grigas 2022 mandatory; Donti et al. 2017 (NeurIPS) for differentiable end-to-end. The decision-aware loss is the local instantiation — say so.

**Pitfall:** prune ruthlessly. Cite, do not summarise.

### 3. Forecasting Module (2.5 pp) — D15

- **3.1 Refined SEI_aI_sHCRD** (~0.7 pp). Equations from `02_METHODOLOGY.md` §1.1. **Half a page on the ω/φ refinement** — explain why prior literature's overloading was biologically wrong; this is contribution 5.
- **3.2 Per-region PINN architecture** (~0.5 pp). State network (5 layers, 20 units) and parameter network (3 layers); cite the 2025 book chapter for hyperparameters.
- **3.3 GRU temporal head** (~0.4 pp). Augmented feature `[Ũ_r(t) ‖ X̃_r(t) ‖ x_{r,t}]`; multi-horizon decoder; one paragraph defending the per-region (not graph-coupled) choice — the AIIM differentiation.
- **3.4 Decision-aware composite loss** (~0.6 pp). Derive Huber + PINN residual + asymmetric underestimation hinge + smoothness; explain operationally why underforecasting hurts twice.
- **3.5 MC Dropout uncertainty** (~0.3 pp). Bracher et al. 2021 for WIS.

**Figure 1:** pipeline schematic. **Figure 2:** SEIRD compartmental flow showing ω/φ separation.
**Pitfall:** do not relitigate the 2025 book chapter's PINN; one-paragraph cite-and-reuse.

### 4. Optimisation Module (3.0 pp — load-bearing centre of gravity) — D16

- **4.1 Sets, parameters, decision variables** (~0.4 pp). Compact table from §0.
- **4.2 Deterministic MILP** (~0.6 pp). Objective + constraints from §3.1.
- **4.3 Equity linearisation** (~0.4 pp). Show the max-shortage-ratio derivation. Genuine math contribution — give it room.
- **4.4 Robust extension** (~0.4 pp). CVaR-flavoured objective with λ_3 W.
- **4.5 GA + LP-slave decomposition** (~0.5 pp). Cite Resende & Werneck 2004; pseudocode for the GA master loop.
- **4.6 NSGA-II encoding** (~0.4 pp). Cite Deb et al. 2002. **Pre-empt MOEA/D-style reviewer questions** — one paragraph defending NSGA-II (well-cited baseline, mature pymoo support, computational tractability).
- **4.7 Simulated Annealing comparator** (~0.3 pp). Brief.

**Table 1:** GA/NSGA-II/SA hyperparameters.
**Pitfall:** UKCI reviewers are CI specialists. Justify every metaheuristic choice. Tabulate `pop=100, gens=200, p_c=0.9, p_m=1/|J|`.

### 5. NHS England Case Study (1.0 pp) — D17

Data sources, regions (n=7), chronological-by-wave split (Alpha train / Delta val / Omicron test), explicit cap at 31 Aug 2022 (NHS moved to weekly reporting). One regional map figure; one MV-bed-occupancy time-series across 7 regions on the test period.

**Figure 3:** NHS England regions map with adjacency overlay.
**Figure 4:** MV-bed time series, 7 regions, test period.

### 6. Results (2.25 pp) — D18

- **6.1 Forecasting accuracy** (~0.7 pp). Table 2 — 5 main models × 4 horizons × {MAE, RMSE, MASE} (push the rest to supplementary). Figure 5 — per-horizon error bars. Mention WIS for proposed.
- **6.2 Regional allocation** (~0.7 pp). Table 3 — 6 policies × {Unmet, Coverage, Travel, Max Shortage, Theil}. Figure 6 — allocation heatmap. Figure 7 — sensitivity to budget.
- **6.3 Trust-level metaheuristic comparison + Pareto** (~0.85 pp). Table 4 — runtime/quality. Figure 8 — Pareto front projected to 2D pairs.

**Pitfall:** do not dump 9 baselines × 4 horizons × 4 metrics in main text. Supplementary.

### 7. Discussion and Limitations (0.5 pp) — D20

Three paragraphs: equity-vs-efficiency Pareto trade-off; computational cost across MILP/NSGA-II; **explicit limitations** — single country, COVID-specific waves, MC Dropout is approximate UQ, no rolling-horizon revision. The limitations paragraph protects against "but what about X?" reviewer questions.

### 8. Conclusion (0.25 pp) — D20

Three sentences summary. Future-work bullet: graph-coupled forecaster (name MSAGAT-Net obliquely as "ongoing work"), online rolling-horizon, multi-pathogen.

### References (1.0 pp, ~25–30 entries)

Mandatory anchors: Elmachtoub & Grigas 2022; Deb et al. 2002; Bracher et al. 2021; Raissi et al. 2019; Blank & Deb 2020 (pymoo); Liu & Cao 2026; Shams Eddin & El Hajj 2025; the 2025 book chapter; Resende & Werneck 2004; Lauer et al. 2020 (incubation period); Byrne et al. 2020 (clinical periods); Boddington et al. 2021 (symptomatic ratio); Docherty et al. 2020 (UK clinical features). Fill remainder with healthcare OR, NHS-specific epidemiology, metaheuristic prior art.

---

## 5. Gate-driven contingency plan

Hard pre-committed scope reductions at each gate. No silent slippage.

- **Gate G2 (16 May, D7) — forecasting credible?**
  - PASS: full E1 sweep with 9 baselines × 5 seeds.
  - FAIL: drop PINN, keep "DA-GRU" (decision-aware GRU without ODE residual). The composite loss remains the contribution. ω/φ refinement stays as parameter-calibration methodology.
- **Gate G3 (19 May, D10) — MILP solving?**
  - PASS: continue with robust MILP and metaheuristics.
  - FAIL: keep deterministic MILP only; drop CVaR variant; document robust extension as future work.
- **Gate G4 (24 May, D15) — sufficient results?**
  - PASS: full programme.
  - PARTIAL FAIL: drop in this order — E6 sensitivity, E5 robustness, E4 full Pareto front, E3 Midlands (keep London).
  - HARD FAIL: pivot to 6–8 page short paper "Decision-Aware Physics-Informed Forecasting for NHS Critical-Care Surge Demand" with optimisation as future work. Still novel relative to MSAGAT-Net.
- **Household disruption escape hatch.** If a full week is lost (likelihood: high per `01_RESEARCH_PROGRAMME.md` §9), declare scope reduction in writing in `docs/status/` the same day. Do not work in fragments while exhausted.

---

## 6. Verification

End-of-day sanity checks (cumulative):

- **D2:** `python -c "from src..data.loaders import load_regional_csv; df = load_regional_csv(); assert df.shape[0] >= 700 and df['region'].nunique() == 7"` passes. EDA notebook produced. `data/processed/regional_daily.csv` exists.
- **D5:** `python scripts/run_e1.py --models arima,gru --seeds 42 --horizons 14` produces a metrics CSV. Both Tier 1 baselines run end-to-end on the validation set. Forecast metric harness reproduces a hand-computed sanity value for ARIMA MAE within 1%.
- **D8:** PINN-GRU end-to-end run on training set with composite loss converges (val Huber loss decreases monotonically over 50 epochs). MC Dropout produces non-degenerate quantiles (q^0.9 > q^0.5 > q^0.1 strictly for all (region, horizon) on test). WIS computed and within plausible range vs ARIMA and GRU baselines.
- **D10:** `python scripts/run_e2.py --policy det_milp --budget medium --travel 120` produces an allocation plan with non-zero `b_jh`, all constraints satisfied (verified via Pyomo `solver_status == optimal`), unmet demand strictly less than no-surge baseline.
- **D13:** NSGA-II Pareto front for regional problem produces ≥10 non-dominated solutions; hypervolume increasing across generations.
- **D15:** Internal sanity sweep: rerun E1 + E2 with seed 42 only; verify all 6 policies produce non-degenerate metrics. Status note in `docs/status/2026-05-24.md` with go/no-go on full programme.
- **D17:** All experiments E1–E3 complete. Tables 1–4 generated as CSV. Figures 3–8 generated as PNG/PDF.
- **D20:** **AIIM overlap audit pass.** Read MSAGAT-Net submission and this paper side-by-side. Confirm: zero shared figures, zero shared tables, zero copy-paste text. MSAGAT-Net is **not cited** (still under review at AIIM; cite once published). The differentiation is structural — per-region (no graph), PINN-SEIRD physics, decision-aware composite loss, predict-then-optimise pipeline — not citation-based.
- **D21 (31 May):** Camera-ready PDF compiled clean. CMT submission confirmation email saved. Repo public-ready (Apache 2.0 or MIT licence).

---

## 7. Critical files reference

**To keep (no deletion):** all 16 existing files in `data/`. Reorganised (not deleted) into `data/external/` (international datasets) and `data/legacy/` (any older NHS snapshots superseded by the fresh download). `data/nhs_timeseries.txt` is the primary MV-bed forecasting target; `data/nhs-adj.txt` is the 7×7 region adjacency; `data/ltla_timeseries.txt` is a candidate for the trust-level / fine-grained generalisation experiment.

**To reuse unchanged:** `src/forecasting/pinn_seird.py`, `scripts/download_nhs_data.py` (URL list refreshed 2026-05-10), `docs/01_RESEARCH_PROGRAMME.md`, `docs/02_METHODOLOGY.md`, `docs/03_TIMELINE.md`, `pyproject.toml`, `README.md`, `.gitignore`.

**To implement (load-bearing, must work or paper fails):** `src/forecasting/composite_loss.py`, `src/optimization/milp.py`, `src/optimization/metaheuristics.py`, `scripts/build_regional_dataset.py`.

**To implement (supporting, can scope-reduce at gates):** the rest of `src/{data,evaluation,forecasting,optimization,utils}/`, `configs/baselines/`, `notebooks/`, `tests/`, `scripts/run_e*.py`.

---

## D1 status (10 May 2026, completed)

For ground-truth tracking, the following were completed on D1:

- ✅ NHS download script URLs refreshed; the 4 daily archives covering 1 Aug 2020 – 31 Aug 2022 fetched into `data/raw/` (~446 KB total) with SHA-256 manifest.
- ✅ `data/` reorganised: `external/` holds the 12 MSAGAT-era international datasets; root keeps the 4 NHS-related files; `legacy/`, `processed/`, `graphs/`, `raw/` each have `.gitkeep`.
- ✅ `data/INVENTORY.md` documents what is where, why, and how to reproduce.
- ✅ `data/graphs/nhs_region_adj.txt` (canonical copy) and `data/graphs/README.md` (encoding caveat).
- ✅ `src/utils/seed.py` for reproducible seeding.
- ✅ `pandas` pinned to 2.3.3 + `openpyxl` 3.1.5 in the conda env (replacing the broken pandas 3.0.2 preview).

---

## D2 status (12 May 2026, completed)

For ground-truth tracking, the following were completed on D2:

**Literature review pass (8 papers).** Read and analysed Luo & Stellato (2024, arXiv), Bertsimas et al. (2022, *NRL*), Kargar et al. (2024, *TR-E*), Tang et al. (2025, *TR-E*), Brown et al. (2020, *NEJM*), Valente et al. (2022, *Stat. J. IAOS*), Gao et al. (2021, *JAMIA* — STAN), and Gao et al. (2023, *Nat. Comm.* — HOIST). Synthesis in `C:\Users\olarinoyem\.claude\plans\as-a-researcher-i-rustling-yeti.md`. Identified Bertsimas (2022) as the closest predict-then-optimise precedent and Valente (2022) as the closest ICU-forecasting target precedent.

**§2 Related Work rewrite.** Replaced the entire `docs/paper/sections/02_related_work.tex`, anchored on Bertsimas (2022) as the closest end-to-end precedent and Luo & Stellato (2024) as the closest neural-ODE-based predict-then-optimise architecture. Added subsections on (i) spatiotemporal forecasting (with Valente, STAN, HOIST), (ii) robust optimisation (with Eriskin, Wan, Shams Eddin, Kargar, Tang), (iii) forecast-to-decision pipelines (with Bertsimas, Luo & Stellato, Liu & Cao). LaTeX build verified clean, no undefined citations.

**Bibliography fixes and additions.** Corrected `liu2024hoist` → `gao2023hoist` (real authors: Gao, Heintz, Mack, Glass, Cross, Sun; correct journal *Nat. Comm.*; DOI 10.1038/s41467-023-38756-3). Added 8 new bib entries: `bertsimas2022where`, `luo2024equitable`, `valente2022forecasting`, `kargar2024data`, `tang2025mobile`, `kerr2021covasim`, `carteni2020mobility`, `brown2020allocating`.

**External data downloads.** Authored `scripts/download_external_data.py` and fetched 6 datasets into `data/raw/external/` (~57 MB total): Google Community Mobility Reports for GB 2020–2022 (3 CSVs); ONS Mid-Year Population Estimates 2021; English IMD 2019 File 7 (LSOA scores) and File 10 (LA summaries). SHA-256 manifest at `data/raw/external/MANIFEST.txt`. `data/INVENTORY.md` updated.

**Title change.** Working title moved from *"From Forecasts to Capacity Decisions: A Physics-Informed and Metaheuristic Pipeline for NHS Critical-Care Surge Planning"* to **"Decision-Aware Physics-Informed Forecasting and Metaheuristic Allocation for NHS Critical-Care Surge Capacity Under Demand Uncertainty"** — front-loads the headline novelty (*Decision-Aware*), names both methodological halves with equal billing, signals robust framing. Reflected in `docs/paper/main.tex`, `docs/paper/title.txt`, `README.md` (working title + bib citation block), and `docs/01_RESEARCH_PROGRAMME.md`.

**Baseline trim.** E1 forecasting sweep reduced to **7 models** (was 13 in original plan): Seasonal Naive (rule floor), ARIMA (statistical floor), GRU per region (DL control), PINN-GRU (proposed), N-BEATS, TFT, DeepAR. Dropped: Persistence (Seasonal Naive covers the rule floor), reduced STAN (calling a graph-stripped model "STAN" misrepresents the published method; GRU per region vs PINN-GRU now provides the physics ablation directly), PatchTST, Prophet, Seq2Seq attention, XGBoost, LSTM per region. Two clean ablations remain: (i) GRU per region vs PINN-GRU isolates physics contribution; (ii) PINN-GRU with MSE vs PINN-GRU with composite loss isolates decision-aware-loss contribution.

**Optimisation method expansion.** Added augmented ε-constraint as a Phase C method (`epsilon_constraint.py`) — Kargar et al. (2024) precedent; provides exact non-dominated Pareto points complementary to NSGA-II's heuristic approximation. Allocation comparator set is now: deterministic MILP, robust MILP (CVaR), augmented ε-constraint, NSGA-II, GA (LP-slave), SA (Appendix-only). IMD-weighted intra-region and population-share inter-region fairness constraints added to `milp.py` per Bertsimas (2022) eq. (25) and Luo & Stellato (2024) SVI patterns.

**New counterfactual experiment E7.** `scripts/run_e7.py` adapts the HOIST (Gao et al. 2023) number-needed-to-treat / cost-ratio pattern to surge planning: perturb total budget ∈ {0.8, 0.9, 1.0, 1.1, 1.2}, compute marginal beds-per-£ by region, IMD-weighted ranking. Drives §6.4 policy-impact paragraph. Estimated 1–2 days.

**AIIM differentiation clarified (12 May author decision).** Three confirmations: (a) PINN-GRU is **per-region** (no graph coupling) by design; (b) STAN and HOIST cited in §2 as physics-informed-graph precedents but **not** implemented as baselines (full STAN would re-open graph overlap with MSAGAT-Net; reduced STAN misrepresents the method); (c) MSAGAT-Net is **not cited** in §2 because it is still under review at AIIM (cite after publication); the AIIM-distinctness is structural, not citation-based. Google Mobility data is per-region time series (not pairwise flow matrices) — using it as covariates does not introduce graph structure.

**Implementation plan deltas reflected in:** Context revisions list, §1.5 (new external features section), Phase 0 (new `build_regional_features.py`), Phase 1 Tier 1+3 baseline lists, Phase 3 (`epsilon_constraint.py`, IMD-fairness MILP constraints, IMD-proportional heuristic), Phase 4 (added E7 row), §6 D5+D20 verification, this D2 status entry.
