# UKCI 2026 — Research Programme

**Lead author:** Michael Ajao-Olarinoye, Centre for Computational Sciences and Mathematical Modelling, Coventry University
**Co-author (optimisation):** Abiola Babatunde, CSM, Coventry University
**Supervisor / co-author:** Vasile Palade, CSM, Coventry University
**Conference:** UKCI 2026, Coventry, 9–11 September 2026 (Springer LNNS proceedings, 12-page full paper)
**Submission deadline:** 31 May 2026 (Microsoft CMT)

---

## 1. Why one combined paper, not two

**Strategic reality check.** MSAGAT-Net is currently under review at *Artificial Intelligence in Medicine* and is benchmarked on **LTLA-COVID** and **NHS-ICUBeds** datasets. A separate UKCI paper on NHS critical-care forecasting alone — even with a different graph attention layer or a decision-aware loss bolted on — would constitute self-overlap. Reviewers familiar with MSAGAT-Net would see the resemblance, and the AIIM editor could legitimately flag it.

**Better framing.** Position the UKCI paper as a **forecast-driven optimisation** contribution where forecasting is supporting infrastructure and the methodological core is the metaheuristic resource-allocation layer. This:

1. Sidesteps the AIIM overlap entirely — the paper's central claim is about *decisions*, not *forecasts*.
2. Aligns with current literature precedent (Liu & Cao 2026 in *Frontiers in Public Health*; Shams Eddin & El Hajj 2025 in *Healthcare*; both combine forecasting and optimisation in single papers).
3. Makes Abiola's metaheuristic contribution central, justifying his co-authorship and matching his published expertise on heuristic algorithms for set covering.
4. Targets UKCI's computational-intelligence audience cleanly — physics-informed neural networks **and** evolutionary metaheuristics in one pipeline is a strong CI story.

---

## 2. Title candidates

The title must signal three things to UKCI reviewers: physics-informed forecasting, metaheuristic optimisation, and operational health-system relevance.

**Approved title (12 May 2026, after literature-review pass):**

> **"Decision-Aware Physics-Informed Forecasting and Metaheuristic Allocation for NHS Critical-Care Surge Capacity Under Demand Uncertainty"**

Sixteen words; single clause; front-loaded keyword *Decision-Aware* names the headline loss-function novelty; *Physics-Informed Forecasting and Metaheuristic Allocation* gives both halves equal billing; *NHS Critical-Care Surge Capacity* anchors the domain; *Under Demand Uncertainty* signals the robust / scenario framing without committing to a specific terminology in the title.

This title was chosen over the four earlier candidates (now superseded) below, after the 12 May literature review identified the *decision-aware composite loss* as the single methodologically novel contribution that must be front-loaded in the title:

1. *"Decision-Aware Computational Intelligence for NHS Critical-Care Surge Planning: Physics-Informed Forecasting with Metaheuristic Allocation Under Demand Uncertainty"* — long but explicit.
2. *"From Forecasts to Capacity Decisions: A Physics-Informed and Metaheuristic Pipeline for NHS Critical-Care Surge Planning"* — clearer narrative but does not front-load the decision-aware keyword.
3. *"Predict-then-Optimise for Pandemic Surge Capacity: Physics-Informed Forecasting and Multi-Objective Metaheuristic Allocation for the NHS"* — invokes the operations-research framework but Bertsimas et al. (2022) already occupy the predict-then-optimise framing for COVID-19 resource allocation.
4. *"A Computational Intelligence Pipeline for Robust NHS Critical-Care Surge Planning Under Demand Uncertainty"* — shortest and too vague.

---

## 3. Research questions

- **RQ1.** Can a per-region physics-informed neural epidemic forecaster, coupled with a decision-aware training objective, produce demand scenarios that yield better critical-care surge allocation decisions than scenarios from standard time-series baselines?
- **RQ2.** How do exact MILP, scenario-based robust MILP, and metaheuristic methods (GA, NSGA-II, Simulated Annealing) compare on solution quality, robustness to demand uncertainty, and computational cost for NHS regional and trust-level surge capacity allocation?
- **RQ3.** What is the trade-off between equity (regional shortage parity), efficiency (total cost), and operational feasibility (travel-time, capacity limits) on the Pareto frontier of NHS surge allocation, and how do these trade-offs shift under different forecast scenarios?

---

## 4. Contributions (paper claim)

1. A four-stage forecast-to-decision pipeline that links physics-informed neural epidemic forecasting to metaheuristic surge capacity allocation, end-to-end and reproducible on open NHS England data.
2. A **decision-aware training objective** that combines Huber data fit, PINN-SEIRD ODE residual regularisation, an asymmetric underestimation penalty, and temporal smoothness — designed so that forecast errors are penalised more heavily where they would lead to costly allocation shortages.
3. A **multi-objective robust optimisation** formulation for NHS critical-care surge capacity (cost vs unmet demand vs travel burden vs equity), solved exactly via MILP at regional scale and via NSGA-II at trust-level scale.
4. **Empirical comparison of metaheuristics** (Genetic Algorithm, Non-dominated Sorting Genetic Algorithm II, Simulated Annealing) against exact MILP and population/demand-proportional baselines, on a real NHS England case study.
5. A **refined $SEI_aI_sHCRD$ compartmental formulation** that separates the hospitalisation ratio $\omega$ ($I_s \to H$) from the critical-care escalation proportion $\phi$ ($H \to C$), correcting an over-loading of $\omega$ in the prior compartmental literature.
6. A reproducibility package: cleaned NHS regional dataset, model implementations, optimisation formulation, and full experiment pipeline.

The italicised items are the load-bearing claims. Items 1, 5, and 6 are scaffolding for the contribution.

---

## 5. High-level methodology (full detail in `02_METHODOLOGY.md`)

The paper executes four phases:

```
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE A: PER-REGION PHYSICS-INFORMED FORECASTING                    │
│  Inputs: regional time series                                        │
│  PINN-SEIRD per region → state estimates + time-varying parameters   │
│  Temporal encoder (GRU) → multi-horizon point forecasts              │
│  MC Dropout → empirical predictive quantiles                         │
│  Output: ŷ_{r,t+h}, q_{r,h}^{0.1}, q_{r,h}^{0.5}, q_{r,h}^{0.9}    │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE B: DEMAND SCENARIO GENERATION                                 │
│  Quantile-based discrete scenarios { low, median, high, tail }       │
│  Scenario weights π_s                                                │
│  Output: { d_{i,h}^s, π_s }                                          │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE C: ROBUST METAHEURISTIC ALLOCATION                            │
│  Decision variables: x_j (open), b_jh (capacity), z_ijhs (transfer)  │
│  Multi-objective: cost, unmet demand, travel, equity                 │
│  Methods: MILP (exact, regional), Robust MILP, GA, NSGA-II, SA       │
│  Output: allocation plan and Pareto front                            │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE D: EVALUATION                                                 │
│  Forecasting: MAE/RMSE/sMAPE/MASE per horizon, WIS calibration       │
│  Decisions: unmet demand, coverage, travel burden, equity index      │
│  Robustness: worst-case shortage, value of robust solution (VRS)     │
│  Computation: runtime, MILP gap, NSGA-II hypervolume                 │
└──────────────────────────────────────────────────────────────────────┘
```

The unifying thread is the **decision-aware training objective**: the forecaster is not optimised purely for statistical accuracy but for downstream decision quality. This is a single-paper version of the "Smart Predict-then-Optimize" framework from Elmachtoub & Grigas (*Management Science* 2022), adapted to healthcare resource allocation.

---

## 6. Data

| Dataset | Source | Use |
|---|---|---|
| NHS England COVID-19 Hospital Activity (Daily Admissions and Beds, three XLSX archives) | https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-hospital-activity/ | Primary forecasting targets and covariates: admissions, occupied COVID beds, MV beds, daily, regional, 1 Aug 2020 – 31 Aug 2022 |
| NHS UEC Critical Care and General & Acute Beds | NHS England Urgent and Emergency Care SitRep | Adult critical-care capacity at trust level; baseline `C_{j,h}` for optimisation |
| ONS NHS England Region geographies | ONS Open Geography Portal | Adjacency graph and region centroids |
| ONS mid-year population estimates | ONS | Equity weighting and per-capita normalisation |
| ONS English Indices of Deprivation (IMD) | DHSC | Optional equity weighting term |
| Approximate travel times (centroid-to-centroid) | OSRM public API or great-circle × 1.3 detour factor | Travel-time matrix `T_{ij}` for allocation feasibility |

**Spatial resolution.** NHS England regions (n = 7) for the main forecasting and regional MILP. Trust-level (n ≈ 150 acute trusts) for the metaheuristic comparison experiment, where MILP becomes expensive and NSGA-II earns its place.

**Temporal resolution.** Daily, 1 August 2020 to 31 August 2022 for the main analysis period. After 31 August 2022, NHS England moved from daily to weekly publication, so we cap the daily-resolution analysis there.

**Train/validation/test split (chronological by epidemic wave):**

- Train: 1 Aug 2020 – 31 May 2021 (Alpha + early vaccination)
- Validation: 1 Jun 2021 – 30 Nov 2021 (Delta)
- Test: 1 Dec 2021 – 31 Aug 2022 (Omicron and beyond)

This deliberately holds the most operationally distinct wave (Omicron, with shifted case-to-hospitalisation ratios) entirely in the test set.

---

## 7. Baselines

### 7.1 Forecasting baselines

| Category | Baseline | Implementation |
|---|---|---|
| Trivial | Persistence; seasonal naive (7-day lag) | NumPy |
| Statistical | ARIMA per region; Prophet per region | `statsmodels`, `prophet` |
| ML | XGBoost with engineered lag features | `xgboost` |
| Deep | LSTM, GRU, Seq2Seq attention (per region, no graph) | PyTorch |
| Physics-informed | PINN-SEIRD + GRU per region (this paper's forecaster) | PyTorch |

We deliberately exclude graph attention forecasting backbones (e.g., STAN, EpiGNN, ST-GAT) from the headline comparison to keep the paper focused on decision quality rather than forecasting accuracy benchmarking. The pipeline is forecasting-backend agnostic: any model producing point and quantile demand estimates per region per horizon can be substituted into Phase A. We include a sensitivity experiment (E5) that swaps the forecaster for ARIMA to demonstrate this. Detailed comparison against graph-based forecasting backbones is left as a clearly identified line of future work.

### 7.2 Allocation baselines

| Policy | Description |
|---|---|
| Current capacity only | No surge expansion; pure shortage scenario |
| Population-proportional | Allocate proportional to regional population |
| Demand-proportional | Allocate proportional to median forecast demand |
| Greedy shortage-first | Iteratively assign capacity to the region with largest predicted shortage |
| Historical Nightingale-style | Reproduce actual UK Nightingale hospital placements as a real-world baseline |
| Deterministic MILP | Optimal under median forecast only |
| Robust MILP | Optimal across scenario set |
| GA (single-objective scalarised) | Genetic Algorithm with binary facility encoding |
| NSGA-II (multi-objective) | Pareto front |
| Simulated Annealing | Comparator metaheuristic |

---

## 8. Experimental design (full matrix in `03_TIMELINE.md` §3)

Six experiments, each producing a table or figure for the paper:

- **E1 — Forecasting accuracy.** All forecasting baselines vs the proposed PINN-GRU with decision-aware loss; horizons 7/14/21/28; metrics MAE/RMSE/sMAPE/MASE/WIS; ablations on PINN loss, underestimation penalty, smoothness.
- **E2 — Regional MILP allocation.** All allocation baselines on 7-region problem; deterministic vs robust; metrics unmet demand, coverage, travel burden, equity index.
- **E3 — Trust-level metaheuristic comparison.** GA vs NSGA-II vs SA vs MILP on a tractable trust-level subset (e.g., London region, ~30 trusts); solution quality, runtime, hypervolume.
- **E4 — Pareto front analysis.** NSGA-II at full trust-level scale; visualise cost vs unmet demand vs travel burden trade-offs.
- **E5 — Robustness to forecast quality.** Re-run optimisation with (a) PINN-GRU forecasts, (b) ARIMA forecasts, (c) ground truth (oracle), (d) historical median; quantify value of better forecasts for allocation quality.
- **E6 — Sensitivity analysis.** Vary budget, travel-time threshold, equity weight, and scenario set size; report robustness of allocation plan.

---

## 9. Risks and explicit mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Vasile blocks UKCI submission due to AIIM overlap concern | Medium | Critical | Email today (6 May), get explicit written sign-off before any further work |
| Abiola declines or has no bandwidth | Medium | Critical | Confirm by 7 May; if no, pivot to a forecasting-only short paper or cancel UKCI submission |
| HOPE-MOVE workload conflict | High | High | Hard time-box: 3 hours/evening, 6 hours/weekend day; HOPE-MOVE deliverables to Matt take precedence |
| NHS data harmonisation across the three XLSX archives is messy | High | Medium | Day 1 priority; automated tests for date contiguity and region naming |
| MILP scaling at trust level | Medium | Medium | Pre-plan NSGA-II for trust-level; MILP only for regional and small subsets |
| Travel-time data acquisition takes longer than expected | Medium | Low | Use great-circle distance × 1.3 as fallback; OSRM stretch goal |
| PINN training instability | Medium | Medium | Reuse 2025 book chapter hyperparameters and Xavier initialisation; gradient clipping |
| Wife's late pregnancy / household disruptions | High | Medium | Build buffer into timeline; declare scope reduction by 24 May if behind |

---

## 10. 12-page paper structure (Springer LNNS template)

| Section | Pages | Content |
|---|---|---|
| Abstract | 0.25 | 200 words |
| 1. Introduction | 1.0 | Surge capacity problem, gap, contributions (4 bullets) |
| 2. Related Work | 1.25 | (a) Spatiotemporal epidemic and hospital demand forecasting, (b) Robust optimisation for healthcare resource allocation, (c) Forecast-to-decision pipelines |
| 3. Forecasting Module | 2.5 | Per-region PINN-SEIRD (cite book chapter), temporal encoder, decision-aware loss, scenario generation |
| 4. Optimisation Module | 3.0 | MILP formulation, robust scenario-based extension, GA and NSGA-II encoding, **load-bearing methodological core** |
| 5. NHS England Case Study | 1.0 | Data, regions, capacity, scenarios |
| 6. Results | 2.25 | Forecasting accuracy (compact), allocation comparison (full), Pareto fronts, sensitivity, robustness |
| 7. Discussion and Limitations | 0.5 | Equity vs efficiency, robustness, generalisation, computational cost |
| 8. Conclusion | 0.25 | Summary + future work |
| References | 1.0 | ~25–30 entries |

The proportions deliberately tilt toward the optimisation contribution to make the paper's centre of gravity clear and to protect against AIIM overlap concerns.

---

## 11. Authorship and acknowledgements

| Role | Person |
|---|---|
| Lead author, forecasting, integration | Michael Ajao-Olarinoye |
| Co-author (joint), optimisation core, metaheuristic implementation | Abiola Babatunde |
| Co-author, supervision, methodology validation | Vasile Palade |

We deliberately keep authorship to three. Adding more co-authors (e.g., Fei He, Petra Wark, Seyed Mousavi, Zindoga Mukandavire from the AIIM submission) risks blurring the paper's contribution and creating overlap-perception issues with that submission. If Matthew England wants to be involved, he should be invited explicitly with a defined contribution.

Acknowledgements should include funding source (Coventry University CSM, EPSRC if applicable), HOPE-MOVE programme contextualisation if relevant, and any HPC compute used (Brosnan/Zeus).

---

## 12. Reproducibility package

The submission must include:

- **Code.** Public GitHub repository (private until acceptance, then made public). Apache 2.0 or MIT licence.
- **Data.** Cleaned regional CSV with provenance script; raw NHS XLSX archives are public so we link to them rather than redistributing.
- **Configurations.** YAML files for every experiment, with random seeds.
- **Environment.** `pyproject.toml` and `requirements.txt`, Python version pinned.
- **Trained models.** Saved weights for the proposed model and all baselines.
- **Results.** All metrics tables in machine-readable form (CSV/JSON), all figures with source data.
- **Documentation.** README with quickstart, full methodology document, paper preprint link.

A line in the paper saying "Code and data are available at [URL]" is essential for UKCI's reproducibility expectations and for the paper's long-term value.

---

## 13. Beyond UKCI

If the work is well-received at UKCI, natural follow-ups include:

- A **journal extension** (e.g., *European Journal of Operational Research*, *Health Care Management Science*, *IISE Transactions on Healthcare Systems Engineering*) with full trust-level analysis, additional metaheuristics (Ant Colony, Tabu Search, Particle Swarm), and a proper online/rolling-horizon formulation.
- A **collaboration with NHS planners** if the framework is operationally useful — Coventry's CSM has links to NHS digital teams.
- A **methods paper** on the decision-aware training loss specifically, with broader application beyond healthcare (energy demand, supply chain, etc.).

---

*End of programme document. See `02_METHODOLOGY.md` for the full mathematical formulation and `03_TIMELINE.md` for the day-by-day work plan.*
