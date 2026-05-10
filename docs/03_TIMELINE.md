# 03 — Timeline and Experiments

This document specifies day-by-day milestones and the full experiment matrix for the UKCI 2026 paper.

---

## 1. Strategic gates

Three explicit go/no-go gates protect against runaway commitment to a struggling plan:

| Gate | Date | Decision |
|---|---|---|
| G1: AIIM and Abiola sign-off | 7 May | Stop if Vasile blocks UKCI submission. Stop if Abiola declines. |
| G2: Forecasting module functional | 16 May | Stop or scope-reduce if PINN-GRU end-to-end pipeline does not produce credible test forecasts. |
| G3: MILP scaffold functional with synthetic demand | 19 May | Stop or scope-reduce if optimisation pipeline does not produce credible allocation plans by this date. |
| G4: Sufficient results to write paper | 24 May | If E1–E3 results not available, drop E4–E6 from scope. |

Each gate produces a written status note (one paragraph) in the repo `docs/status/` directory, dated. This forces honest accounting.

---

## 2. Day-by-day work plan (6 May — 31 May 2026)

### Week 1: Foundations and gate G1 (6–10 May)

**Wednesday 6 May (today)** — *3 hours, evening*
- [x] Project repository structure created and initialised in git
- [x] Documentation framework (this document, methodology, programme) drafted
- [ ] Send Vasile email asking for blessing on UKCI submission given AIIM overlap considerations
- [ ] Send Abiola email inviting co-authorship; specify Paper 2 / now combined paper role
- [ ] Run NHS data download script locally; verify all three XLSX archives downloaded

**Thursday 7 May** — *3 hours, evening*
- [ ] **Gate G1 check.** Have responses from Vasile and Abiola been received?
- [ ] Build regional daily tidy CSV. Test for (a) date contiguity, (b) region naming consistency, (c) handling of definition changes in 2021–2022, (d) missingness pattern
- [ ] EDA notebook: time-series plots per region for admissions, occupied beds, MV beds; correlation matrix; missingness audit
- [ ] If G1 fails: pivot decision (forecasting-only short paper, or cancel)

**Friday 8 May** — *3 hours, evening*
- [ ] Build NHS region adjacency graph (manual, 7 nodes). Document.
- [ ] Implement persistence and seasonal-naive baselines + standardised metric harness
- [ ] First baseline metrics on validation set

**Saturday 9 May** — *6 hours, daytime*
- [ ] ARIMA + Prophet baselines per region
- [ ] XGBoost with engineered lag features
- [ ] Baseline comparison table v1
- [ ] Begin LSTM/GRU baselines (port from Ajao-Olarinoye et al. 2024 codebase)

**Sunday 10 May** — *6 hours, daytime*
- [ ] LSTM, GRU, Seq2Seq attention baselines on regional data
- [ ] Baseline comparison table v2 (with all classical baselines)
- [ ] Buffer / catch-up time
- [ ] Status note in `docs/status/2025-05-10.md`

### Week 2: Forecasting module and gate G2 (11–17 May)

**Monday 11 May** — *3 hours*
- [ ] Port PINN-SEIRD module from book chapter codebase
- [ ] Per-region instantiation with shared architecture, separate weights
- [ ] Unit tests for PINN forward pass and ODE residual

**Tuesday 12 May** — *3 hours*
- [ ] Train PINN per region on training set; verify state estimates and parameter trajectories
- [ ] Smoke test: PINN-derived signals + simple GRU forecaster

**Wednesday 13 May** — *3 hours*
- [ ] Multi-horizon GRU temporal head
- [ ] Composite loss implementation (Huber + PINN + underestimation + smoothness)
- [ ] First end-to-end training run on small subset

**Thursday 14 May** — *3 hours*
- [ ] Full training on regional data with composite loss
- [ ] First proposed-model results on validation set

**Friday 15 May** — *3 hours*
- [ ] *(Abiola begins MILP scaffolding)* — Pyomo deterministic MILP with synthetic demand
- [ ] MC Dropout uncertainty quantification at inference
- [ ] Quantile extraction (Low/Median/High scenarios)

**Saturday 16 May** — *6 hours*
- [ ] **Gate G2 check.** Is the forecasting module producing credible test forecasts?
- [ ] If yes: ablations A1 (no PINN loss), A2 (no underestimation loss), A3 (no smoothness)
- [ ] If no: scope reduction — remove PINN component, keep simple GRU + decision-aware loss only

**Sunday 17 May** — *6 hours*
- [ ] Multi-seed runs (5 seeds) for proposed model and key baselines
- [ ] First version of forecasting results table (paper table 1)
- [ ] Status note in `docs/status/2025-05-17.md`

### Week 3: Optimisation, integration, and gate G3 (18–24 May)

**Monday 18 May** — *3 hours*
- [ ] Validate Abiola's MILP scaffold on regional problem with synthetic demand
- [ ] Implement allocation baselines (population-proportional, demand-proportional, greedy)

**Tuesday 19 May** — *3 hours*
- [ ] **Gate G3 check.** Is the MILP solving regional allocation problems with synthetic demand?
- [ ] Connect Phase A forecasts to Phase B scenario generation to Phase C MILP
- [ ] First end-to-end pipeline run with real forecasts

**Wednesday 20 May** — *3 hours*
- [ ] Robust MILP variant (CVaR-flavoured)
- [ ] Begin paper draft: Introduction and Related Work

**Thursday 21 May** — *3 hours*
- [ ] *(Abiola)* GA implementation in pymoo
- [ ] *(Abiola)* NSGA-II implementation in pymoo
- [ ] Paper draft: Data and Forecasting Module sections

**Friday 22 May** — *3 hours*
- [ ] *(Abiola)* Simulated Annealing comparator
- [ ] *(Abiola)* GA, NSGA-II, SA on regional problem; compare with MILP
- [ ] Paper draft: Optimisation Module section

**Saturday 23 May** — *6 hours*
- [ ] Sensitivity analysis: budget, travel-time threshold, equity weight
- [ ] Trust-level scaling experiment (subset of trusts)
- [ ] Paper draft: Case Study and Results sections

**Sunday 24 May** — *6 hours*
- [ ] **Gate G4 check.** Sufficient results to write the paper?
- [ ] First full paper draft assembled
- [ ] Status note in `docs/status/2025-05-24.md`
- [ ] Decide whether to drop E4, E5, or E6 from scope

### Week 4: Polish and submission (25–31 May)

**Monday 25 May** — *3 hours*
- [ ] All figures generated: NHS regions map, pipeline diagram, forecast plots, allocation maps, Pareto fronts
- [ ] Tables polished: forecasting comparison, allocation comparison, ablation, sensitivity

**Tuesday 26 May** — *3 hours*
- [ ] Discussion and Limitations section
- [ ] Conclusion section
- [ ] References tidy in BibTeX

**Wednesday 27 May** — *3 hours*
- [ ] Trust-level NSGA-II at full scale (computational comparison table)
- [ ] Final E5 robustness experiment results

**Thursday 28 May** — *3 hours*
- [ ] Springer LNNS template formatting
- [ ] First proofread pass
- [ ] Internal review pass (look for AIIM overlap red flags specifically)

**Friday 29 May** — *3 hours*
- [ ] Address internal review comments
- [ ] Final figure polish
- [ ] Cross-reference check (every citation, every figure label, every equation reference)

**Saturday 30 May** — *6 hours*
- [ ] Camera-ready PDF generated
- [ ] Source files prepared for CMT (.tex, figures, .bib, title.txt, authors.txt)
- [ ] Reproducibility statement and code repo link

**Sunday 31 May** — *Submission day*
- [ ] Submit via Microsoft CMT before 23:59 UTC
- [ ] Save submission confirmation email and screenshot
- [ ] Status note in `docs/status/2025-05-31.md`

### Time-box rules

1. **Maximum 3 hours per evening** on UKCI work, **6 hours per weekend day**. HOPE-MOVE deliverables to Matt come first.
2. If behind by more than 2 days at any gate, declare scope reduction in writing, not silently.
3. **Last 5 days are writing only.** No new feature work after 26 May.
4. If household disruptions occur (Precious's pregnancy is third trimester), take the full day off and document in status note. Do not work in fragments while exhausted.

---

## 3. Experiment matrix

Six experiments, each producing a defined paper artefact.

### Experiment E1: Forecasting accuracy

**Question:** Does the proposed PINN-GRU with decision-aware loss outperform standard time-series and deep-learning baselines on multi-horizon NHS regional critical-care demand forecasting?

**Variables:**
- Model: {Persistence, Seasonal naive, ARIMA, Prophet, XGBoost, LSTM, GRU, Seq2Seq attention, **PINN-GRU (proposed)**}
- Horizon: {7, 14, 21, 28} days
- Target: {MV beds, occupied COVID beds, admissions}

**Metrics:** MAE, RMSE, sMAPE, MASE, Underestimation Rate, Expected Shortage, Peak Error, Peak Timing Error, WIS (proposed model only)

**Output:** Paper Table 1 (main results), Paper Figure 3 (per-horizon error bars), Appendix tables for secondary targets

**Estimated compute:** ~30 model-runs × 5 seeds = 150 runs total. Each PINN-GRU run ~30 minutes on A100. Baselines much faster. Total ~20 hours wall-clock with parallelisation.

### Experiment E2: Regional MILP allocation

**Question:** Does forecast-driven robust MILP allocation outperform population/demand-proportional and greedy baselines on the 7-region NHS problem?

**Variables:**
- Allocation policy: {No surge, Pop-proportional, Demand-proportional, Greedy, Det-MILP, Robust-MILP}
- Forecast source: {Proposed model, Oracle (ground truth)}
- Budget: 3 levels {low, medium, high}
- Travel-time threshold: 3 levels {60, 120, 240 min}

**Metrics:** Total Unmet Demand, Coverage Rate, Mean Travel Burden, Max Regional Shortage Ratio, Theil Index

**Output:** Paper Table 2 (allocation comparison), Paper Figure 4 (allocation heatmap by region), Paper Figure 5 (sensitivity to budget)

**Estimated compute:** 6 policies × 2 forecast sources × 3 budgets × 3 thresholds = 108 MILP solves. Each solves in seconds for regional problem. Total <2 hours.

### Experiment E3: Trust-level metaheuristic comparison

**Question:** At trust-level scale, do GA, NSGA-II, and SA achieve solution quality comparable to MILP at substantially lower runtime?

**Variables:**
- Solver: {Det-MILP, Robust-MILP, GA, NSGA-II, SA}
- Trust-level scope: {London region (~30 trusts), Midlands (~25 trusts)}
- Random seeds: 5 per metaheuristic

**Metrics:** Best-found objective, runtime, MILP gap (MILP only), hypervolume (NSGA-II only), spread (NSGA-II only)

**Output:** Paper Table 3 (computational comparison), Paper Figure 6 (objective vs runtime curves)

**Estimated compute:** 5 solvers × 2 scopes × 5 seeds = 50 runs. MILP solves in minutes; metaheuristics ~10 minutes each. Total ~10 hours.

### Experiment E4: Pareto front analysis (NSGA-II)

**Question:** What is the Pareto frontier between cost, expected unmet demand, and travel burden under realistic NHS data?

**Variables:**
- Trust scope: {full England trust-level subset}
- Scenario set: {3 quantile scenarios}
- Generations: 200

**Metrics:** Hypervolume vs generations, Pareto front visualisation (3-objective)

**Output:** Paper Figure 7 (Pareto front), Appendix Figure (hypervolume convergence)

**Estimated compute:** 5 seeds × ~1 hour = 5 hours.

### Experiment E5: Robustness to forecast quality

**Question:** How sensitive is allocation quality to forecast quality? Specifically, does a worse forecaster (ARIMA) lead to substantially worse allocation decisions?

**Variables:**
- Forecast source: {Proposed model, ARIMA, Historical median, Oracle}
- All other variables held at default

**Metrics:** Total Unmet Demand under realised demand (oracle), comparison across forecasters

**Output:** Paper Table 4 or Paper Figure 8 (decision quality vs forecast quality)

**Estimated compute:** ~30 minutes wall-clock. Cheap.

### Experiment E6: Sensitivity analysis

**Question:** How sensitive are allocation results to scenario count, equity weight, and CVaR weight?

**Variables:**
- Number of scenarios: {1, 3, 5}
- Equity weight $\lambda_2$: {0, 1, 10, 100, 1000}
- CVaR weight $\lambda_3$: {0, 1, 10, 100}

**Metrics:** Allocation maps, Total Unmet Demand, Max Regional Shortage Ratio

**Output:** Appendix Tables and Figures. Possibly relegated to supplementary material if space tight.

**Estimated compute:** ~3 × 5 × 4 = 60 MILP solves. Total ~1 hour.

### Total compute envelope

Approximately **40 hours of GPU/CPU compute** across all experiments. Comfortably within Brosnan/Zeus capacity for 25 days. The bottleneck is wall-clock time per individual run, not aggregate compute.

---

## 4. Status note template

Each gate decision is recorded in `docs/status/YYYY-MM-DD.md`:

```
# Status — {date}

## Where I am
- Completed: ...
- In progress: ...
- Blocked: ...

## Gate decision
- Gate: {G1 / G2 / G3 / G4}
- Decision: {pass / fail / scope-reduce}
- Reasoning: ...

## Adjustments to plan
- ...

## Next 48 hours
- ...
```

---

*End of timeline document.*
