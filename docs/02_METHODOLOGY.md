# 02 — Methodology Deep Dive

This document specifies the full methodology for the UKCI 2026 paper. Cross-references to other documents:
`01_RESEARCH_PROGRAMME.md` (overall plan), `03_TIMELINE.md` (work plan and experiments).

---

## 0. Notation

| Symbol | Domain | Meaning |
|---|---|---|
| $r$ | $\mathcal{R}$, $\|\mathcal{R}\| = 7$ | NHS England region (NE+Y, NW, Mid, EE, Lon, SE, SW) |
| $j$ | $\mathcal{J}$ | Candidate surge facility site |
| $i$ | $\mathcal{I}$ | Demand node (region for regional model, trust for trust-level) |
| $t$ | $\mathbb{N}$ (days) | Time index |
| $h$ | $\{7, 14, 21, 28\}$ | Forecast horizon (days) |
| $L$ | $= 28$ | Lookback window (days) |
| $\mathbf{x}_{r,t}$ | $\mathbb{R}^{d_x}$ | Observed feature vector at region $r$, time $t$ |
| $y_{r,t}$ | $\mathbb{R}_{\geq 0}$ | Forecasting target (mechanical ventilation bed occupancy unless stated) |
| $\hat{y}_{r,t+h}$ | $\mathbb{R}_{\geq 0}$ | Point forecast |
| $q_{r,h}^{p}$ | $\mathbb{R}_{\geq 0}$ | Empirical $p$-th quantile of forecast |
| $s$ | $\mathcal{S}$ | Demand scenario index |
| $\pi_s$ | $[0,1]$ | Scenario probability/weight |
| $d_{i,h}^s$ | $\mathbb{R}_{\geq 0}$ | Forecast demand at node $i$, horizon $h$, scenario $s$ |
| $C_{j,h}$ | $\mathbb{R}_{\geq 0}$ | Baseline capacity at facility $j$ at horizon $h$ |
| $K_j$ | $\mathbb{R}_{\geq 0}$ | Maximum surge capacity expansion at site $j$ |
| $F_j$, $g_j$ | $\mathbb{R}_{\geq 0}$ | Fixed activation cost, marginal capacity cost |
| $c_{ij}$, $T_{ij}$ | $\mathbb{R}_{\geq 0}$ | Per-patient transfer cost, travel time |
| $\tau$ | $\mathbb{R}_{\geq 0}$ | Maximum acceptable travel time |
| $B$ | $\mathbb{R}_{\geq 0}$ | Total budget |
| $p_i$ | $\mathbb{N}$ | Population of node $i$ |
| $x_j$ | $\{0,1\}$ | Decision: open/activate facility $j$ |
| $b_{j,h}$ | $\mathbb{R}_{\geq 0}$ | Decision: extra capacity at $j$, horizon $h$ |
| $z_{ij,h}^s$ | $\mathbb{R}_{\geq 0}$ | Decision: patients allocated $i \to j$, horizon $h$, scenario $s$ |
| $u_{i,h}^s$ | $\mathbb{R}_{\geq 0}$ | Decision: unmet demand at $i$, horizon $h$, scenario $s$ |

---

## 1. Phase A — Per-Region Physics-Informed Forecasting

### 1.1 Compartmental model (refined from Ajao-Olarinoye et al. 2025)

For each region $r$, we model the local epidemic using a refined $SEI_aI_sHCRD$ system. With population $N_r$:

$$
\begin{aligned}
\dot{S}_r &= -\beta_r S_r (I_{s,r} + I_{a,r}) / N_r + \eta_r R_r \\
\dot{E}_r &= \beta_r S_r (I_{s,r} + I_{a,r}) / N_r - \alpha E_r \\
\dot{I}_{s,r} &= \alpha \rho E_r - d_s I_{s,r} \\
\dot{I}_{a,r} &= \alpha (1-\rho) E_r - d_a I_{a,r} \\
\dot{H}_r &= d_s \omega I_{s,r} - d_H H_r - \mu H_r \\
\dot{C}_r &= \phi \, d_H H_r - \gamma_{c,r} C_r - \delta_{c,r} C_r \\
\dot{R}_r &= d_s (1-\omega) I_{s,r} + d_a I_{a,r} + (1-\phi) d_H H_r + \gamma_{c,r} C_r - \eta_r R_r \\
\dot{D}_r &= \mu H_r + \delta_{c,r} C_r
\end{aligned}
$$

with conservation $\sum_{\bullet} \bullet_r(t) = N_r$.

**Refinement over Ajao-Olarinoye et al. (2025).** The book chapter overloaded the parameter $\omega$ to represent both the proportion of symptomatic cases requiring hospitalisation ($I_s \to H$) and, via $(1-\omega)$, the proportion of hospitalised cases progressing to critical care ($H \to C$). These are biologically distinct quantities — the hospitalisation ratio of symptomatic cases is determined by clinical severity at admission, whereas the critical-care escalation rate is determined by in-hospital deterioration — and using a single parameter for both produces unrealistic coupling. We therefore introduce a separate parameter $\phi$ for the $H \to C$ escalation proportion, leaving $\omega$ to denote the hospitalisation ratio alone. The compartmental flow becomes biologically interpretable: of those leaving hospital alive (rate $d_H$), a proportion $\phi$ progress to critical care and $(1-\phi)$ recover directly, while a separate rate $\mu$ governs in-hospital mortality.

**Fixed and learned parameters.** The fixed (non-learned) parameters $\alpha, \rho, d_s, d_a, d_H, \mu, \omega, \phi$ are reciprocals of clinical periods or proportions, and are calibrated against the values in Ajao-Olarinoye et al. (2025) Table 1. The time-varying parameters $\beta_r, \gamma_{c,r}, \delta_{c,r}, \eta_r$ are learned by the parameter network $X^r_{NN}$ (§1.2). The overall compartmental structure is reused infrastructure from the book chapter, cited explicitly; the $\phi$ refinement is the only modelling change.

**Notation note on rates and periods.** The symbols $\alpha, d_s, d_a, d_H$ are *rates* with units of inverse-days; their reciprocals $1/\alpha, 1/d_s, 1/d_a, 1/d_H$ are the corresponding *periods* in days (e.g., incubation period $1/\alpha = 5$ days, hospitalisation period $1/d_H = 13.4$ days). We follow this convention strictly in the paper to avoid the rate/period confusion present in some compartmental modelling literature.

### 1.2 Per-region PINN architecture

For each region $r$, two neural networks (Ajao-Olarinoye et al. 2025 hyperparameters):

- **State network** $U^r_{NN}: t \mapsto (S_r, E_r, I_{a,r}, I_{s,r}, H_r, C_r, R_r, D_r)$. 5 hidden layers, 20 units, tanh activation, sigmoid output. Xavier initialisation.
- **Parameter network** $X^r_{NN}: t \mapsto (\beta_r, \gamma_{c,r}, \delta_{c,r}, \eta_r)$. 3 hidden layers, 20 units, tanh activation, sigmoid output.

We instantiate one pair of networks per region (7 pairs total). The networks share architecture but not weights — this respects regional heterogeneity in transmission and clinical pathway dynamics.

### 1.3 PINN training loss

The PINN loss for region $r$ has data and residual components:

$$
\mathcal{L}^{PINN}_r = \mathcal{L}^{data}_r + \lambda_{ode} \mathcal{L}^{ode}_r
$$

where the data fit term is on observable compartments only ($I_s, H, C, D$ from public data; $S, E, I_a, R$ are latent):

$$
\mathcal{L}^{data}_r = \sum_{t \in \mathcal{T}_{train}} \sum_{k \in \{I_s, H, C, D\}} \big( U^r_{NN,k}(t) - y_{r,t,k} \big)^2
$$

and the ODE residual is the standard PINN loss against the SEIRD system $\mathcal{N}$:

$$
\mathcal{L}^{ode}_r = \sum_{t \in \mathcal{T}_{collocation}} \left\| \frac{d U^r_{NN}(t)}{dt} - \mathcal{N}\big(U^r_{NN}(t); X^r_{NN}(t), \theta_{fixed}\big) \right\|^2
$$

evaluated at collocation points via PyTorch automatic differentiation. We use $\lambda_{ode} = 0.1$ initial, tuned on validation.

### 1.4 Temporal forecasting head

Once the PINN is trained, the state estimates $\tilde{U}^r(t) = U^r_{NN}(t)$ and parameter estimates $\tilde{X}^r(t) = X^r_{NN}(t)$ become inputs to a temporal forecasting head, alongside observed covariates.

For each region $r$, we form an augmented feature sequence:

$$
\mathbf{f}_{r,t} = \big[ \tilde{U}^r(t) \,\|\, \tilde{X}^r(t) \,\|\, \mathbf{x}_{r,t} \big] \in \mathbb{R}^{d_x + 12}
$$

The temporal head is a 2-layer GRU with hidden dimension 64, applied per region:

$$
\mathbf{h}_{r,t} = \text{GRU}\big(\mathbf{f}_{r,t-L+1}, \ldots, \mathbf{f}_{r,t}\big)
$$

A multi-horizon decoder produces forecasts for all four horizons jointly (multi-task, not iterative):

$$
\hat{y}_{r,t+h} = \mathbf{w}_h^\top \mathbf{h}_{r,t} + b_h, \quad h \in \{7, 14, 21, 28\}
$$

We deliberately use a **per-region GRU** rather than a shared/graph-coupled architecture for two reasons: (i) it provides clear separation from the MSAGAT-Net AIIM submission, (ii) regional heterogeneity in transmission and hospital-flow dynamics may genuinely outweigh inter-regional coupling at multi-horizon scales, where the temporal autocorrelation structure dominates. We test this assumption in Experiment E1 by comparing against a regional-pooled GRU baseline.

### 1.5 Decision-aware composite loss

The **load-bearing methodological choice** at training time. The composite loss is:

$$
\mathcal{L}_{total} = \mathcal{L}_{forecast} + \lambda_{phys} \sum_r \mathcal{L}^{PINN}_r + \lambda_{under} \mathcal{L}_{under} + \lambda_{smooth} \mathcal{L}_{smooth}
$$

Each term:

- **Forecast loss (Huber, robust to spike outliers):**
  $$
  \mathcal{L}_{forecast} = \sum_{r, h} \rho_\delta \big( y_{r,t+h} - \hat{y}_{r,t+h} \big)
  $$
  with $\rho_\delta(x) = \frac{1}{2}x^2$ for $|x| \leq \delta$, $\delta(|x| - \frac{1}{2}\delta)$ otherwise; $\delta = 1$ on standardised values.

- **Asymmetric underestimation penalty (decision-aware core):**
  $$
  \mathcal{L}_{under} = \sum_{r, h} \max\big(0,\; y_{r,t+h} - \hat{y}_{r,t+h}\big)
  $$
  This is a one-sided hinge that penalises underforecasting only. Operationally, underestimating critical-care demand by 10 beds is much worse than overestimating by 10: the former produces unmet demand, the latter wastes prepared capacity.

- **Temporal smoothness:**
  $$
  \mathcal{L}_{smooth} = \sum_{r, h} \big( \hat{y}_{r,t+h+1} - \hat{y}_{r,t+h} \big)^2
  $$
  prevents the multi-horizon decoder from producing implausibly oscillating multi-horizon forecasts.

**Initial weights:** $\lambda_{phys} = 0.1$, $\lambda_{under} = 0.5$, $\lambda_{smooth} = 0.01$. Tuned on validation by grid over $\{0.05, 0.1, 0.5, 1.0\}$ for $\lambda_{phys}, \lambda_{under}$.

### 1.6 Uncertainty quantification via MC Dropout

At inference, we enable dropout during the forward pass and run $K = 100$ stochastic forward passes:

$$
\{\hat{y}_{r,t+h}^{(k)}\}_{k=1}^{K}
$$

Then compute empirical quantiles:

$$
q_{r,h}^p = \text{Quantile}_p\big(\{\hat{y}_{r,t+h}^{(k)}\}_{k=1}^{K}\big), \quad p \in \{0.1, 0.5, 0.9\}
$$

We use $p = 0.5$ as the point forecast for E1, and the triple $(q^{0.1}, q^{0.5}, q^{0.9})$ as the basis for scenario generation in Phase B.

**Calibration check.** We report Weighted Interval Score (WIS) on the test set at 50% and 90% prediction intervals, following the COVID-19 Forecast Hub convention.

---

## 2. Phase B — Demand Scenario Generation

### 2.1 Discrete scenario set

From the predictive quantiles, we construct $|\mathcal{S}| = 3$ canonical scenarios per region per horizon:

| Scenario $s$ | Demand $d_{i,h}^s$ | Weight $\pi_s$ |
|---|---|---|
| Low | $q_{i,h}^{0.1}$ | 0.20 |
| Median | $q_{i,h}^{0.5}$ | 0.60 |
| High | $q_{i,h}^{0.9}$ | 0.20 |

Weights are chosen to roughly approximate the implied cumulative distribution (10–50–90 quantiles correspond to 0.4–0.4–0.2 mass within those deciles, but for 3-point discretisation we use 0.2/0.6/0.2 to give the median scenario the dominant weight).

**Optional tail scenario.** For the worst-case planning experiment, we add a fourth scenario at $q_{i,h}^{0.95}$ with weight 0.05 and rebalance. This is reported under Experiment E5.

### 2.2 Scenario consistency

Scenarios are generated independently per (region, horizon). For the full optimisation, scenarios are concatenated across (region, horizon) to form a global demand matrix per scenario. We do **not** assume cross-regional or cross-horizon independence in the forecasts themselves — the MC Dropout samples preserve any coupling the model has learned through shared features and PINN parameters.

---

## 3. Phase C — Robust Metaheuristic Allocation

### 3.1 Mathematical formulation

#### 3.1.1 Sets, parameters, decision variables

See §0 Notation.

#### 3.1.2 Objective function

The deterministic objective is:

$$
\min \underbrace{\sum_{j} F_j x_j + \sum_{j,h} g_j b_{j,h}}_{\text{infrastructure cost}} + \underbrace{\sum_{i,j,h,s} \pi_s c_{ij} z_{ij,h}^s}_{\text{transfer cost}} + \underbrace{\lambda_1 \sum_{i,h,s} \pi_s u_{i,h}^s}_{\text{shortage penalty}} + \underbrace{\lambda_2 \theta}_{\text{equity penalty}}
$$

where $\theta$ is an auxiliary variable representing the maximum per-capita shortage ratio across regions (linearisation of the equity term, see below).

#### 3.1.3 Constraints

**Demand satisfaction (slack form):**
$$
\sum_j z_{ij,h}^s + u_{i,h}^s \geq d_{i,h}^s \quad \forall i, h, s
$$

**Capacity:**
$$
\sum_i z_{ij,h}^s \leq C_{j,h} + b_{j,h} \quad \forall j, h, s
$$
$$
b_{j,h} \leq K_j x_j \quad \forall j, h
$$

**Budget:**
$$
\sum_j F_j x_j + \sum_{j,h} g_j b_{j,h} \leq B
$$

**Travel-time feasibility:**
$$
z_{ij,h}^s = 0 \quad \text{if } T_{ij} > \tau
$$

**Equity linearisation (max-shortage-ratio form):**
$$
\theta \geq \frac{\sum_{h,s} \pi_s u_{i,h}^s / p_i}{\sum_{h,s} \pi_s d_{i,h}^s / p_i + \varepsilon} \quad \forall i
$$

where $\varepsilon = 1$ avoids division-by-zero. Equivalently, we introduce per-region auxiliary variables $\theta_i$ and linearise the ratio by normalising the numerator only (since the denominator is a known constant once forecasts are fixed):

$$
\theta_i \cdot \Big( \sum_{h,s} \pi_s d_{i,h}^s / p_i + \varepsilon \Big) \geq \sum_{h,s} \pi_s u_{i,h}^s / p_i \quad \forall i
$$
$$
\theta \geq \theta_i \quad \forall i
$$

This gives a clean MILP formulation with $\theta$ a single auxiliary continuous variable.

**Variable domains:**
$$
x_j \in \{0,1\}, \quad b_{j,h}, z_{ij,h}^s, u_{i,h}^s, \theta_i, \theta \geq 0
$$

#### 3.1.4 Robust extension (CVaR-flavoured)

Define the worst-case shortage:

$$
W = \max_{s \in \mathcal{S}_{\text{worst}}} \sum_{i,h} u_{i,h}^s
$$

where $\mathcal{S}_{\text{worst}}$ is the upper-quantile scenario subset (e.g., the High scenario alone, or {Median, High, Tail}). Linearise via:

$$
W \geq \sum_{i,h} u_{i,h}^s \quad \forall s \in \mathcal{S}_{\text{worst}}
$$

and add $\lambda_3 W$ to the objective. The hyperparameter $\lambda_3$ controls risk aversion, with $\lambda_3 = 0$ recovering the expected-value formulation and $\lambda_3 \to \infty$ recovering pure worst-case.

### 3.2 Solution method 1 — Exact MILP (regional)

For the regional problem with $|\mathcal{R}| = 7$, $|\mathcal{J}| \leq 15$, $|\mathcal{H}| = 4$, $|\mathcal{S}| = 3$, the MILP has on the order of $10^3$–$10^4$ variables and is solved exactly via Gurobi (academic licence at Coventry) or CBC as fallback. Expected runtime: under 60 seconds.

**Implementation:** Python `pyomo` for model construction, Gurobi solver via `gurobipy`. CBC fallback through `pulp`.

### 3.3 Solution method 2 — Robust MILP

Same model with the CVaR extension. Runtime grows roughly linearly with $|\mathcal{S}_{\text{worst}}|$.

### 3.4 Solution method 3 — Genetic Algorithm (single-objective)

For trust-level scaling, the MILP becomes expensive. We use a master–slave decomposition:

- **Master (GA):** binary chromosome $\mathbf{x} = (x_1, \ldots, x_{|\mathcal{J}|})$ encoding facility-opening decisions.
- **Slave (LP):** for fixed $\mathbf{x}$, the residual problem in $b, z, u$ is a linear programme. Solve via Gurobi/CBC; the LP is small (no binaries) and fast.

**GA configuration (initial values, tuned on validation):**

| Parameter | Value |
|---|---|
| Population size | 100 |
| Generations | 200 |
| Selection | Tournament, $k=3$ |
| Crossover | Uniform, $p_c = 0.9$ |
| Mutation | Bit-flip, $p_m = 1/|\mathcal{J}|$ |
| Elitism | Top 5 |
| Repair operator | If budget violated, deactivate facilities with lowest cost-effectiveness ratio until feasible |
| Stopping criterion | Generations or no improvement for 30 generations |

**Implementation:** `pymoo` library (Blank & Deb 2020).

### 3.5 Solution method 4 — NSGA-II (multi-objective)

The same encoding, but with three objectives optimised simultaneously:

- $f_1 = \sum_j F_j x_j + \sum_{j,h} g_j b_{j,h}$ (infrastructure cost)
- $f_2 = \sum_{i,h,s} \pi_s u_{i,h}^s$ (expected unmet demand)
- $f_3 = \sum_{i,j,h,s} \pi_s c_{ij} z_{ij,h}^s$ (transfer burden)

NSGA-II (Deb et al. 2002) produces a Pareto front. We report:

- Hypervolume relative to a fixed reference point
- Spread / generational distance
- Number of non-dominated solutions

**Configuration:** Same population size (100) and generations (200). `pymoo` `NSGA2` with `BinaryRandomSampling`, `TwoPointCrossover`, `BitflipMutation`.

### 3.6 Solution method 5 — Simulated Annealing (comparator)

Single-objective comparator using the scalarised objective.

| Parameter | Value |
|---|---|
| Initial temperature $T_0$ | 100 |
| Cooling schedule | Geometric, $\gamma = 0.95$ |
| Iterations per temperature | 50 |
| Stopping temperature | 0.01 |
| Neighbourhood | Single bit-flip on $\mathbf{x}$, with LP re-solve |

**Implementation:** Custom Python (about 80 lines) or `scipy.optimize.dual_annealing` with custom callable.

### 3.7 Heuristic baselines (no metaheuristic)

For completeness and as a lower bound on solution quality:

- **Population-proportional:** $b_{j,h} = (B - \sum_j F_j x_j) \cdot p_j / \sum p_{j'}$ for all $j$ with $x_j = 1$.
- **Demand-proportional:** $b_{j,h} = (B - \sum_j F_j x_j) \cdot \bar{d}_j / \sum \bar{d}_{j'}$ where $\bar{d}_j$ is mean forecast demand near $j$.
- **Greedy shortage-first:** iteratively allocate one unit of capacity to the region with current largest predicted shortage until budget is exhausted.

---

## 4. Phase D — Evaluation

### 4.1 Forecasting metrics (Experiment E1)

Per horizon $h$:

- **Mean Absolute Error:** $\text{MAE}_h = \frac{1}{|\mathcal{R}||\mathcal{T}_{test}|} \sum_{r,t} |y_{r,t+h} - \hat{y}_{r,t+h}|$
- **Root Mean Squared Error:** $\text{RMSE}_h = \sqrt{\frac{1}{|\mathcal{R}||\mathcal{T}_{test}|} \sum_{r,t} (y_{r,t+h} - \hat{y}_{r,t+h})^2}$
- **symmetric Mean Absolute Percentage Error:** $\text{sMAPE}_h = \frac{1}{|\mathcal{R}||\mathcal{T}_{test}|} \sum_{r,t} \frac{2 |y_{r,t+h} - \hat{y}_{r,t+h}|}{|y_{r,t+h}| + |\hat{y}_{r,t+h}|}$
- **Mean Absolute Scaled Error:** $\text{MASE}_h = \frac{\text{MAE}_h}{\frac{1}{T-1}\sum_{t=2}^{T} |y_{r,t} - y_{r,t-1}|}$

Calibration:

- **Weighted Interval Score** at 50% and 90% prediction intervals (Bracher et al. 2021), reported per horizon and averaged.

Decision-relevant forecast metrics:

- **Underestimation Rate:** $\frac{1}{|\mathcal{R}||\mathcal{T}_{test}|} \sum_{r,t} \mathbb{1}[\hat{y}_{r,t+h} < y_{r,t+h}]$
- **Expected Shortage:** $\sum_{r,t} \max(0, y_{r,t+h} - \hat{y}_{r,t+h})$
- **Peak Error:** $|\max_t y_{r,t} - \max_t \hat{y}_{r,t}|$ averaged over regions
- **Peak Timing Error:** $|\arg\max_t y_{r,t} - \arg\max_t \hat{y}_{r,t}|$ in days, averaged

### 4.2 Allocation metrics (Experiments E2–E5)

Operational:

- **Total Unmet Demand:** $\sum_{i,h,s} \pi_s u_{i,h}^s$
- **Coverage Rate:** $1 - \frac{\sum_{i,h,s} \pi_s u_{i,h}^s}{\sum_{i,h,s} \pi_s d_{i,h}^s}$
- **Mean Patient-weighted Travel Burden:** $\frac{\sum_{i,j,h,s} \pi_s T_{ij} z_{ij,h}^s}{\sum_{i,j,h,s} \pi_s z_{ij,h}^s}$

Equity:

- **Maximum Regional Shortage Ratio:** $\max_i \frac{\sum_{h,s} \pi_s u_{i,h}^s / p_i}{\sum_{h,s} \pi_s d_{i,h}^s / p_i + \varepsilon}$
- **Theil Index of Regional Per-capita Unmet Demand**

Robustness:

- **Worst-Case Unmet Demand:** $\max_s \sum_{i,h} u_{i,h}^s$ over all scenarios
- **Value of the Robust Solution (VRS):** $\frac{\text{cost}_{\text{robust}}^{\text{worst}} - \text{cost}_{\text{deterministic}}^{\text{worst}}}{\text{cost}_{\text{deterministic}}^{\text{worst}}}$

### 4.3 Computational metrics (Experiment E3)

- Wall-clock runtime
- MILP optimality gap at termination
- NSGA-II hypervolume after $g$ generations
- Number of LP slave solves (for GA/NSGA-II decomposition)

---

## 5. Implementation choices and rationale

### 5.1 Why per-region GRU and not graph attention

We considered using a graph attention forecasting backbone for the spatial coupling step. We deliberately chose per-region PINN-SEIRD plus GRU for four reasons:

1. **Avoids AIIM submission overlap.** A separate manuscript (MSAGAT-Net) currently under review at *Artificial Intelligence in Medicine* benchmarks adaptive graph attention on NHS-ICUBeds. Using any graph attention forecasting backbone in this UKCI paper would invite a direct comparison and risk overlap perception.
2. **Sufficient for the paper's claim.** The contribution of this paper is decision quality — how forecasts translate into surge-capacity allocation outcomes — not forecast accuracy. We need *good enough* forecasts to drive a credible optimisation, not state-of-the-art forecasts.
3. **Forecasting-backend agnostic by design.** The pipeline is structured so that any forecasting model producing point and quantile demand estimates per region per horizon can be substituted. We make this explicit in the discussion: alternative graph-based forecasting backbones are a natural future direction, but evaluating them is outside the scope of this paper.
4. **Implementation timeline.** Reusing the per-region PINN-SEIRD module from Ajao-Olarinoye et al. (2025) plus a small GRU temporal head fits the 25-day window cleanly.

### 5.2 Why MC Dropout and not deep ensembles

Both are valid uncertainty quantification methods. MC Dropout is cheaper (single model), well-supported by PyTorch, and produces credible quantiles for moderately-sized RNNs. Deep ensembles are more accurate but require training $K$ separate models, which is too expensive for the timeline.

### 5.3 Why NSGA-II over MOEA/D, SMS-EMOA, etc.

NSGA-II is the most-cited multi-objective metaheuristic, has mature `pymoo` support, and provides a clean Pareto front for visualisation. SMS-EMOA tends to dominate on hypervolume but is computationally heavier; MOEA/D requires reference-direction tuning. NSGA-II is the right balance for a conference paper.

### 5.4 Why GA + LP decomposition rather than pure GA

Pure GA on the full $(x, b, z, u)$ space is wasteful: for fixed $x$, the residual problem is a linear programme with a known optimal solution. Decomposing into a GA master (over the $|\mathcal{J}|$-dimensional binary $x$) plus an LP slave is standard in facility-location metaheuristics (see Resende & Werneck 2004 for hub location, Cordeau et al. 2007 for capacitated facility location). The decomposition reduces the effective search space dimension by orders of magnitude.

### 5.5 Why Huber over MSE

MSE is sensitive to outliers; NHS critical-care occupancy has occasional spikes (data-collection anomalies, definition changes). Huber loss is quadratic for small errors and linear for large errors, producing more robust training.

### 5.6 Why three scenarios and not 100

Computational cost. Each scenario adds a copy of $\{z, u\}$ to the MILP. Three quantile-based scenarios (Low/Median/High) capture the bulk of the predictive distribution while keeping the MILP tractable. For sensitivity, we run a 5-scenario variant (10/30/50/70/90 quantiles) in Experiment E6.

---

## 6. Hyperparameter summary

### 6.1 SEIRD fixed parameters (calibrated, not learned)

| Parameter | Symbol | Initial value | Source |
|---|---|---|---|
| Symptomatic proportion | $\rho$ | 0.80 | Boddington et al. (2021) |
| Incubation rate (1/period) | $\alpha$ | 1/5 day$^{-1}$ | Lauer et al. (2020) |
| Symptomatic infectious rate | $d_s$ | 1/4 day$^{-1}$ | Docherty et al. (2020) |
| Asymptomatic infectious rate | $d_a$ | 1/7 day$^{-1}$ | Byrne et al. (2020) |
| Hospitalisation outflow rate | $d_H$ | 1/13.4 day$^{-1}$ | Byrne et al. (2020) |
| Hospitalisation ratio ($I_s \to H$) | $\omega$ | calibrated to NHS data | regional fit |
| Critical-care escalation ($H \to C$) | $\phi$ | calibrated to NHS data | regional fit (new in this work) |
| Hospital death rate | $\mu$ | calibrated to NHS data | regional fit |

The values for $\omega$, $\phi$, $\mu$ are estimated per region by least-squares fit of the deterministic SEIRD trajectory to NHS regional data over the training period, prior to PINN training. This gives stable initial conditions for the per-region PINN.

### 6.2 PINN, forecaster, and optimiser hyperparameters

| Component | Parameter | Initial value | Tuning range |
|---|---|---|---|
| PINN | Learning rate | 1e-3 | [1e-4, 1e-2] |
| PINN | $\lambda_{ode}$ | 0.1 | [0.01, 1.0] |
| PINN | Collocation points per epoch | 200 | fixed |
| PINN | Epochs | 5000 | early stopping on val |
| GRU | Hidden dim | 64 | [32, 128] |
| GRU | Layers | 2 | [1, 3] |
| GRU | Dropout | 0.2 | [0.1, 0.5] |
| Composite loss | $\lambda_{phys}$ | 0.1 | [0.05, 1.0] |
| Composite loss | $\lambda_{under}$ | 0.5 | [0.1, 2.0] |
| Composite loss | $\lambda_{smooth}$ | 0.01 | [0.001, 0.1] |
| MC Dropout | Samples $K$ | 100 | fixed |
| MILP | $\lambda_1$ (shortage) | 1000 | tuned for unit balance |
| MILP | $\lambda_2$ (equity) | 100 | sensitivity in E6 |
| MILP | $\lambda_3$ (CVaR) | 10 | sensitivity in E6 |
| GA | Population | 100 | [50, 200] |
| GA | Generations | 200 | early stop on stagnation |
| GA | $p_c$, $p_m$ | 0.9, 1/\|J\| | standard |
| NSGA-II | Same as GA | | |
| SA | $T_0$, $\gamma$ | 100, 0.95 | [50,200], [0.9, 0.99] |

---

## 7. Software stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Standard |
| ML framework | PyTorch 2.x | PINN with autograd, GPU support |
| Optimisation modelling | Pyomo | Solver-agnostic, clean MILP construction |
| MILP solver | Gurobi (academic licence) | State-of-the-art commercial; CBC fallback |
| Metaheuristics | pymoo 0.6+ | NSGA-II, GA, encoding utilities |
| Data | pandas 2.x, numpy, openpyxl | NHS XLSX ingestion |
| Visualisation | matplotlib, seaborn, geopandas | Maps, time series, Pareto fronts |
| Reproducibility | hydra-core, mlflow | Config management, run tracking |
| Testing | pytest, hypothesis | Property-based tests for data harmonisation |
| Packaging | pyproject.toml + uv | Modern Python packaging |
| CI | GitHub Actions | Linting, tests on push |

---

*End of methodology document.*
