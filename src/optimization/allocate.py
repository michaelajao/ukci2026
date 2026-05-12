"""Consolidated allocation module for the UKCI 2026 robust surge-capacity
pipeline.

Per the 12 May 2026 author decision to reduce file count, this single module
combines what was originally six separate files:

1. ``OptimisationData`` — sets, parameters, decision-variable bookkeeping
   (``02_METHODOLOGY.md`` §0 + §3.1.1).
2. ``build_milp_model`` + ``solve_milp`` — Pyomo deterministic MILP with
   optional CVaR-flavoured robust extension and optional IMD-weighted
   fairness constraints (§3.1.2-3.1.4).
3. Heuristics — ``no_surge_allocation``, ``population_proportional``,
   ``imd_proportional``, ``demand_proportional``, ``greedy_shortage_first``
   (§3.7).
4. Metaheuristics — ``ga_with_lp_slave``, ``nsga2_pareto`` (§3.4-3.5).
5. Comparator — ``simulated_annealing`` (§3.6); kept Appendix-only.
6. ``augmented_epsilon_constraint`` — exact Pareto-front generator per
   Mavrotas (2009) + Kargar et al. (2024); methodologically complements
   NSGA-II.

Smoke test runs via ``python -m optimization.allocate``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from evaluation.allocation_metrics import AllocationSolution

EPS = 1e-9


# ===========================================================================
# 1. Data model
# ===========================================================================


@dataclass
class OptimisationData:
    """Container for everything the allocation problem needs.

    Conventions follow ``02_METHODOLOGY.md`` §0.

    Attributes:
        regions: ordered list of region identifiers (the demand-node index ``i``).
        facilities: ordered list of facility identifiers (``j``).
        horizons: ordered list of horizons ``H`` in days.
        scenarios: ordered list of scenario names.
        scenario_weights: dict mapping scenario -> probability ``π_s``.
        demand: 3-D array ``d[i, h, s]``.
        baseline_capacity: 2-D array ``C[j, h]``.
        max_expansion: 1-D array ``K[j]`` (upper bound on extra capacity).
        fixed_cost: 1-D array ``F[j]``.
        marginal_cost: 1-D array ``g[j]``.
        transfer_cost: 2-D array ``c[i, j]``.
        travel_time: 2-D array ``T[i, j]``.
        max_travel: scalar ``τ`` (max acceptable travel time).
        population: 1-D array ``p[i]`` (per region; for equity).
        budget: scalar ``B`` (total infrastructure budget).
        imd_weight: optional 1-D array per region for IMD fairness;
            larger values indicate higher deprivation. Used to weight the
            equity penalty when present.
        population_share_tolerance: ``Θ_L`` in Bertsimas et al. (2022)
            eq. (25); allowable deviation from population-share-proportional
            allocation. Set to ``None`` to disable.
        worst_scenarios: subset of ``scenarios`` defining the worst-case set
            ``S_worst`` used by the CVaR-flavoured robust extension.
        lambda1: penalty weight on expected unmet demand.
        lambda2: penalty weight on max-shortage-ratio equity term.
        lambda3: penalty weight on worst-case unmet demand (CVaR).
        epsilon: numerical fudge for the equity denominator.
    """

    regions: list[str]
    facilities: list[str]
    horizons: list[int]
    scenarios: list[str]
    scenario_weights: dict[str, float]
    demand: np.ndarray
    baseline_capacity: np.ndarray
    max_expansion: np.ndarray
    fixed_cost: np.ndarray
    marginal_cost: np.ndarray
    transfer_cost: np.ndarray
    travel_time: np.ndarray
    max_travel: float
    population: np.ndarray
    budget: float
    imd_weight: np.ndarray | None = None
    population_share_tolerance: float | None = None
    worst_scenarios: list[str] = field(default_factory=list)
    lambda1: float = 1.0
    lambda2: float = 0.5
    lambda3: float = 0.0
    epsilon: float = 1.0

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    @property
    def n_facilities(self) -> int:
        return len(self.facilities)

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)

    @property
    def n_scenarios(self) -> int:
        return len(self.scenarios)

    def feasible_pairs(self) -> list[tuple[int, int]]:
        """Return ``(i, j)`` index pairs with ``T_ij <= τ``."""
        pairs: list[tuple[int, int]] = []
        for i in range(self.n_regions):
            for j in range(self.n_facilities):
                if self.travel_time[i, j] <= self.max_travel:
                    pairs.append((i, j))
        return pairs


# ===========================================================================
# 2. MILP (Pyomo)
# ===========================================================================


def build_milp_model(data: OptimisationData, robust: bool = False):
    """Construct the Pyomo MILP for the allocation problem.

    Implements every constraint in ``02_METHODOLOGY.md`` §3.1.3:
    demand satisfaction, capacity, budget, travel-time feasibility,
    and equity linearisation via per-region θ_i auxiliaries. When
    ``robust=True``, also adds the CVaR-flavoured worst-case shortage
    term per §3.1.4.

    Args:
        data: ``OptimisationData`` instance.
        robust: Whether to include the CVaR extension (``λ_3 W``).

    Returns:
        A configured ``pyomo.environ.ConcreteModel`` ready to solve.
    """
    import pyomo.environ as pyo

    R = range(data.n_regions)
    J = range(data.n_facilities)
    H = range(data.n_horizons)
    S = range(data.n_scenarios)

    pi = np.array(
        [data.scenario_weights[s] for s in data.scenarios], dtype=float
    )

    m = pyo.ConcreteModel("ukci2026_allocation")

    # ---- decision variables ----
    m.x = pyo.Var(J, within=pyo.Binary)
    m.b = pyo.Var(J, H, within=pyo.NonNegativeReals)
    m.z = pyo.Var(R, J, H, S, within=pyo.NonNegativeReals)
    m.u = pyo.Var(R, H, S, within=pyo.NonNegativeReals)
    m.theta_i = pyo.Var(R, within=pyo.NonNegativeReals)
    m.theta = pyo.Var(within=pyo.NonNegativeReals)
    if robust:
        m.W = pyo.Var(within=pyo.NonNegativeReals)

    # ---- demand satisfaction ----
    def demand_rule(m, i, h, s):
        return (
            sum(m.z[i, j, h, s] for j in J) + m.u[i, h, s]
            >= float(data.demand[i, h, s])
        )

    m.demand_con = pyo.Constraint(R, H, S, rule=demand_rule)

    # ---- capacity ----
    def capacity_rule(m, j, h, s):
        return (
            sum(m.z[i, j, h, s] for i in R)
            <= float(data.baseline_capacity[j, h]) + m.b[j, h]
        )

    m.capacity_con = pyo.Constraint(J, H, S, rule=capacity_rule)

    def expansion_rule(m, j, h):
        return m.b[j, h] <= float(data.max_expansion[j]) * m.x[j]

    m.expansion_con = pyo.Constraint(J, H, rule=expansion_rule)

    # ---- budget ----
    m.budget_con = pyo.Constraint(
        expr=sum(float(data.fixed_cost[j]) * m.x[j] for j in J)
        + sum(float(data.marginal_cost[j]) * m.b[j, h] for j in J for h in H)
        <= float(data.budget)
    )

    # ---- travel-time feasibility ----
    for i in R:
        for j in J:
            if data.travel_time[i, j] > data.max_travel:
                for h in H:
                    for s in S:
                        m.z[i, j, h, s].fix(0)

    # ---- equity linearisation (max-shortage-ratio) ----
    def theta_i_rule(m, i):
        denom = sum(
            pi[s] * float(data.demand[i, h, s]) / float(data.population[i])
            for h in H
            for s in S
        ) / max(float(data.population[i]), 1e-6) + data.epsilon
        numer = sum(
            pi[s] * m.u[i, h, s] / float(data.population[i])
            for h in H
            for s in S
        )
        return m.theta_i[i] * denom >= numer

    m.theta_i_con = pyo.Constraint(R, rule=theta_i_rule)

    def theta_max_rule(m, i):
        return m.theta >= m.theta_i[i]

    m.theta_max_con = pyo.Constraint(R, rule=theta_max_rule)

    # ---- optional IMD-weighted upper bound on θ_i for high-IMD regions ----
    if data.imd_weight is not None:
        # Strengthen the equity term for the most-deprived regions by upper
        # bounding θ_i with an IMD-weighted slack. We keep the constraint
        # mild — the dominant equity signal flows through θ in the objective.
        imd = np.asarray(data.imd_weight, dtype=float)
        imd_norm = imd / max(imd.max(), 1e-9)

        def imd_bound_rule(m, i):
            return m.theta_i[i] <= 1.0 - 0.3 * float(imd_norm[i]) + 0.5

        m.imd_bound_con = pyo.Constraint(R, rule=imd_bound_rule)

    # ---- optional population-share bound on facility share (Bertsimas 2022 eq. 25) ----
    if data.population_share_tolerance is not None:
        total_pop = float(data.population.sum())
        N_open = pyo.summation(m.x)

        def lo_rule(m, j):
            # Use population of the *region* nearest to facility j (assume
            # a 1:1 mapping when facilities are co-located with regions; if
            # not, the caller can adapt this constraint).
            i_ref = j if j < data.n_regions else 0
            share = float(data.population[i_ref]) / total_pop
            return m.x[j] >= (share - data.population_share_tolerance) * N_open

        m.pop_share_lo = pyo.Constraint(J, rule=lo_rule)

    # ---- robust extension (CVaR-flavoured) ----
    if robust:
        worst_idx = [data.scenarios.index(s) for s in data.worst_scenarios]
        if not worst_idx:
            worst_idx = [int(np.argmax(pi))]

        def worst_rule(m, sk):
            return m.W >= sum(m.u[i, h, sk] for i in R for h in H)

        m.worst_con = pyo.Constraint(worst_idx, rule=worst_rule)

    # ---- objective ----
    fixed_cost_term = sum(float(data.fixed_cost[j]) * m.x[j] for j in J)
    expansion_term = sum(
        float(data.marginal_cost[j]) * m.b[j, h] for j in J for h in H
    )
    transfer_term = sum(
        pi[s] * float(data.transfer_cost[i, j]) * m.z[i, j, h, s]
        for i in R
        for j in J
        for h in H
        for s in S
    )
    unmet_term = sum(
        pi[s] * m.u[i, h, s] for i in R for h in H for s in S
    )
    equity_term = m.theta

    obj_expr = (
        fixed_cost_term
        + expansion_term
        + transfer_term
        + data.lambda1 * unmet_term
        + data.lambda2 * equity_term
    )
    if robust:
        obj_expr = obj_expr + data.lambda3 * m.W

    m.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

    return m


DEFAULT_SOLVER: str = "glpk"
"""Default MILP solver. GLPK is the conda-installed solver in the
``pyt_env`` environment and is the only one routinely tested. Override by
passing ``solver_name=`` to :func:`solve_milp`."""


def solve_milp(
    model,
    *,
    solver_name: str = DEFAULT_SOLVER,
    time_limit: int | None = None,
) -> dict:
    """Solve a Pyomo model with a single named MILP solver.

    No multi-solver fallback: if the requested solver is unavailable or the
    solve fails the error propagates so the experiment fails loudly.

    Args:
        model: A ``pyomo.environ.ConcreteModel``.
        solver_name: MILP solver to use. Defaults to :data:`DEFAULT_SOLVER`.
        time_limit: Optional time limit in seconds.

    Returns:
        Dict with keys ``"status"``, ``"runtime"``, ``"objective"``,
        ``"solver"``, ``"optimal"``.
    """
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, TerminationCondition

    solver = SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            f"MILP solver {solver_name!r} is not available. "
            f"For the UKCI 2026 conda env install GLPK via "
            f"``conda install -c conda-forge glpk`` and ensure "
            f"``$CONDA_PREFIX/Library/bin`` is on PATH."
        )
    if time_limit is not None:
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = time_limit
        elif solver_name == "cbc":
            solver.options["seconds"] = time_limit
        elif solver_name == "glpk":
            solver.options["tmlim"] = time_limit
    results = solver.solve(model, tee=False)
    tc = results.solver.termination_condition
    return {
        "status": str(tc),
        "objective": float(pyo.value(model.obj)),
        "runtime": float(results.solver.time or 0.0),
        "solver": solver_name,
        "optimal": tc == TerminationCondition.optimal,
    }


def solution_from_model(model, data: OptimisationData) -> AllocationSolution:
    """Extract an ``AllocationSolution`` from a solved Pyomo model."""
    import pyomo.environ as pyo

    b_rows = [
        {
            "facility": data.facilities[j],
            "horizon": data.horizons[h],
            "b": float(pyo.value(model.b[j, h])),
        }
        for j in range(data.n_facilities)
        for h in range(data.n_horizons)
    ]
    z_rows = [
        {
            "region": data.regions[i],
            "facility": data.facilities[j],
            "horizon": data.horizons[h],
            "scenario": data.scenarios[s],
            "z": float(pyo.value(model.z[i, j, h, s])),
        }
        for i in range(data.n_regions)
        for j in range(data.n_facilities)
        for h in range(data.n_horizons)
        for s in range(data.n_scenarios)
    ]
    u_rows = [
        {
            "region": data.regions[i],
            "horizon": data.horizons[h],
            "scenario": data.scenarios[s],
            "u": float(pyo.value(model.u[i, h, s])),
        }
        for i in range(data.n_regions)
        for h in range(data.n_horizons)
        for s in range(data.n_scenarios)
    ]
    return AllocationSolution(
        b=pd.DataFrame.from_records(b_rows).set_index(["facility", "horizon"]),
        z=pd.DataFrame.from_records(z_rows).set_index(
            ["region", "facility", "horizon", "scenario"]
        ),
        u=pd.DataFrame.from_records(u_rows).set_index(
            ["region", "horizon", "scenario"]
        ),
        objective_value=float(pyo.value(model.obj)),
    )


def deterministic_milp(data: OptimisationData, **solver_kwargs) -> AllocationSolution:
    """Convenience: build → solve → extract for the deterministic MILP."""
    m = build_milp_model(data, robust=False)
    info = solve_milp(m, **solver_kwargs)
    sol = solution_from_model(m, data)
    sol.meta.update(info)
    return sol


def robust_milp(data: OptimisationData, **solver_kwargs) -> AllocationSolution:
    """Convenience: build → solve → extract for the CVaR-robust MILP."""
    m = build_milp_model(data, robust=True)
    info = solve_milp(m, **solver_kwargs)
    sol = solution_from_model(m, data)
    sol.meta.update(info)
    return sol


# ===========================================================================
# 3. Heuristics (no metaheuristic) — §3.7
# ===========================================================================


def _empty_solution(data: OptimisationData) -> AllocationSolution:
    """Return an empty (no-surge) solution skeleton."""
    b = pd.DataFrame(
        [
            {"facility": j, "horizon": h, "b": 0.0}
            for j in data.facilities
            for h in data.horizons
        ]
    ).set_index(["facility", "horizon"])
    z = pd.DataFrame(
        [
            {
                "region": r,
                "facility": j,
                "horizon": h,
                "scenario": s,
                "z": 0.0,
            }
            for r in data.regions
            for j in data.facilities
            for h in data.horizons
            for s in data.scenarios
        ]
    ).set_index(["region", "facility", "horizon", "scenario"])
    u = pd.DataFrame(
        [
            {"region": r, "horizon": h, "scenario": s, "u": 0.0}
            for r in data.regions
            for h in data.horizons
            for s in data.scenarios
        ]
    ).set_index(["region", "horizon", "scenario"])
    return AllocationSolution(b=b, z=z, u=u)


def _greedy_assign(data: OptimisationData, sol: AllocationSolution) -> None:
    """Fill ``sol.z`` and ``sol.u`` greedily given ``sol.b``.

    For each (region, horizon, scenario): satisfy demand from the
    cheapest feasible facility first; residual unmet demand goes to
    ``u``. Mutates ``sol`` in place.
    """
    cap = (
        data.baseline_capacity
        + sol.b.unstack("horizon")["b"]
        .reindex(data.facilities)
        .reindex(columns=data.horizons)
        .to_numpy()
    )
    z_index = sol.z.index
    z_arr = np.zeros(len(z_index))
    u_index = sol.u.index
    u_arr = np.zeros(len(u_index))
    sorted_facilities_per_region = []
    for i in range(data.n_regions):
        feasible = [
            (data.transfer_cost[i, j], j)
            for j in range(data.n_facilities)
            if data.travel_time[i, j] <= data.max_travel
        ]
        feasible.sort()
        sorted_facilities_per_region.append([j for _, j in feasible])

    for h_idx, h in enumerate(data.horizons):
        for s_idx, s in enumerate(data.scenarios):
            remaining_cap = cap[:, h_idx].copy()
            for i_idx, region in enumerate(data.regions):
                demand = float(data.demand[i_idx, h_idx, s_idx])
                for j_idx in sorted_facilities_per_region[i_idx]:
                    if demand <= EPS or remaining_cap[j_idx] <= EPS:
                        continue
                    take = min(demand, remaining_cap[j_idx])
                    key = (
                        region,
                        data.facilities[j_idx],
                        h,
                        s,
                    )
                    pos = z_index.get_loc(key)
                    z_arr[pos] = take
                    remaining_cap[j_idx] -= take
                    demand -= take
                upos = u_index.get_loc((region, h, s))
                u_arr[upos] = demand
    sol.z = pd.DataFrame({"z": z_arr}, index=z_index)
    sol.u = pd.DataFrame({"u": u_arr}, index=u_index)


def no_surge_allocation(data: OptimisationData) -> AllocationSolution:
    """No additional capacity — assigns only against the baseline.

    A pure shortage scenario. Useful as the lower bound for E2 comparisons.
    """
    sol = _empty_solution(data)
    _greedy_assign(data, sol)
    return sol


def _proportional_template(
    data: OptimisationData, weights: np.ndarray
) -> AllocationSolution:
    """Build a proportional allocation against an arbitrary weight vector
    over facilities (one weight per facility).

    Activates the top facilities whose cumulative weight reaches the budget,
    then distributes expansion capacity proportionally.
    """
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    if weights.sum() < EPS:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    # Choose facility activation pattern: open all facilities (binary x = 1)
    # up to the budget allowed by fixed costs; remaining budget goes into b.
    budget = float(data.budget)
    fixed_total = float(data.fixed_cost.sum())
    if fixed_total > budget:
        # Activate cheapest first
        order = np.argsort(data.fixed_cost)
        active = np.zeros(data.n_facilities, dtype=bool)
        spent = 0.0
        for j in order:
            if spent + float(data.fixed_cost[j]) <= budget:
                active[j] = True
                spent += float(data.fixed_cost[j])
    else:
        active = np.ones(data.n_facilities, dtype=bool)
        spent = fixed_total

    remaining = max(budget - spent, 0.0)
    # Allocate remaining proportionally to weights (only on active facilities)
    sol = _empty_solution(data)
    if remaining > EPS and active.any():
        share = np.where(active, weights, 0.0)
        if share.sum() < EPS:
            share = active.astype(float) / max(active.sum(), 1)
        else:
            share = share / share.sum()
        for j in range(data.n_facilities):
            if not active[j]:
                continue
            per_horizon = (
                remaining * share[j] / (data.n_horizons * max(data.marginal_cost[j], 1.0))
            )
            cap = float(data.max_expansion[j])
            per_horizon = min(per_horizon, cap)
            for h in data.horizons:
                sol.b.loc[(data.facilities[j], h), "b"] = per_horizon
    _greedy_assign(data, sol)
    return sol


def population_proportional(data: OptimisationData) -> AllocationSolution:
    """Allocate extra capacity in proportion to population per facility.

    Assumes facility ``j`` is associated with the region of the same index,
    which holds for the NHS regional scope. Caller can pre-aggregate
    populations before constructing ``OptimisationData`` for trust-level.
    """
    weights = data.population[: data.n_facilities].astype(float)
    return _proportional_template(data, weights)


def imd_proportional(data: OptimisationData) -> AllocationSolution:
    """Allocate extra capacity in proportion to IMD weights (more deprived
    regions receive more), falling back to population-proportional if
    ``imd_weight`` is ``None``."""
    if data.imd_weight is None:
        return population_proportional(data)
    weights = np.asarray(data.imd_weight[: data.n_facilities], dtype=float)
    return _proportional_template(data, weights)


def demand_proportional(data: OptimisationData) -> AllocationSolution:
    """Allocate extra capacity in proportion to scenario-weighted demand
    per facility (treated as same-index as region)."""
    pi = np.array([data.scenario_weights[s] for s in data.scenarios])
    per_region = np.einsum("ihs,s->i", data.demand, pi)
    weights = per_region[: data.n_facilities]
    return _proportional_template(data, weights)


def greedy_shortage_first(data: OptimisationData) -> AllocationSolution:
    """Iteratively allocate budget to the facility-horizon pair with the
    largest projected shortfall, until the budget is exhausted.

    Cheap, useful comparator for E2.
    """
    sol = _empty_solution(data)
    pi = np.array([data.scenario_weights[s] for s in data.scenarios])
    remaining_budget = float(data.budget)
    # Activate every facility (assume sunk fixed cost is paid)
    fixed = float(data.fixed_cost.sum())
    if fixed > remaining_budget:
        return population_proportional(data)
    remaining_budget -= fixed

    expansion = np.zeros((data.n_facilities, data.n_horizons))
    while remaining_budget > EPS:
        cap = data.baseline_capacity + expansion
        # Compute shortfall per (facility, horizon) under expected demand.
        per_jh = np.zeros((data.n_facilities, data.n_horizons))
        for i in range(data.n_regions):
            for h in range(data.n_horizons):
                for s in range(data.n_scenarios):
                    demand = pi[s] * data.demand[i, h, s]
                    # Assign to facility i (1:1 NHS-regional default), with
                    # spillover to the nearest other feasible facility.
                    j0 = i if i < data.n_facilities else 0
                    served = min(demand, cap[j0, h])
                    cap[j0, h] -= served
                    per_jh[j0, h] += max(demand - served, 0.0)
        flat_idx = int(np.argmax(per_jh))
        j_top, h_top = divmod(flat_idx, data.n_horizons)
        max_short = per_jh[j_top, h_top]
        if max_short < EPS:
            break
        max_extra = data.max_expansion[j_top] - expansion[j_top].max(initial=0.0)
        unit_cost = max(float(data.marginal_cost[j_top]), 1e-3)
        step = min(max_short, max_extra, remaining_budget / unit_cost)
        if step <= EPS:
            break
        expansion[j_top, h_top] += step
        remaining_budget -= step * unit_cost
    for j_idx, j_name in enumerate(data.facilities):
        for h_idx, h_name in enumerate(data.horizons):
            sol.b.loc[(j_name, h_name), "b"] = float(expansion[j_idx, h_idx])
    _greedy_assign(data, sol)
    return sol


# ===========================================================================
# 4. Metaheuristics — §3.4-3.6
# ===========================================================================


def _ga_evaluate(data: OptimisationData, x_mask: np.ndarray) -> AllocationSolution:
    """LP slave: given an opening pattern ``x_mask``, solve the residual
    LP in ``b, z, u``. Returns the resulting ``AllocationSolution``.

    Used by both single-objective GA and multi-objective NSGA-II.
    """
    import pyomo.environ as pyo

    # Build a copy of data with x fixed
    m = build_milp_model(data, robust=False)
    for j_idx, opened in enumerate(x_mask):
        m.x[j_idx].fix(int(opened))
    info = solve_milp(m)
    sol = solution_from_model(m, data)
    sol.meta.update(info)
    return sol


def ga_with_lp_slave(
    data: OptimisationData,
    *,
    pop_size: int = 100,
    n_gen: int = 200,
    p_crossover: float = 0.9,
    p_mutation: float | None = None,
    elitism: int = 5,
    rng_seed: int = 42,
) -> AllocationSolution:
    """Single-objective GA with LP-slave decomposition (``§3.4``).

    Master chromosome is a binary vector over facilities. For each candidate
    we solve the residual LP in ``b, z, u``. Returns the best solution found.

    Args:
        data: ``OptimisationData``.
        pop_size: GA population size.
        n_gen: Maximum number of generations.
        p_crossover: Uniform-crossover probability.
        p_mutation: Bit-flip probability per gene. Defaults to ``1/n_facilities``.
        elitism: Number of top individuals carried over each generation.
        rng_seed: Seed for the random number generator.

    Returns:
        The best ``AllocationSolution`` found.
    """
    rng = random.Random(rng_seed)
    n = data.n_facilities
    if p_mutation is None:
        p_mutation = 1.0 / max(n, 1)

    def random_individual() -> np.ndarray:
        return np.array([rng.random() < 0.7 for _ in range(n)], dtype=bool)

    def crossover(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        child1 = a.copy()
        child2 = b.copy()
        for k in range(n):
            if rng.random() < p_crossover:
                child1[k], child2[k] = child2[k], child1[k]
        return child1, child2

    def mutate(c: np.ndarray) -> np.ndarray:
        out = c.copy()
        for k in range(n):
            if rng.random() < p_mutation:
                out[k] = not out[k]
        return out

    pop = [random_individual() for _ in range(pop_size)]
    best_sol: AllocationSolution | None = None
    best_obj = math.inf
    plateau = 0
    for gen in range(n_gen):
        evaluated: list[tuple[float, AllocationSolution, np.ndarray]] = []
        for ind in pop:
            sol = _ga_evaluate(data, ind.astype(int))
            evaluated.append((sol.objective_value, sol, ind))
        evaluated.sort(key=lambda t: t[0])
        if evaluated[0][0] < best_obj - 1e-6:
            best_obj = evaluated[0][0]
            best_sol = evaluated[0][1]
            plateau = 0
        else:
            plateau += 1
            if plateau >= 30:
                break

        # Elite carry-over
        new_pop = [t[2] for t in evaluated[:elitism]]
        # Tournament selection
        while len(new_pop) < pop_size:
            a = min(rng.sample(evaluated, 3), key=lambda t: t[0])[2]
            b = min(rng.sample(evaluated, 3), key=lambda t: t[0])[2]
            c1, c2 = crossover(a, b)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = new_pop
    if best_sol is None:
        raise RuntimeError("GA: no feasible solution found")
    return best_sol


def nsga2_pareto(
    data: OptimisationData,
    *,
    pop_size: int = 100,
    n_gen: int = 200,
    rng_seed: int = 42,
) -> list[AllocationSolution]:
    """NSGA-II Pareto front over three objectives (``§3.5``):

    1. Infrastructure cost ``f_1 = ΣF_j x_j + Σg_j b_{j,h}``
    2. Expected unmet demand ``f_2 = Σπ_s u_{i,h,s}``
    3. Transfer burden ``f_3 = Σπ_s c_ij z_{ij,h,s}``

    Uses ``pymoo`` with binary chromosome encoding for ``x``. Each
    evaluation calls the LP slave to fix the residual problem.

    Args:
        data: ``OptimisationData``.
        pop_size: Population size.
        n_gen: Number of generations.
        rng_seed: Seed for reproducibility.

    Returns:
        List of ``AllocationSolution``s on the final non-dominated front.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize

    pi = np.array([data.scenario_weights[s] for s in data.scenarios])

    class SurgeProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=data.n_facilities,
                n_obj=3,
                n_constr=0,
                xl=0,
                xu=1,
                vtype=bool,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            sol = _ga_evaluate(data, np.asarray(x, dtype=int))
            fixed = float((data.fixed_cost * np.asarray(x, dtype=int)).sum())
            expansion = float(
                (data.marginal_cost[:, None] * sol.b.unstack("horizon")["b"].to_numpy()).sum()
            )
            f1 = fixed + expansion
            f2 = float(
                sum(
                    pi[s_idx]
                    * sol.u.xs(s, level="scenario")["u"].sum()
                    for s_idx, s in enumerate(data.scenarios)
                )
            )
            f3 = 0.0
            z_df = sol.z.reset_index()
            for i_idx, ri in enumerate(data.regions):
                for j_idx, rj in enumerate(data.facilities):
                    c_ij = float(data.transfer_cost[i_idx, j_idx])
                    if c_ij <= 0:
                        continue
                    sub = z_df[(z_df["region"] == ri) & (z_df["facility"] == rj)]
                    for _, row in sub.iterrows():
                        f3 += c_ij * pi[data.scenarios.index(row["scenario"])] * row["z"]
            out["F"] = [f1, f2, f3]

    problem = SurgeProblem()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob=1.0 / data.n_facilities),
        eliminate_duplicates=True,
    )
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=rng_seed, verbose=False)

    solutions: list[AllocationSolution] = []
    for x_row in res.X:
        sol = _ga_evaluate(data, np.asarray(x_row, dtype=int))
        sol.meta["nsga2_obj"] = list(map(float, res.F[len(solutions)]))
        solutions.append(sol)
    return solutions


def simulated_annealing(
    data: OptimisationData,
    *,
    T0: float = 100.0,
    gamma: float = 0.95,
    iters_per_T: int = 50,
    T_stop: float = 0.01,
    rng_seed: int = 42,
) -> AllocationSolution:
    """SA comparator with single-bit-flip neighbourhood (``§3.6``).

    Kept Appendix-only per the 12 May 2026 author decision (augmented
    ε-constraint replaces SA in the headline comparison). Implementation
    is intentionally minimal.
    """
    rng = random.Random(rng_seed)
    n = data.n_facilities

    current = np.array([rng.random() < 0.7 for _ in range(n)], dtype=bool)
    current_sol = _ga_evaluate(data, current.astype(int))
    best = current
    best_sol = current_sol

    T = T0
    while T > T_stop:
        for _ in range(iters_per_T):
            neighbour = current.copy()
            idx = rng.randint(0, n - 1)
            neighbour[idx] = not neighbour[idx]
            cand_sol = _ga_evaluate(data, neighbour.astype(int))
            delta = cand_sol.objective_value - current_sol.objective_value
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                current, current_sol = neighbour, cand_sol
                if current_sol.objective_value < best_sol.objective_value:
                    best, best_sol = current, current_sol
        T *= gamma
    return best_sol


# ===========================================================================
# 5. Augmented ε-constraint — Mavrotas (2009) / Kargar et al. (2024)
# ===========================================================================


def augmented_epsilon_constraint(
    data: OptimisationData,
    *,
    n_points: int = 10,
    augment_weight: float = 1e-3,
    solver_kwargs: dict | None = None,
) -> list[AllocationSolution]:
    """Augmented ε-constraint Pareto-front generator for cost-vs-unmet.

    Implements the bi-objective formulation from §3.4 of Kargar et al.
    (2024) adapted to the surge problem:

        minimise   cost  -  augment_weight * s
        s.t.       unmet_demand <= ε_k
                   slack s = ε_k - unmet_demand,  s >= 0
                   (plus all original feasibility constraints)

    Sweeps ``ε`` across ``n_points`` values evenly spaced between the
    minimum-unmet solution (``ε_max``) and the most-permissive
    unmet bound. Returns one ``AllocationSolution`` per ε.

    Args:
        data: ``OptimisationData``.
        n_points: Number of Pareto points to generate.
        augment_weight: ``s_k`` augmentation weight; must be small enough
            that it does not perturb the primary cost ranking but large
            enough to push solutions to non-dominated boundaries.
        solver_kwargs: Passed through to ``solve_milp``.

    Returns:
        List of ``AllocationSolution``s.
    """
    import pyomo.environ as pyo

    solver_kwargs = solver_kwargs or {}

    # Estimate ε range using two anchor solves
    data_min_unmet = OptimisationData(
        **{
            **data.__dict__,
            "lambda1": 1e6,  # crush cost weight; focus on unmet
            "lambda2": 0.0,
        }
    )
    anchor_min = deterministic_milp(data_min_unmet, **solver_kwargs)
    eps_min = float(anchor_min.u["u"].sum())

    anchor_max = deterministic_milp(data, **solver_kwargs)
    eps_max = float(anchor_max.u["u"].sum())

    if eps_max <= eps_min + EPS:
        return [anchor_max]

    eps_grid = np.linspace(eps_min, eps_max, n_points)

    front: list[AllocationSolution] = []
    for eps_k in eps_grid:
        m = build_milp_model(data, robust=False)
        m.s_slack = pyo.Var(within=pyo.NonNegativeReals)

        m.eps_con = pyo.Constraint(
            expr=sum(m.u[i, h, s] for i in range(data.n_regions)
                     for h in range(data.n_horizons)
                     for s in range(data.n_scenarios)) + m.s_slack == float(eps_k)
        )
        # Re-objective: cost - augment_weight * s
        cost_expr = sum(
            float(data.fixed_cost[j]) * m.x[j] for j in range(data.n_facilities)
        ) + sum(
            float(data.marginal_cost[j]) * m.b[j, h]
            for j in range(data.n_facilities)
            for h in range(data.n_horizons)
        )
        m.del_component("obj")
        m.obj = pyo.Objective(expr=cost_expr - augment_weight * m.s_slack,
                              sense=pyo.minimize)
        info = solve_milp(m, **solver_kwargs)
        sol = solution_from_model(m, data)
        sol.meta.update(info)
        sol.meta["epsilon"] = float(eps_k)
        front.append(sol)
    return front

