"""Allocation evaluation metrics for Experiments E2-E5 and the
multi-objective Pareto-front evaluation in E4.

Implements every operational, equity, robustness, and Pareto metric in
``02_METHODOLOGY.md`` §4.2 and §4.3:

- ``total_unmet_demand``
- ``coverage_rate``
- ``mean_travel_burden``
- ``max_regional_shortage_ratio``
- ``theil_index``
- ``worst_case_unmet``
- ``value_of_robust_solution``
- ``hypervolume``
- ``pareto_spread``

The metrics take ``AllocationSolution`` instances and a ``ScenarioSet`` (see
``data.scenarios``). A solution is a thin dataclass holding the decision
variables :math:`b_{j,h}, z_{ij,h}^s, u_{i,h}^s` and a few cost components.

Smoke test runs via ``python -m evaluation.allocation_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

EPS = 1e-9


# ---------------------------------------------------------------------------
# Solution container
# ---------------------------------------------------------------------------


@dataclass
class AllocationSolution:
    """Container for an allocation problem solution.

    Attributes:
        b: Extra surge capacity per facility per horizon. ``DataFrame``
            indexed by ``(facility, horizon)`` with column ``b``.
        z: Transfers from region ``i`` to facility ``j`` per horizon per
            scenario. ``DataFrame`` indexed by
            ``(region, facility, horizon, scenario)`` with column ``z``.
        u: Unmet demand per region per horizon per scenario. ``DataFrame``
            indexed by ``(region, horizon, scenario)`` with column ``u``.
        objective_value: Optimal objective (composite cost or chosen
            scalarisation).
        cost_components: Optional dict of cost-component contributions
            (e.g. ``{"fixed": ..., "transfer": ..., "unmet": ...}``).
        meta: Free-form metadata (solver runtime, gap, etc.).
    """

    b: pd.DataFrame
    z: pd.DataFrame
    u: pd.DataFrame
    objective_value: float = 0.0
    cost_components: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Operational metrics (§4.2)
# ---------------------------------------------------------------------------


def total_unmet_demand(solution: AllocationSolution, weights: pd.Series) -> float:
    r"""Total scenario-weighted unmet demand.

    .. math::
        U = \sum_{i, h, s} \pi_s\, u_{i,h}^s

    Args:
        solution: An ``AllocationSolution``.
        weights: Scenario weight ``Series`` (index = scenario name).

    Returns:
        Float total unmet demand.
    """
    u = solution.u.reset_index()
    u["weight"] = u["scenario"].map(weights).fillna(0.0)
    return float((u["u"] * u["weight"]).sum())


def coverage_rate(
    solution: AllocationSolution,
    weights: pd.Series,
    demand: pd.DataFrame,
) -> float:
    r"""Scenario-weighted coverage rate.

    .. math::
        \text{Coverage} = 1 - \frac{\sum_{i,h,s} \pi_s u_{i,h}^s}
        {\sum_{i,h,s} \pi_s d_{i,h}^s}

    Args:
        solution: An ``AllocationSolution``.
        weights: Scenario weight ``Series``.
        demand: Demand ``DataFrame`` indexed by
            ``(region, horizon, scenario)`` with column ``d``.

    Returns:
        Float in ``[0, 1]`` (or higher if over-allocated; we do not clip).
    """
    u = solution.u.reset_index()
    u["weight"] = u["scenario"].map(weights).fillna(0.0)
    total_unmet = float((u["u"] * u["weight"]).sum())
    d = demand.reset_index()
    d["weight"] = d["scenario"].map(weights).fillna(0.0)
    total_demand = float((d["d"] * d["weight"]).sum())
    if total_demand < EPS:
        return 1.0
    return 1.0 - total_unmet / total_demand


def mean_travel_burden(
    solution: AllocationSolution,
    weights: pd.Series,
    travel_time: pd.DataFrame,
) -> float:
    r"""Mean patient-weighted travel burden.

    .. math::
        \bar{T} = \frac{\sum_{i,j,h,s} \pi_s T_{ij} z_{ij,h}^s}
        {\sum_{i,j,h,s} \pi_s z_{ij,h}^s}

    Args:
        solution: An ``AllocationSolution``.
        weights: Scenario weight ``Series``.
        travel_time: ``DataFrame`` indexed by ``(region, facility)`` with
            column ``T``.

    Returns:
        Patient-weighted mean travel time.
    """
    z = solution.z.reset_index()
    z["weight"] = z["scenario"].map(weights).fillna(0.0)
    z = z.merge(travel_time.reset_index(), on=["region", "facility"], how="left")
    numerator = float((z["z"] * z["T"] * z["weight"]).sum())
    denominator = float((z["z"] * z["weight"]).sum())
    if denominator < EPS:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Equity metrics (§4.2)
# ---------------------------------------------------------------------------


def max_regional_shortage_ratio(
    solution: AllocationSolution,
    weights: pd.Series,
    demand: pd.DataFrame,
    population: pd.Series,
) -> float:
    r"""Maximum across regions of the per-capita-normalised shortage ratio.

    .. math::
        \theta = \max_i \frac{\sum_{h,s} \pi_s u_{i,h}^s / p_i}
        {\sum_{h,s} \pi_s d_{i,h}^s / p_i + \varepsilon}

    Note that the per-capita normalisation cancels in the ratio when applied
    uniformly to numerator and denominator. We retain ``population`` in the
    signature so callers can switch to per-capita ratios if desired.

    Args:
        solution: An ``AllocationSolution``.
        weights: Scenario weight ``Series``.
        demand: Demand ``DataFrame``.
        population: ``Series`` indexed by region (unused in the simplified
            ratio; kept for API parity).

    Returns:
        Maximum regional shortage ratio in ``[0, 1+]``.
    """
    u = solution.u.reset_index()
    u["weight"] = u["scenario"].map(weights).fillna(0.0)
    u_region = u.groupby("region").apply(
        lambda df: float((df["u"] * df["weight"]).sum())
    )
    d = demand.reset_index()
    d["weight"] = d["scenario"].map(weights).fillna(0.0)
    d_region = d.groupby("region").apply(
        lambda df: float((df["d"] * df["weight"]).sum())
    )
    ratios = u_region / (d_region + EPS)
    return float(ratios.max())


def theil_index(
    solution: AllocationSolution,
    weights: pd.Series,
    population: pd.Series,
) -> float:
    r"""Theil T inequality index over regional per-capita unmet demand.

    .. math::
        T = \sum_i \frac{x_i}{\bar x}\, \log \frac{x_i}{\bar x}, \quad
        x_i = \frac{\sum_{h,s}\pi_s u_{i,h}^s}{p_i}

    Args:
        solution: An ``AllocationSolution``.
        weights: Scenario weight ``Series``.
        population: ``Series`` indexed by region with column ``p``.

    Returns:
        Theil index in ``[0, log(n)]``. Returns ``0`` for uniform allocations.
    """
    u = solution.u.reset_index()
    u["weight"] = u["scenario"].map(weights).fillna(0.0)
    per_region = u.groupby("region").apply(
        lambda df: float((df["u"] * df["weight"]).sum())
    )
    x = per_region / (population.reindex(per_region.index).astype(float) + EPS)
    x = x[x > 0]
    if x.empty:
        return 0.0
    mean_x = float(x.mean())
    if mean_x < EPS:
        return 0.0
    ratios = x / mean_x
    return float((ratios * np.log(ratios)).sum() / len(x))


# ---------------------------------------------------------------------------
# Robustness metrics (§4.2)
# ---------------------------------------------------------------------------


def worst_case_unmet(solution: AllocationSolution) -> float:
    r"""Maximum unmet demand across scenarios:
    :math:`\max_s \sum_{i,h} u_{i,h}^s`.

    Args:
        solution: An ``AllocationSolution``.

    Returns:
        Worst-case unmet demand (unweighted).
    """
    per_scenario = solution.u.reset_index().groupby("scenario")["u"].sum()
    return float(per_scenario.max()) if not per_scenario.empty else 0.0


def value_of_robust_solution(
    cost_robust_worst: float, cost_deterministic_worst: float
) -> float:
    r"""Value of the Robust Solution (VRS).

    .. math::
        \text{VRS} = \frac{c^{\text{worst}}_{\text{rob}}
        - c^{\text{worst}}_{\text{det}}}{c^{\text{worst}}_{\text{det}}}

    Positive VRS means the robust solution achieves a strictly worse
    worst-case cost (unusual; usually negative). Returns ``0`` if the
    deterministic worst-case cost is non-positive.

    Args:
        cost_robust_worst: Worst-case cost of the robust solution
            (max over scenarios of the actualised cost).
        cost_deterministic_worst: Same quantity for the deterministic
            solution.

    Returns:
        VRS scalar.
    """
    if cost_deterministic_worst < EPS:
        return 0.0
    return (cost_robust_worst - cost_deterministic_worst) / cost_deterministic_worst


# ---------------------------------------------------------------------------
# Pareto / multi-objective metrics (§4.3)
# ---------------------------------------------------------------------------


def hypervolume(
    pareto_points: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    r"""Hypervolume indicator of a non-dominated set against a reference point.

    Uses ``pymoo.indicators.hv.HV``.

    Args:
        pareto_points: ``(n_points, n_objectives)`` array of non-dominated
            solutions to be **minimised**.
        reference_point: Length-``n_objectives`` array. Should be strictly
            dominated by every Pareto point.

    Returns:
        Hypervolume scalar.
    """
    from pymoo.indicators.hv import HV

    if pareto_points.size == 0:
        return 0.0
    return float(HV(ref_point=reference_point)(pareto_points))


def pareto_spread(pareto_points: np.ndarray) -> float:
    r"""Spread (uniformity) of a Pareto front using mean pairwise Euclidean
    distance between consecutive solutions sorted lexicographically.

    Args:
        pareto_points: ``(n_points, n_objectives)`` array.

    Returns:
        Mean inter-point distance; ``0`` for fewer than 2 points.
    """
    if pareto_points.shape[0] < 2:
        return 0.0
    sorted_pts = pareto_points[np.lexsort(pareto_points.T)]
    diffs = np.diff(sorted_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(dists.mean())


# ---------------------------------------------------------------------------
# Convenience: full report
# ---------------------------------------------------------------------------


def evaluate_solution(
    solution: AllocationSolution,
    scenarios,  # data.scenarios.ScenarioSet
    travel_time: pd.DataFrame | None = None,
    population: pd.Series | None = None,
) -> dict[str, float]:
    """Compute every operational + equity + robustness metric in one call.

    Args:
        solution: An ``AllocationSolution``.
        scenarios: A ``data.scenarios.ScenarioSet`` providing ``demand`` and
            ``weights``.
        travel_time: Optional ``DataFrame`` for travel-burden computation.
        population: Optional ``Series`` for the Theil index.

    Returns:
        ``dict[str, float]`` with the computed metrics.
    """
    weights = scenarios.weights
    demand = scenarios.demand
    out: dict[str, float] = {
        "total_unmet": total_unmet_demand(solution, weights),
        "coverage_rate": coverage_rate(solution, weights, demand),
        "worst_case_unmet": worst_case_unmet(solution),
    }
    if travel_time is not None:
        out["mean_travel_burden"] = mean_travel_burden(solution, weights, travel_time)
    if population is not None:
        out["max_regional_shortage_ratio"] = max_regional_shortage_ratio(
            solution, weights, demand, population
        )
        out["theil_index"] = theil_index(solution, weights, population)
    return out

