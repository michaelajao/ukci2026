"""Regional bed-surge allocation — data + LP formulations + baselines.

This single module consolidates data prep, the exact LPs, and the naive
baseline policies for Phase C of the UKCI 2026 pipeline. Section markers
below split the responsibilities:

  §1. Region centroids, great-circle distance / travel-time matrices,
      baseline-capacity computation, scenario generation from PinnGRU
      q10/q50/q90 forecasts. The ``AllocationProblem`` dataclass packages
      everything every method downstream needs.

  §2. Deterministic and tail-weighted risk-averse LPs via PuLP + CBC.

  §3. Naive baseline policies (status quo, population-proportional,
      demand-proportional, greedy-shortage-first) sharing an LP slave
      that fills in transfers and unmet demand given a fixed ``b``.

Hybrid-scope formulation: sites collapse to NHS regions (``j ≡ r``),
every region is always "open" (no ``x_j`` binary), per-bed cost uniform,
transfer cost proportional to centroid distance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pulp

from utils import repo_root

ROOT = repo_root()


# ===========================================================================
# §1. Data preparation
# ===========================================================================

# NHS England region centroids (latitude, longitude in WGS84). Source: ONS
# Open Geography Portal NHS England Region (E40) centroids, 2022 boundaries.
REGION_CENTROIDS: dict[str, tuple[float, float]] = {
    "Y56": (51.5074, -0.1278),    # London
    "Y58": (50.7772, -3.9997),    # South West (Exeter area)
    "Y59": (51.2787, -0.5217),    # South East (Guildford area)
    "Y60": (52.4862, -1.8904),    # Midlands (Birmingham area)
    "Y61": (52.2053, 0.1218),     # East of England (Cambridge area)
    "Y62": (53.4808, -2.2426),    # North West (Manchester area)
    "Y63": (53.7997, -1.5492),    # North East and Yorkshire (Leeds area)
}

# Real road distance is roughly 1.3× great-circle in the UK motorway network.
# At inter-city motorway average speed of 80 km/h, travel-time in minutes
# follows directly.
DETOUR_FACTOR = 1.3
AVG_SPEED_KMH = 80.0

DEFAULT_REGION_CODES: list[str] = [
    "Y56", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63",
]
DEFAULT_REGION_NAMES: list[str] = [
    "London", "South West", "South East", "Midlands",
    "East of England", "North West", "North East and Yorkshire",
]
DEFAULT_HORIZONS: tuple[int, ...] = (7, 14, 21, 28)
DEFAULT_SCENARIOS: list[str] = ["low", "median", "high"]
DEFAULT_SCENARIO_WEIGHTS: np.ndarray = np.array([0.2, 0.6, 0.2], dtype=float)
DEFAULT_MAX_TRAVEL_MIN: float = 240.0           # 4 hours mutual-aid window
DEFAULT_TRAVEL_COST_PER_KM: float = 1.0
DEFAULT_MAX_EXPANSION_FRACTION: float = 0.5     # K_r = 50% of baseline cap
DEFAULT_BUDGET_FRACTION: float = 0.20           # B = 20% of total baseline cap


@dataclass
class AllocationProblem:
    """All inputs every allocation method needs, in canonical region order."""

    regions: list[str]
    region_codes: list[str]
    horizons: list[int]
    scenarios: list[str]
    scenario_weights: np.ndarray            # (S,)
    demand: np.ndarray                      # (R, H, S)
    baseline_capacity: np.ndarray           # (R,)
    max_expansion: np.ndarray               # (R,)
    population: np.ndarray                  # (R,)
    centroids: np.ndarray                   # (R, 2) lat/lon
    distance_km: np.ndarray                 # (R, R)
    travel_time_min: np.ndarray             # (R, R)
    travel_cost: np.ndarray                 # (R, R)
    budget: float
    max_travel_min: float
    forecast_source: str = "pinn_gru"
    origin: pd.Timestamp | None = None
    # Per-region cap on outbound displaced patients sustained at any
    # checkpoint (clinically: regional critical-care transfer-service
    # capacity). None = uncapped, the headline operating point.
    max_transfer_out: float | None = None

    @property
    def n_regions(self) -> int: return len(self.regions)

    @property
    def n_horizons(self) -> int: return len(self.horizons)

    @property
    def n_scenarios(self) -> int: return len(self.scenarios)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return r * 2.0 * np.arcsin(np.sqrt(a))


def build_distance_matrices(region_codes: Iterable[str]):
    codes = list(region_codes)
    centroids = np.array([REGION_CENTROIDS[c] for c in codes], dtype=float)
    n = len(codes)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist[i, j] = _haversine_km(*centroids[i], *centroids[j])
    travel = (dist * DETOUR_FACTOR / AVG_SPEED_KMH) * 60.0
    return centroids, dist, travel


def build_baseline_capacity(
    regional_daily_csv: Path | str,
    region_codes: list[str],
    val_start: str = "2021-06-01",
    val_end: str = "2021-11-30",
    margin: float = 1.05,
) -> np.ndarray:
    """Operational baseline MV-bed capacity proxied by ``margin × max(mv_beds)``
    over the Delta validation period — the last sustained operational ceiling
    before Omicron. Alpha peaks were one-time Nightingale levels and would
    trivialise the allocation problem if used."""
    daily = pd.read_csv(regional_daily_csv, parse_dates=["date"])
    daily = daily[(daily["date"] >= pd.Timestamp(val_start)) &
                  (daily["date"] <= pd.Timestamp(val_end))]
    cap = np.zeros(len(region_codes), dtype=float)
    for i, code in enumerate(region_codes):
        sub = daily[daily["region_code"] == code]
        if sub.empty:
            raise RuntimeError(f"No daily data for region code {code!r}")
        cap[i] = float(sub["mv_beds"].max()) * margin
    return cap


def build_scenarios_at_origin(
    forecasts_parquet: Path | str,
    region_codes: list[str],
    region_names: list[str],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    forecast_model: str = "pinn_gru",
    origin: pd.Timestamp | None = None,
):
    fc = pd.read_parquet(forecasts_parquet)
    fc = fc[fc["model"] == forecast_model].copy()
    if fc.empty:
        raise RuntimeError(f"No forecasts for model {forecast_model!r}.")
    if origin is None:
        for cand in sorted(fc["origin"].unique()):
            cov = fc[fc["origin"] == cand]
            if cov["region"].nunique() == len(region_codes) and \
               set(cov["horizon"].unique()) >= set(horizons):
                origin = pd.Timestamp(cand)
                break
        if origin is None:
            raise RuntimeError("No origin has full (region × horizon) coverage.")
    sub = fc[fc["origin"] == origin]
    demand = np.zeros((len(region_codes), len(horizons), 3), dtype=float)
    for r_idx, name in enumerate(region_names):
        row_r = sub[sub["region"] == name]
        if row_r.empty:
            raise RuntimeError(f"No forecast for region {name!r} at origin {origin}")
        for h_idx, h in enumerate(horizons):
            row = row_r[row_r["horizon"] == h]
            if row.empty:
                raise RuntimeError(
                    f"No forecast for {name!r}, h={h} at origin {origin}"
                )
            y_hat = float(row["y_hat"].iloc[0])
            q_lo = float(row["q_lo"].iloc[0]) if pd.notna(row["q_lo"].iloc[0]) else y_hat
            q_hi = float(row["q_hi"].iloc[0]) if pd.notna(row["q_hi"].iloc[0]) else y_hat
            # MV-bed demand is non-negative; clip each raw head at zero so the
            # LP demand RHS is never negative. This does not reorder crossings.
            demand[r_idx, h_idx, 0] = max(0.0, q_lo)
            demand[r_idx, h_idx, 1] = max(0.0, y_hat)
            demand[r_idx, h_idx, 2] = max(0.0, q_hi)
    return demand, origin


def realised_demand_at_origin(
    forecasts_parquet: Path | str,
    region_codes: list[str],
    region_names: list[str],
    origin: pd.Timestamp,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
):
    """Pull ``y_true`` at ``origin`` for the given regions and horizons,
    returning a deterministic ``(R, H, 1)`` demand tensor for re-evaluation
    of any allocation under the realised demand path.
    """
    fc = pd.read_parquet(forecasts_parquet)
    fc = fc[fc["model"] == "pinn_gru"]
    sub = fc[fc["origin"] == origin]
    out = np.zeros((len(region_codes), len(horizons), 1), dtype=float)
    for r_idx, name in enumerate(region_names):
        row_r = sub[sub["region"] == name]
        for h_idx, h in enumerate(horizons):
            row = row_r[row_r["horizon"] == h]
            out[r_idx, h_idx, 0] = float(row["y_true"].iloc[0])
    return out


def load_allocation_problem(
    forecasts_parquet: Path | str | None = None,
    regional_daily_csv: Path | str | None = None,
    regional_static_csv: Path | str | None = None,
    forecast_model: str = "pinn_gru",
    origin: pd.Timestamp | None = None,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
    max_expansion_fraction: float = DEFAULT_MAX_EXPANSION_FRACTION,
    max_travel_min: float = DEFAULT_MAX_TRAVEL_MIN,
    travel_cost_per_km: float = DEFAULT_TRAVEL_COST_PER_KM,
    max_transfer_out: float | None = None,
) -> AllocationProblem:
    if forecasts_parquet is None:
        forecasts_parquet = ROOT / "results" / "forecasting" / "forecasts.parquet"
    if regional_daily_csv is None:
        regional_daily_csv = ROOT / "data" / "processed" / "regional_daily.csv"
    if regional_static_csv is None:
        regional_static_csv = ROOT / "data" / "processed" / "regional_static.csv"

    centroids, dist_km, travel_min = build_distance_matrices(DEFAULT_REGION_CODES)
    baseline_cap = build_baseline_capacity(regional_daily_csv, DEFAULT_REGION_CODES)
    demand, picked_origin = build_scenarios_at_origin(
        forecasts_parquet, DEFAULT_REGION_CODES, DEFAULT_REGION_NAMES,
        forecast_model=forecast_model, origin=origin,
    )
    static = pd.read_csv(regional_static_csv)
    pop_lookup = dict(zip(static["region_code"], static["population"]))
    population = np.array(
        [float(pop_lookup[c]) for c in DEFAULT_REGION_CODES], dtype=float
    )
    return AllocationProblem(
        regions=DEFAULT_REGION_NAMES,
        region_codes=DEFAULT_REGION_CODES,
        horizons=list(DEFAULT_HORIZONS),
        scenarios=DEFAULT_SCENARIOS,
        scenario_weights=DEFAULT_SCENARIO_WEIGHTS,
        demand=demand,
        baseline_capacity=baseline_cap,
        max_expansion=baseline_cap * max_expansion_fraction,
        population=population,
        centroids=centroids,
        distance_km=dist_km,
        travel_time_min=travel_min,
        travel_cost=dist_km * travel_cost_per_km,
        budget=float(baseline_cap.sum() * budget_fraction),
        max_travel_min=max_travel_min,
        forecast_source=forecast_model,
        origin=picked_origin,
        max_transfer_out=max_transfer_out,
    )


# ===========================================================================
# §2. Solution container shared by every method
# ===========================================================================

@dataclass
class AllocationSolution:
    method: str
    b: np.ndarray                  # (R, H) extra beds per region per horizon
    z: np.ndarray                  # (R, R, H, S) transfers
    u: np.ndarray                  # (R, H, S) unmet demand
    expected_unmet: float
    transfer_burden: float
    worst_case_unmet: float
    total_surge_beds: float
    objective: float
    runtime_s: float
    status: str = "ok"
    extra: dict | None = None


def _solver(time_limit: float | None = None):
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)


def _lp_slave(
    p: AllocationProblem,
    b_peak: np.ndarray,
    transfer_cost_weight: float = 1e-3,
    time_limit: float | None = 30.0,
):
    """Given a fixed ``b`` per region (peak over horizons), solve the residual
    LP to fill in transfers ``z`` and unmet demand ``u`` minimising
    ``E[unmet] + transfer_cost_weight · transfer_burden``. Returns
    ``(z_arr, u_arr, expected_unmet, transfer_burden, worst_case_unmet)``."""
    R, H, S = p.n_regions, p.n_horizons, p.n_scenarios
    feasible_rr = [
        (r, rp) for r in range(R) for rp in range(R)
        if p.travel_time_min[r, rp] <= p.max_travel_min
    ]
    prob = pulp.LpProblem("lp_slave", pulp.LpMinimize)
    z = {
        (r, rp, h, s): pulp.LpVariable(f"z_{r}_{rp}_{h}_{s}", lowBound=0)
        for (r, rp) in feasible_rr for h in range(H) for s in range(S)
    }
    u = {
        (r, h, s): pulp.LpVariable(f"u_{r}_{h}_{s}", lowBound=0)
        for r in range(R) for h in range(H) for s in range(S)
    }
    for r in range(R):
        out_pairs = [(r, rp) for (r2, rp) in feasible_rr if r2 == r]
        for h in range(H):
            for s in range(S):
                prob += (
                    pulp.lpSum(z[(rr, rp, h, s)] for (rr, rp) in out_pairs)
                    + u[r, h, s] >= float(p.demand[r, h, s]),
                    f"demand_{r}_{h}_{s}",
                )
    if p.max_transfer_out is not None:
        for r in range(R):
            off_pairs = [(r2, rp) for (r2, rp) in feasible_rr
                         if r2 == r and rp != r]
            for h in range(H):
                for s in range(S):
                    prob += (
                        pulp.lpSum(z[(rr, rp, h, s)] for (rr, rp) in off_pairs)
                        <= float(p.max_transfer_out),
                        f"transfer_cap_{r}_{h}_{s}",
                    )
    for rp in range(R):
        in_pairs = [(r, rp) for (r, rp2) in feasible_rr if rp2 == rp]
        for h in range(H):
            cap_rp = float(p.baseline_capacity[rp]) + float(b_peak[rp])
            for s in range(S):
                prob += (
                    pulp.lpSum(z[(r, rp, h, s)] for (r, _) in in_pairs)
                    <= cap_rp,
                    f"cap_{rp}_{h}_{s}",
                )
    pi = p.scenario_weights
    eu = pulp.lpSum(
        pi[s] * u[r, h, s] for r in range(R) for h in range(H) for s in range(S)
    )
    tb = pulp.lpSum(
        pi[s] * float(p.distance_km[r, rp]) * z[r, rp, h, s]
        for (r, rp) in feasible_rr if r != rp
        for h in range(H) for s in range(S)
    )
    prob += eu + transfer_cost_weight * tb
    prob.solve(_solver(time_limit))
    z_arr = np.zeros((R, R, H, S), dtype=float)
    for (r, rp, h, s), var in z.items():
        z_arr[r, rp, h, s] = var.value() or 0.0
    u_arr = np.zeros((R, H, S), dtype=float)
    for (r, h, s), var in u.items():
        u_arr[r, h, s] = var.value() or 0.0
    eu_val = float(sum(pi[s] * u_arr[:, :, s].sum() for s in range(S)))
    tb_val = 0.0
    for r in range(R):
        for rp in range(R):
            if r == rp:
                continue
            tb_val += sum(
                pi[s] * float(p.distance_km[r, rp]) * z_arr[r, rp, :, s].sum()
                for s in range(S)
            )
    wc = float(max(u_arr[:, :, s].sum() for s in range(S)))
    return z_arr, u_arr, eu_val, tb_val, wc


def _wrap_solution(p, method, b_peak, runtime_s,
                   transfer_cost_weight: float = 1e-3) -> AllocationSolution:
    z_arr, u_arr, eu, tb, wc = _lp_slave(p, b_peak, transfer_cost_weight)
    R, H = p.n_regions, p.n_horizons
    b_full = np.broadcast_to(b_peak.reshape(R, 1), (R, H)).copy()
    return AllocationSolution(
        method=method, b=b_full, z=z_arr, u=u_arr,
        expected_unmet=eu, transfer_burden=tb, worst_case_unmet=wc,
        total_surge_beds=float(b_peak.sum()),
        objective=float(eu + transfer_cost_weight * tb),
        runtime_s=runtime_s, status="ok",
    )


# ===========================================================================
# §3. LP formulation (deterministic + tail-weighted risk-averse)
# ===========================================================================

def _build_milp(
    p: AllocationProblem,
    *,
    scenarios_used: list[int],
    scenario_weights: np.ndarray,
    cvar_lambda: float = 0.0,
    cvar_scenarios: list[int] | None = None,
    transfer_cost_weight: float = 1e-3,
    facility_opening: bool = False,
    open_cost: float = 0.0,
):
    R, H = p.n_regions, p.n_horizons
    feasible_rr = [
        (r, rp) for r in range(R) for rp in range(R)
        if p.travel_time_min[r, rp] <= p.max_travel_min
    ]
    pi = {s: float(scenario_weights[i]) for i, s in enumerate(scenarios_used)}
    prob = pulp.LpProblem("regional_bed_surge", pulp.LpMinimize)

    # Single per-region surge variable. Surge only relaxes the capacity
    # constraint and the budget charges the standing allocation once, so a
    # per-horizon copy b_{r,h} would always sit at its peak — the collapsed
    # form is exact (and matches the LP re-evaluation slave).
    b = {
        r: pulp.LpVariable(f"b_{r}", lowBound=0, upBound=float(p.max_expansion[r]))
        for r in range(R)
    }
    z = {
        (r, rp, h, s): pulp.LpVariable(f"z_{r}_{rp}_{h}_{s}", lowBound=0)
        for (r, rp) in feasible_rr for h in range(H) for s in scenarios_used
    }
    u = {
        (r, h, s): pulp.LpVariable(f"u_{r}_{h}_{s}", lowBound=0)
        for r in range(R) for h in range(H) for s in scenarios_used
    }
    # With a single tail scenario, W is equivalent to extra weight on that
    # scenario's unmet term; the auxiliary is retained because it
    # generalises unchanged to multiple tail scenarios (journal extension).
    W = pulp.LpVariable("W_worst", lowBound=0) if cvar_lambda > 0.0 else None
    x = None
    if facility_opening:
        # binary "open a surge unit at r"; surge only where opened; fixed
        # opening cost (in bed-equivalents) competes with beds for the budget.
        x = {r: pulp.LpVariable(f"x_{r}", cat="Binary") for r in range(R)}
        for r in range(R):
            prob += b[r] <= float(p.max_expansion[r]) * x[r], f"open_{r}"
        prob += (pulp.lpSum(b[r] for r in range(R))
                 + open_cost * pulp.lpSum(x[r] for r in range(R))
                 <= float(p.budget)), "budget"
    else:
        prob += pulp.lpSum(b[r] for r in range(R)) <= float(p.budget), "budget"

    for r in range(R):
        out_pairs = [(r, rp) for (r2, rp) in feasible_rr if r2 == r]
        for h in range(H):
            for s in scenarios_used:
                prob += (
                    pulp.lpSum(z[(rr, rp, h, s)] for (rr, rp) in out_pairs)
                    + u[r, h, s] >= float(p.demand[r, h, s]),
                    f"demand_{r}_{h}_{s}",
                )
    if p.max_transfer_out is not None:
        for r in range(R):
            off_pairs = [(r2, rp) for (r2, rp) in feasible_rr
                         if r2 == r and rp != r]
            for h in range(H):
                for s in scenarios_used:
                    prob += (
                        pulp.lpSum(z[(rr, rp, h, s)] for (rr, rp) in off_pairs)
                        <= float(p.max_transfer_out),
                        f"transfer_cap_{r}_{h}_{s}",
                    )
    for rp in range(R):
        in_pairs = [(r, rp) for (r, rp2) in feasible_rr if rp2 == rp]
        for h in range(H):
            for s in scenarios_used:
                prob += (
                    pulp.lpSum(z[(r, rp, h, s)] for (r, _) in in_pairs)
                    <= float(p.baseline_capacity[rp]) + b[rp],
                    f"cap_{rp}_{h}_{s}",
                )
    if cvar_lambda > 0.0:
        if cvar_scenarios is None:
            cvar_scenarios = scenarios_used
        for s in cvar_scenarios:
            prob += (
                W >= pulp.lpSum(u[r, h, s] for r in range(R) for h in range(H)),
                f"W_lb_{s}",
            )

    expected_unmet = pulp.lpSum(
        pi[s] * u[r, h, s]
        for r in range(R) for h in range(H) for s in scenarios_used
    )
    transfer_burden = pulp.lpSum(
        pi[s] * float(p.distance_km[r, rp]) * z[r, rp, h, s]
        for (r, rp) in feasible_rr if r != rp
        for h in range(H) for s in scenarios_used
    )
    obj = expected_unmet + transfer_cost_weight * transfer_burden
    if cvar_lambda > 0.0:
        obj = obj + cvar_lambda * W
    prob += obj, "obj"
    return prob, {"b": b, "z": z, "u": u, "W": W, "x": x,
                  "feasible_rr": feasible_rr, "S_used": scenarios_used, "pi": pi}


def solve_deterministic(
    p: AllocationProblem,
    transfer_cost_weight: float = 1e-3,
    time_limit: float | None = 60.0,
) -> AllocationSolution:
    """Optimise under median (q50) scenario only; honestly re-evaluate the
    chosen ``b`` against the full 3-scenario set via the LP slave."""
    median_idx = p.scenarios.index("median")
    prob, h = _build_milp(p,
                          scenarios_used=[median_idx],
                          scenario_weights=np.array([1.0]),
                          transfer_cost_weight=transfer_cost_weight)
    t0 = time.time()
    status_int = prob.solve(_solver(time_limit))
    runtime = time.time() - t0
    b_peak = np.array(
        [h["b"][r].value() or 0.0 for r in range(p.n_regions)], dtype=float
    )
    z_arr, u_arr, eu, tb, wc = _lp_slave(p, b_peak, transfer_cost_weight)
    R, H = p.n_regions, p.n_horizons
    b_full = np.broadcast_to(b_peak.reshape(R, 1), (R, H)).copy()
    return AllocationSolution(
        method="deterministic_milp",
        b=b_full, z=z_arr, u=u_arr,
        expected_unmet=eu, transfer_burden=tb, worst_case_unmet=wc,
        total_surge_beds=float(b_peak.sum()),
        objective=float(pulp.value(prob.objective) or 0.0),
        runtime_s=runtime, status=pulp.LpStatus[status_int],
    )


def solve_robust(
    p: AllocationProblem,
    transfer_cost_weight: float = 1e-3,
    cvar_lambda: float = 1.0,
    cvar_scenarios: list[str] | None = None,
    time_limit: float | None = 60.0,
    facility_opening: bool = False,
    open_cost: float = 0.0,
) -> AllocationSolution:
    """Scenario-weighted objective plus a tail penalty on the upper-quantile
    scenario set (default ``{high}``). With one tail scenario this is not
    CVaR; the ``cvar_*`` argument names are retained for backward compatibility.
    With ``facility_opening`` the surge placement becomes a MILP (binary
    open/closed per node, fixed ``open_cost`` in bed-equivalents)."""
    S_used = list(range(p.n_scenarios))
    if cvar_scenarios is None:
        cvar_idx = [p.scenarios.index("high")]
    else:
        cvar_idx = [p.scenarios.index(s) for s in cvar_scenarios]
    prob, h = _build_milp(p,
                          scenarios_used=S_used,
                          scenario_weights=p.scenario_weights,
                          cvar_lambda=cvar_lambda, cvar_scenarios=cvar_idx,
                          transfer_cost_weight=transfer_cost_weight,
                          facility_opening=facility_opening, open_cost=open_cost)
    t0 = time.time()
    status_int = prob.solve(_solver(time_limit))
    runtime = time.time() - t0
    R, H, S = p.n_regions, p.n_horizons, p.n_scenarios
    b_peak = np.array(
        [h["b"][r].value() or 0.0 for r in range(R)], dtype=float
    )
    b_arr = np.broadcast_to(b_peak.reshape(R, 1), (R, H)).copy()
    feasible = set(h["feasible_rr"])
    z_arr = np.zeros((R, R, H, S), dtype=float)
    for (r, rp, hh, s), var in h["z"].items():
        if (r, rp) in feasible:
            z_arr[r, rp, hh, s] = var.value() or 0.0
    u_arr = np.zeros((R, H, S), dtype=float)
    for (r, hh, s), var in h["u"].items():
        u_arr[r, hh, s] = var.value() or 0.0
    eu = float(sum(p.scenario_weights[s] * u_arr[:, :, s].sum() for s in range(S)))
    tb = 0.0
    for r in range(R):
        for rp in range(R):
            if r == rp:
                continue
            tb += sum(
                p.scenario_weights[s] * float(p.distance_km[r, rp]) * z_arr[r, rp, :, s].sum()
                for s in range(S)
            )
    wc = float(max(u_arr[:, :, s].sum() for s in range(S)))
    n_open = None
    if h["x"] is not None:
        n_open = int(sum(1 for r in range(R) if (h["x"][r].value() or 0) > 0.5))
    return AllocationSolution(
        method=f"robust_milp_cvar{cvar_lambda:g}",
        b=b_arr, z=z_arr, u=u_arr,
        expected_unmet=eu, transfer_burden=tb, worst_case_unmet=wc,
        total_surge_beds=float(b_arr.max(axis=1).sum()),
        objective=float(pulp.value(prob.objective) or 0.0),
        runtime_s=runtime, status=pulp.LpStatus[status_int],
        extra=({"n_open": n_open} if n_open is not None else None),
    )


# ===========================================================================
# §4. Naive baseline policies (each calls the LP slave)
# ===========================================================================

def status_quo(p: AllocationProblem) -> AllocationSolution:
    """No surge: ``b = 0`` everywhere."""
    t0 = time.time()
    return _wrap_solution(p, "status_quo", np.zeros(p.n_regions), time.time() - t0)


def population_proportional(p: AllocationProblem) -> AllocationSolution:
    t0 = time.time()
    shares = p.population / p.population.sum()
    b_peak = np.minimum(shares * p.budget, p.max_expansion)
    return _wrap_solution(p, "population_proportional", b_peak, time.time() - t0)


def demand_proportional(p: AllocationProblem) -> AllocationSolution:
    t0 = time.time()
    md = p.demand.mean(axis=(1, 2))
    shares = md / md.sum()
    b_peak = np.minimum(shares * p.budget, p.max_expansion)
    return _wrap_solution(p, "demand_proportional", b_peak, time.time() - t0)


def greedy_shortage_first(
    p: AllocationProblem, increment: float = 1.0,
) -> AllocationSolution:
    """Allocate one bed at a time to the region with largest current
    expected shortage."""
    t0 = time.time()
    b_peak = np.zeros(p.n_regions, dtype=float)
    remaining = float(p.budget)
    while remaining > 0:
        _, u_arr, _, _, _ = _lp_slave(p, b_peak)
        shortage = np.array([
            sum(p.scenario_weights[s] * u_arr[r, :, s].sum()
                for s in range(p.n_scenarios))
            for r in range(p.n_regions)
        ])
        shortage[b_peak >= p.max_expansion - 1e-6] = -np.inf
        if not np.isfinite(shortage).any() or shortage.max() <= 1e-6:
            break
        r_star = int(np.argmax(shortage))
        step = float(min(increment, p.max_expansion[r_star] - b_peak[r_star],
                         remaining))
        if step <= 1e-6:
            break
        b_peak[r_star] += step
        remaining -= step
    return _wrap_solution(p, "greedy_shortage_first", b_peak, time.time() - t0)


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    p = load_allocation_problem()
    print(f"Origin {p.origin.date()}, "
          f"forecast {p.forecast_source}, "
          f"budget {p.budget:.0f} surge beds")
    print()
    funcs = [
        ("status_quo",          lambda: status_quo(p)),
        ("pop_prop",            lambda: population_proportional(p)),
        ("demand_prop",         lambda: demand_proportional(p)),
        ("greedy",              lambda: greedy_shortage_first(p)),
        ("deterministic_milp",  lambda: solve_deterministic(p)),
        ("robust_milp",         lambda: solve_robust(p)),
    ]
    for name, fn in funcs:
        sol = fn()
        print(f"{name:25s}  U_pi={sol.expected_unmet:6.1f}  "
              f"worst={sol.worst_case_unmet:6.1f}  "
              f"transfer={sol.transfer_burden:8.1f}  "
              f"surge={sol.total_surge_beds:5.0f}  "
              f"runtime={sol.runtime_s:.2f}s")
