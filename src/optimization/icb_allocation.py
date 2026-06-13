"""Build and solve the 42-ICB critical-care surge allocation instance (journal).

Inputs (all verified, monthly, real):
  data/processed/sitrep_icb_monthly.csv  cc_available (capacity), cc_occupied (demand)
  data/processed/icb_centroids.csv        ICB lat/lon (ONS April-2023 geography)

Demand scenarios are the SYSTEM low / median / high occupancy MONTHS (the high
month = Nov 2021 = the observed adult-critical-care peak); an optional ``stress``
multiplier scales demand to test a wave worse than the observed peak. Baseline
capacity = the median monthly available beds (the standing capacity). Reuses the
dimension-agnostic AllocationProblem + LP solvers from regional_allocation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from optimization.regional_allocation import (
    AllocationProblem, DEFAULT_SCENARIO_WEIGHTS, DETOUR_FACTOR, AVG_SPEED_KMH,
    _haversine_km, demand_proportional, greedy_shortage_first,
    solve_deterministic, solve_robust, status_quo,
)
from utils import repo_root

ROOT = repo_root()
PROC = ROOT / "data" / "processed"


def load_icb_problem(stress: float = 1.0, budget_fraction: float = 0.20,
                     max_expansion_fraction: float = 0.5,
                     max_travel_min: float = 240.0):
    cen = pd.read_csv(PROC / "icb_centroids.csv").sort_values("icb_name")
    ser = pd.read_csv(PROC / "sitrep_icb_monthly.csv", parse_dates=["month"])
    icbs = cen["icb_name"].tolist()
    codes = cen["icb_code"].tolist()
    centroids = cen[["lat", "lon"]].to_numpy(float)
    R = len(icbs)

    dist = np.zeros((R, R))
    for i in range(R):
        for j in range(R):
            if i != j:
                dist[i, j] = _haversine_km(*centroids[i], *centroids[j])
    travel = dist * DETOUR_FACTOR / AVG_SPEED_KMH * 60.0

    occ = ser.pivot(index="month", columns="icb_name", values="cc_occupied").reindex(columns=icbs)
    avail = ser.pivot(index="month", columns="icb_name", values="cc_available").reindex(columns=icbs)
    totals = occ.sum(axis=1).sort_values()
    lo_m, md_m, hi_m = totals.index[0], totals.index[len(totals) // 2], totals.index[-1]

    demand = np.zeros((R, 1, 3))
    demand[:, 0, 0] = occ.loc[lo_m].to_numpy() * stress
    demand[:, 0, 1] = occ.loc[md_m].to_numpy() * stress
    demand[:, 0, 2] = occ.loc[hi_m].to_numpy() * stress
    baseline = avail.median(axis=0).to_numpy()

    prob = AllocationProblem(
        regions=icbs, region_codes=codes, horizons=[1],
        scenarios=["low", "median", "high"], scenario_weights=DEFAULT_SCENARIO_WEIGHTS,
        demand=demand, baseline_capacity=baseline,
        max_expansion=baseline * max_expansion_fraction,
        population=baseline,  # PLACEHOLDER: replace with ICB MYE before equity work
        centroids=centroids, distance_km=dist, travel_time_min=travel,
        travel_cost=dist, budget=float(baseline.sum() * budget_fraction),
        max_travel_min=max_travel_min, forecast_source="sitrep_icb", origin=hi_m,
    )
    return prob, dict(lo=lo_m, md=md_m, hi=hi_m)


def _run(p, label):
    feas = sum(1 for i in range(p.n_regions) for j in range(p.n_regions)
               if i != j and p.travel_time_min[i, j] <= p.max_travel_min)
    print(f"\n=== {label} ===")
    print(f"  baseline (median available) total={p.baseline_capacity.sum():.0f} beds, "
          f"budget={p.budget:.0f}; high-scenario demand total={p.demand[:,0,2].sum():.0f}; "
          f"mutual-aid links={feas}/{p.n_regions*(p.n_regions-1)}")
    for name, fn in [("status_quo", status_quo), ("demand_prop", demand_proportional),
                     ("greedy", greedy_shortage_first),
                     ("deterministic_lp", solve_deterministic), ("robust_lp", solve_robust)]:
        s = fn(p)
        print(f"  {name:17s} E[u]={s.expected_unmet:7.1f} worst={s.worst_case_unmet:7.1f} "
              f"transfer={s.transfer_burden:10.1f} surge={s.total_surge_beds:6.0f} "
              f"t={s.runtime_s:.2f}s")


def main() -> int:
    p0, months = load_icb_problem(stress=1.0)
    print(f"42-ICB instance | scenario months: low={months['lo'].date()}  "
          f"median={months['md'].date()}  high(peak)={months['hi'].date()}")
    _run(p0, "Observed demand (occupied beds)")
    p1, _ = load_icb_problem(stress=1.3)
    _run(p1, "Stress demand x1.3 (wave 30% above observed peak)")
    # Facility-opening MILP: which ICBs to stand up a surge unit, vs a fixed
    # per-unit opening cost. This is the journal's location-allocation core and
    # the test of whether exact MILP stays tractable at 42 nodes.
    print("\n=== Facility-opening MILP (stress x1.3, robust) ===")
    for oc in (0.0, 5.0, 15.0, 30.0):
        s = solve_robust(p1, facility_opening=True, open_cost=oc)
        no = s.extra.get("n_open") if s.extra else None
        print(f"  open_cost={oc:4.0f} beds/ICB: opened={no}/42  surge={s.total_surge_beds:5.0f}  "
              f"E[u]={s.expected_unmet:6.1f}  worst={s.worst_case_unmet:6.1f}  "
              f"t={s.runtime_s:5.2f}s  {s.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
