"""Run allocation experiments E2 (policy comparison) and E3
(metaheuristic comparison), and build the paper figures from their outputs.

Compares all allocation policies on the same regional bed-surge problem
(PinnGRU q10/q50/q90 forecasts, Delta-peak baseline capacity,
inter-region transfers with great-circle distance × 1.3 cost, 20% budget):

    1. Status quo (no surge)
    2. Population-proportional
    3. Demand-proportional
    4. Greedy shortage-first
    5. Deterministic MILP (median scenario only)
    6. Robust MILP (3-scenario expectation + CVaR on q90)
    7. Genetic Algorithm  (E3)
    8. NSGA-II (representative point on Pareto front) (E3)
    9. Simulated Annealing (E3)

Outputs (E2 / E3 experiments, via ``ukci-run-allocation-e2``):
    results/allocation/table2_allocation.csv      one row per policy
    results/allocation/e2_per_region_b.csv        per-region b allocations
    results/allocation/nsga2_pareto.csv           NSGA-II Pareto front

Figures (via ``ukci-build-allocation-figures``):
    figures/fig_allocation_heatmap.png   per-region surge allocation × policy
    figures/fig_nsga2_pareto.png         3-objective Pareto front from NSGA-II
"""

from __future__ import annotations

import sys

from utils import repo_root, results_dir, set_windows_openmp_env

set_windows_openmp_env()

import numpy as np
import pandas as pd

ROOT = repo_root()

from optimization.regional_allocation import (
    DEFAULT_REGION_CODES,
    DEFAULT_REGION_NAMES,
    DEFAULT_HORIZONS,
    demand_proportional,
    greedy_shortage_first,
    load_allocation_problem,
    population_proportional,
    realised_demand_at_origin,
    solve_deterministic,
    solve_ga,
    solve_nsga2,
    solve_robust,
    solve_sa,
    status_quo,
)

OUT_DIR = results_dir("allocation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_LABELS = {
    "status_quo":               "Status quo (no surge)",
    "population_proportional":  "Population-proportional",
    "demand_proportional":      "Demand-proportional",
    "greedy_shortage_first":    "Greedy shortage-first",
    "deterministic_milp":       "Deterministic MILP",
    "robust_milp_cvar1":        "Robust MILP (CVaR, $\\lambda_3{=}1$)",
    "genetic_algorithm":        "Genetic Algorithm",
    "nsga2_repr_point":         "NSGA-II (repr.\\ point)",
    "simulated_annealing":      "Simulated Annealing",
}


def main() -> int:
    p = load_allocation_problem()
    print(f"Allocation problem at origin {p.origin.date()} "
          f"(forecast source: {p.forecast_source})")
    print(f"Regions: {p.regions}")
    print(f"Baseline capacity (peak Delta + 5%): {p.baseline_capacity.round(0)}")
    print(f"Total budget (20% of baseline): {p.budget:.0f} surge beds")
    print(f"Inter-region travel-time cap: {p.max_travel_min:.0f} min")
    print()

    print("Running policies (E2 + E3)...\n")
    solutions = [
        status_quo(p),
        population_proportional(p),
        demand_proportional(p),
        greedy_shortage_first(p),
        solve_deterministic(p),
        solve_robust(p),
        solve_ga(p, pop_size=50, n_gen=40),
        # NSGA-II returns (sol, F, X) — capture the Pareto front below.
        None,
        solve_sa(p, n_iter=400),
    ]
    sol_n, pareto_F, pareto_X = solve_nsga2(p, pop_size=50, n_gen=40)
    solutions[7] = sol_n

    rows = []
    alloc_rows = []
    for sol in solutions:
        label = POLICY_LABELS.get(sol.method, sol.method)
        rows.append({
            "policy": label,
            "method_key": sol.method,
            "Expected unmet": sol.expected_unmet,
            "Worst-case unmet": sol.worst_case_unmet,
            "Transfer burden": sol.transfer_burden,
            "Total surge beds": sol.total_surge_beds,
            "Runtime (s)": sol.runtime_s,
        })
        b_peak = sol.b.max(axis=1)
        rec = {"policy": label}
        for r_name, b_val in zip(p.regions, b_peak):
            rec[r_name] = float(b_val)
        alloc_rows.append(rec)

    df = pd.DataFrame(rows)
    alloc = pd.DataFrame(alloc_rows)

    print("=== Table 2 (allocation comparison) ===")
    print(df.drop(columns=["method_key"]).to_string(
        index=False,
        formatters={
            "Expected unmet":   "{:6.1f}".format,
            "Worst-case unmet": "{:6.1f}".format,
            "Transfer burden":  "{:8.1f}".format,
            "Total surge beds": "{:5.0f}".format,
            "Runtime (s)":      "{:6.2f}".format,
        },
    ))
    print()
    print("=== Per-region surge allocation (b, peak) ===")
    print(alloc.to_string(
        index=False,
        formatters={c: "{:5.1f}".format for c in alloc.columns if c != "policy"},
    ))

    df.to_csv(OUT_DIR / "table2_allocation.csv", index=False)
    alloc.to_csv(OUT_DIR / "e2_per_region_b.csv", index=False)

    # NSGA-II Pareto front to its own file.
    pareto_df = pd.DataFrame({
        "surge_beds": pareto_F[:, 0],
        "expected_unmet": pareto_F[:, 1],
        "transfer_burden": pareto_F[:, 2],
    })
    pareto_df.to_csv(OUT_DIR / "nsga2_pareto.csv", index=False)
    print(f"\nNSGA-II Pareto front: {len(pareto_F)} non-dominated points")

    print(f"Wrote {OUT_DIR / 'table2_allocation.csv'}")
    print(f"Wrote {OUT_DIR / 'nsga2_pareto.csv'}")
    return 0


# ---------------------------------------------------------------------------
# Figures (consume the CSV outputs above)
# ---------------------------------------------------------------------------

# A policy ordering that puts naive baselines first, exact methods next,
# metaheuristics last — for a readable heatmap row order.
HEATMAP_POLICY_ORDER = (
    "Status quo (no surge)",
    "Population-proportional",
    "Demand-proportional",
    "Greedy shortage-first",
    "Deterministic MILP",
    "Robust MILP (CVaR, $\\lambda_3{=}1$)",
    "Genetic Algorithm",
    "NSGA-II (repr.\\ point)",
    "Simulated Annealing",
)


def _figure_allocation_heatmap() -> "Path":
    """Heatmap of peak surge beds by region (rows) × policy (columns)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from evaluation.figures import (
        FULL_WIDTH_IN, apply_paper_style, save_figure,
    )

    apply_paper_style()
    alloc = pd.read_csv(OUT_DIR / "e2_per_region_b.csv")
    alloc = alloc.set_index("policy")
    alloc = alloc.loc[[p for p in HEATMAP_POLICY_ORDER if p in alloc.index]]

    regions = list(alloc.columns)
    values = alloc.to_numpy(dtype=float).T  # (R, P)

    fig, ax = plt.subplots(
        figsize=(FULL_WIDTH_IN, 3.6), layout="constrained",
    )
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(alloc.index)))
    ax.set_xticklabels(
        [_pretty(p) for p in alloc.index], rotation=30, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    ax.set_xlabel("Allocation policy")
    ax.set_ylabel("NHS region")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            colour = "white" if v > values.max() * 0.55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.5, color=colour)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Peak surge beds", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(
        "Per-region peak surge allocation by policy",
        fontsize=10, pad=8,
    )
    return save_figure(fig, "fig_allocation_heatmap", close=True)


def _figure_nsga2_pareto() -> "Path":
    """3-objective NSGA-II Pareto front (surge beds vs unmet vs transfer)."""
    import matplotlib.pyplot as plt
    from evaluation.figures import (
        FULL_WIDTH_IN, apply_paper_style, save_figure,
    )

    apply_paper_style()
    pareto = pd.read_csv(OUT_DIR / "nsga2_pareto.csv")
    table2 = pd.read_csv(OUT_DIR / "table2_allocation.csv")

    repr_row = table2.set_index("method_key").loc["nsga2_repr_point"]
    deterministic = table2.set_index("method_key").loc["deterministic_milp"]
    robust = table2.set_index("method_key").loc["robust_milp_cvar1"]
    ga = table2.set_index("method_key").loc["genetic_algorithm"]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH_IN, 3.6), layout="constrained",
        gridspec_kw={"wspace": 0.18},
    )

    sc = ax_l.scatter(
        pareto["surge_beds"], pareto["expected_unmet"],
        c=pareto["transfer_burden"], cmap="viridis",
        s=22, edgecolor="white", linewidths=0.4,
    )
    cbar = fig.colorbar(sc, ax=ax_l, shrink=0.9, pad=0.02)
    cbar.set_label("Transfer burden (bed·km)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_l.scatter(
        deterministic["Total surge beds"], deterministic["Expected unmet"],
        marker="s", s=70, color="#D55E00", edgecolor="black",
        label="Deterministic MILP", zorder=5,
    )
    ax_l.scatter(
        robust["Total surge beds"], robust["Expected unmet"],
        marker="D", s=70, color="#0072B2", edgecolor="black",
        label="Robust MILP (CVaR)", zorder=5,
    )
    ax_l.scatter(
        ga["Total surge beds"], ga["Expected unmet"],
        marker="^", s=70, color="#009E73", edgecolor="black",
        label="Genetic Algorithm", zorder=5,
    )
    ax_l.scatter(
        repr_row["Total surge beds"], repr_row["Expected unmet"],
        marker="*", s=160, color="#CC79A7", edgecolor="black",
        label="NSGA-II repr.\\ point", zorder=6,
    )
    ax_l.set_xlabel("Total surge beds (budget used)")
    ax_l.set_ylabel("Expected unmet demand")
    ax_l.set_title("Pareto front: surge vs expected unmet", fontsize=10)
    ax_l.legend(frameon=False, fontsize=7, loc="upper right")

    ax_r.scatter(
        pareto["transfer_burden"], pareto["expected_unmet"],
        c=pareto["surge_beds"], cmap="plasma",
        s=22, edgecolor="white", linewidths=0.4,
    )
    cbar2 = fig.colorbar(
        ax_r.collections[0], ax=ax_r, shrink=0.9, pad=0.02,
    )
    cbar2.set_label("Total surge beds", fontsize=8)
    cbar2.ax.tick_params(labelsize=7)
    ax_r.scatter(
        robust["Transfer burden"], robust["Expected unmet"],
        marker="D", s=70, color="#0072B2", edgecolor="black", zorder=5,
    )
    ax_r.scatter(
        ga["Transfer burden"], ga["Expected unmet"],
        marker="^", s=70, color="#009E73", edgecolor="black", zorder=5,
    )
    ax_r.scatter(
        repr_row["Transfer burden"], repr_row["Expected unmet"],
        marker="*", s=160, color="#CC79A7", edgecolor="black", zorder=6,
    )
    ax_r.set_xlabel("Transfer burden (bed·km)")
    ax_r.set_ylabel("Expected unmet demand")
    ax_r.set_title("Pareto front: transfer vs expected unmet", fontsize=10)

    fig.suptitle(
        "NSGA-II 3-objective Pareto front "
        "($f_1$ surge / $f_2$ unmet / $f_3$ transfer)",
        fontsize=10,
    )
    return save_figure(fig, "fig_nsga2_pareto", close=True)


def _pretty(label: str) -> str:
    """Shorten policy labels for the heatmap x-axis."""
    return (label
            .replace("Robust MILP (CVaR, $\\lambda_3{=}1$)", "Robust MILP")
            .replace("NSGA-II (repr.\\ point)", "NSGA-II")
            .replace("Genetic Algorithm", "GA")
            .replace("Simulated Annealing", "SA")
            .replace("Status quo (no surge)", "Status quo"))


# ---------------------------------------------------------------------------
# E5 + E6: forecast-quality robustness + parameter sweeps
# ---------------------------------------------------------------------------


def _evaluate_under_realised(p, b_peak: "np.ndarray", realised) -> dict[str, float]:
    """Given a chosen allocation ``b_peak`` and realised single-scenario demand,
    re-solve the LP slave under the realised path and return ``(realised_unmet,
    realised_transfer, total_surge_beds)``."""
    from dataclasses import replace
    from optimization.regional_allocation import _lp_slave
    p_real = replace(
        p,
        demand=realised,
        scenarios=["realised"],
        scenario_weights=np.array([1.0], dtype=float),
    )
    _, _, eu, tb, wc = _lp_slave(p_real, b_peak)
    return {
        "realised_unmet": eu,
        "realised_transfer_km": tb,
        "realised_worst_case_unmet": wc,
        "total_surge_beds": float(b_peak.sum()),
    }


def _solve_robust_get_b(forecast_model: str, **load_kwargs) -> tuple["np.ndarray", object]:
    p = load_allocation_problem(forecast_model=forecast_model, **load_kwargs)
    sol = solve_robust(p)
    return sol.b.max(axis=1), p


def run_allocation_sweeps_main() -> int:
    """Run E5 (forecast-quality robustness) and E6 (B / λ₃ / τ sensitivity).

    Outputs four CSVs into ``results/allocation/``:

      e5_forecast_robustness.csv    one row per forecaster
      e6_budget_sweep.csv           one row per budget fraction
      e6_lambda_sweep.csv           one row per CVaR weight
      e6_travel_sweep.csv           one row per travel-time cap
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Common origin and realised demand path -------------------------------
    p0 = load_allocation_problem()  # default PinnGRU origin
    forecasts_pq = ROOT / "results" / "forecasting" / "forecasts.parquet"
    realised = realised_demand_at_origin(
        forecasts_pq, DEFAULT_REGION_CODES, DEFAULT_REGION_NAMES,
        origin=p0.origin, horizons=DEFAULT_HORIZONS,
    )
    print(f"Sweep origin: {p0.origin.date()}")
    print(f"Realised peak demand per region: "
          f"{realised.max(axis=1).flatten().round(0)}")

    # -------- E5: forecast-quality robustness ----------------------------
    forecasters = (
        "pinn_gru",
        "arima_per_region",
        "gru_per_region",
        "xgboost_per_region",
        "seasonal_naive",
    )
    print("\n=== E5: forecast-quality robustness ===")
    rows = []
    for fc in forecasters:
        b_peak, p_fc = _solve_robust_get_b(fc)
        metrics = _evaluate_under_realised(p_fc, b_peak, realised)
        rows.append({"forecaster": fc, **metrics})
        print(f"  {fc:25s}  beds={metrics['total_surge_beds']:6.1f}  "
              f"realised unmet={metrics['realised_unmet']:6.1f}  "
              f"realised transfer={metrics['realised_transfer_km']:8.1f}")
    # Oracle: surge MILP under perfect-foresight demand
    from dataclasses import replace
    realised_3s = np.repeat(realised, 3, axis=2)
    p_oracle = replace(
        p0, demand=realised_3s,
        scenarios=["realised", "realised", "realised"],
        scenario_weights=np.array([0.2, 0.6, 0.2], dtype=float),
    )
    b_oracle = solve_robust(p_oracle).b.max(axis=1)
    metrics_oracle = _evaluate_under_realised(p0, b_oracle, realised)
    rows.append({"forecaster": "oracle (y_true)", **metrics_oracle})
    print(f"  {'oracle (y_true)':25s}  beds={metrics_oracle['total_surge_beds']:6.1f}  "
          f"realised unmet={metrics_oracle['realised_unmet']:6.1f}  "
          f"realised transfer={metrics_oracle['realised_transfer_km']:8.1f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "e5_forecast_robustness.csv", index=False)

    # -------- E6a: budget sweep ------------------------------------------
    print("\n=== E6a: surge-budget sweep ===")
    rows = []
    for frac in (0.10, 0.15, 0.20, 0.25, 0.30):
        p = load_allocation_problem(budget_fraction=frac)
        sol = solve_robust(p)
        rows.append({
            "budget_fraction": frac,
            "budget_beds": float(p.budget),
            "expected_unmet": sol.expected_unmet,
            "worst_case_unmet": sol.worst_case_unmet,
            "transfer_burden": sol.transfer_burden,
            "total_surge_beds": sol.total_surge_beds,
        })
        print(f"  B/Cbase={frac:.2f} -> beds={sol.total_surge_beds:6.1f}  "
              f"E[u]={sol.expected_unmet:6.1f}  WC={sol.worst_case_unmet:6.1f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "e6_budget_sweep.csv", index=False)

    # -------- E6b: CVaR-weight sweep -------------------------------------
    print("\n=== E6b: CVaR-weight sweep ===")
    p = load_allocation_problem()
    rows = []
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0):
        sol = solve_robust(p, cvar_lambda=lam)
        rows.append({
            "lambda_3": lam,
            "expected_unmet": sol.expected_unmet,
            "worst_case_unmet": sol.worst_case_unmet,
            "transfer_burden": sol.transfer_burden,
            "total_surge_beds": sol.total_surge_beds,
        })
        print(f"  lambda3={lam:.2f} -> beds={sol.total_surge_beds:6.1f}  "
              f"E[u]={sol.expected_unmet:6.1f}  WC={sol.worst_case_unmet:6.1f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "e6_lambda_sweep.csv", index=False)

    # -------- E6c: travel-time cap sweep ---------------------------------
    print("\n=== E6c: travel-time cap sweep ===")
    rows = []
    for tau in (120, 180, 240, 300, 360):
        p = load_allocation_problem(max_travel_min=float(tau))
        sol = solve_robust(p)
        rows.append({
            "tau_min": tau,
            "expected_unmet": sol.expected_unmet,
            "worst_case_unmet": sol.worst_case_unmet,
            "transfer_burden": sol.transfer_burden,
            "total_surge_beds": sol.total_surge_beds,
        })
        print(f"  tau={tau:>3d}min -> beds={sol.total_surge_beds:6.1f}  "
              f"E[u]={sol.expected_unmet:6.1f}  WC={sol.worst_case_unmet:6.1f}  "
              f"transfer={sol.transfer_burden:8.1f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "e6_travel_sweep.csv", index=False)

    print(f"\nWrote {OUT_DIR / 'e5_forecast_robustness.csv'}")
    print(f"Wrote {OUT_DIR / 'e6_budget_sweep.csv'}")
    print(f"Wrote {OUT_DIR / 'e6_lambda_sweep.csv'}")
    print(f"Wrote {OUT_DIR / 'e6_travel_sweep.csv'}")
    return 0


def build_allocation_figures_main() -> int:
    """Build the two allocation figures from the saved E2/E3 CSVs."""
    required = [
        OUT_DIR / "table2_allocation.csv",
        OUT_DIR / "e2_per_region_b.csv",
        OUT_DIR / "nsga2_pareto.csv",
    ]
    for path in required:
        if not path.exists():
            print(f"Missing input: {path.relative_to(ROOT)}", file=sys.stderr)
            print("Run ukci-run-allocation-e2 first.", file=sys.stderr)
            return 1
    out_heatmap = _figure_allocation_heatmap()
    out_pareto = _figure_nsga2_pareto()
    print(f"Wrote {out_heatmap}")
    print(f"Wrote {out_pareto}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
