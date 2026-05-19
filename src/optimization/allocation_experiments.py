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
from dataclasses import replace

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

REVISION_FORECASTERS = (
    "pinn_gru",
    "arima_per_region",
    "gru_per_region",
    "xgboost_per_region",
    "seasonal_naive",
)

REVISION_BUDGET_FRACTIONS = (0.10, 0.15, 0.20)


def _solution_row(sol, *, origin=None, budget_fraction: float | None = None) -> dict:
    row = {
        "policy": POLICY_LABELS.get(sol.method, sol.method),
        "method_key": sol.method,
        "Expected unmet": sol.expected_unmet,
        "Worst-case unmet": sol.worst_case_unmet,
        "Transfer burden": sol.transfer_burden,
        "Total surge beds": sol.total_surge_beds,
        "Runtime (s)": sol.runtime_s,
    }
    if origin is not None:
        row["origin"] = pd.Timestamp(origin).date().isoformat()
    if budget_fraction is not None:
        row["budget_fraction"] = budget_fraction
    return row


def _cheap_policy_solutions(p):
    """Policies cheap enough to repeat over every rolling origin."""
    return [
        status_quo(p),
        population_proportional(p),
        demand_proportional(p),
        solve_deterministic(p),
        solve_robust(p),
    ]


def _table2_policy_solutions(p):
    """Exact and heuristic policies for tighter-budget manuscript panels."""
    return [
        status_quo(p),
        population_proportional(p),
        demand_proportional(p),
        greedy_shortage_first(p),
        solve_deterministic(p),
        solve_robust(p),
    ]


def _full_coverage_origins(forecast_model: str = "pinn_gru") -> list[pd.Timestamp]:
    forecasts_pq = ROOT / "results" / "forecasting" / "forecasts.parquet"
    fc = pd.read_parquet(forecasts_pq)
    fc = fc[fc["model"] == forecast_model]
    origins: list[pd.Timestamp] = []
    for origin, sub in fc.groupby("origin"):
        if sub["region"].nunique() == len(DEFAULT_REGION_NAMES) and \
           set(sub["horizon"].unique()) >= set(DEFAULT_HORIZONS):
            origins.append(pd.Timestamp(origin))
    return sorted(origins)


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
        # Metaheuristics are not reported at seven-region scale: the LP is
        # exact in <0.1 s, so GA/NSGA-II/SA add no tractability benefit.
        # Code retained (commented) for the planned trust-level journal
        # extension; uncomment this block and the Pareto export below.
        # solve_ga(p, pop_size=50, n_gen=40),
        # None,  # placeholder for NSGA-II (returns (sol, F, X))
        # solve_sa(p, n_iter=400),
    ]
    # sol_n, pareto_F, pareto_X = solve_nsga2(p, pop_size=50, n_gen=40)
    # solutions[7] = sol_n

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

    # NSGA-II Pareto export, commented with the metaheuristics above.
    # Uncomment together for the trust-level extension.
    # pareto_df = pd.DataFrame({
    #     "surge_beds": pareto_F[:, 0],
    #     "expected_unmet": pareto_F[:, 1],
    #     "transfer_burden": pareto_F[:, 2],
    # })
    # pareto_df.to_csv(OUT_DIR / "nsga2_pareto.csv", index=False)
    # print(f"\nNSGA-II Pareto front: {len(pareto_F)} non-dominated points")

    print(f"Wrote {OUT_DIR / 'table2_allocation.csv'}")
    print(f"Wrote {OUT_DIR / 'e2_per_region_b.csv'}")
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

# Policies shown in the UKCI paper. The metaheuristics are retained in the
# code (HEATMAP_POLICY_ORDER) for the planned trust-level journal extension
# but are not reported at seven-region scale, where the LP is exact.
PAPER_POLICY_ORDER = (
    "Status quo (no surge)",
    "Population-proportional",
    "Demand-proportional",
    "Greedy shortage-first",
    "Deterministic MILP",
    "Robust MILP (CVaR, $\\lambda_3{=}1$)",
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
    alloc = alloc.loc[[p for p in PAPER_POLICY_ORDER if p in alloc.index]]

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


def _figure_alloc_budget() -> "Path":
    """Single two-panel paper figure (one float, fits the 12-page cap):
    (a) per-region peak surge by policy; (b) exact robust-LP cost-shortage
    frontier vs the surge budget."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from evaluation.figures import (
        FULL_WIDTH_IN, apply_paper_style, save_figure,
    )

    apply_paper_style()
    alloc = pd.read_csv(OUT_DIR / "e2_per_region_b.csv").set_index("policy")
    alloc = alloc.loc[[p for p in PAPER_POLICY_ORDER if p in alloc.index]]
    regions = list(alloc.columns)
    values = alloc.to_numpy(dtype=float).T  # (R, P)

    sweep = pd.read_csv(OUT_DIR / "e6_budget_sweep.csv")
    sweep = sweep.sort_values("budget_fraction")
    xb = sweep["budget_fraction"] * 100.0

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 3.4), layout="constrained")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.28, 1.0], wspace=0.05)
    axh = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    im = axh.imshow(values, aspect="auto", cmap="YlGnBu")
    axh.set_xticks(range(len(alloc.index)))
    axh.set_xticklabels([_pretty(p) for p in alloc.index],
                        rotation=35, ha="right", fontsize=7)
    axh.set_yticks(range(len(regions)))
    axh.set_yticklabels(regions, fontsize=7)
    axh.set_xlabel("Allocation policy", fontsize=8)
    axh.set_ylabel("NHS region", fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            axh.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                     color="white" if v > values.max() * 0.55 else "black")
    cbar = fig.colorbar(im, ax=axh, shrink=0.82, pad=0.02)
    cbar.set_label("Peak surge beds", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    axh.set_title("(a) Per-region peak surge allocation", fontsize=8)

    axb.plot(xb, sweep["expected_unmet"], marker="o", ms=4, color="#0072B2",
             label=r"$E[u]$")
    axb.plot(xb, sweep["worst_case_unmet"], marker="s", ms=4, linestyle="--",
             color="#D55E00", label=r"$u^{\mathrm{worst}}$")
    axb.axvline(20.0, color="0.65", linewidth=0.8, linestyle=":")
    axb.set_xlabel(r"Surge budget $B/\sum_r C_r$ (%)", fontsize=8)
    axb.set_ylabel("Unmet demand (beds)", fontsize=8)
    axb.set_title("(b) Exact cost-shortage frontier", fontsize=8)
    axb.legend(frameon=False, fontsize=7)
    axb.grid(True, alpha=0.25)
    axb.tick_params(labelsize=7)

    return save_figure(fig, "fig_alloc_budget", close=True)


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


def _figure_budget_tradeoff() -> "Path":
    """Exact cost-shortage frontier from the robust-LP budget sweep:
    expected and worst-case unmet demand against the surge budget."""
    import matplotlib.pyplot as plt
    from evaluation.figures import (
        FULL_WIDTH_IN, apply_paper_style, save_figure,
    )

    apply_paper_style()
    sweep = pd.read_csv(OUT_DIR / "e6_budget_sweep.csv")
    sweep = sweep.sort_values("budget_fraction")
    x = sweep["budget_fraction"] * 100.0

    fig, ax = plt.subplots(
        figsize=(FULL_WIDTH_IN * 0.60, 3.1), layout="constrained",
    )
    ax.plot(x, sweep["expected_unmet"], marker="o", color="#0072B2",
            label=r"Expected unmet $E[u]$")
    ax.plot(x, sweep["worst_case_unmet"], marker="s", linestyle="--",
            color="#D55E00", label=r"Worst-case unmet $u^{\mathrm{worst}}$")
    ax.axvline(20.0, color="0.65", linewidth=0.8, linestyle=":")
    ax.text(20.4, ax.get_ylim()[1] * 0.88, "operating point",
            fontsize=7, color="0.4")
    ax.set_xlabel(r"Surge budget $B/\sum_r C_r$ (%)")
    ax.set_ylabel("Unmet demand (beds)")
    ax.set_title("Exact cost-shortage frontier (robust LP)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, "fig_budget_tradeoff", close=True)


def _pretty(label: str) -> str:
    """Shorten policy labels for the heatmap x-axis. The regional model has
    no integer variables, so the exact methods are labelled LP, matching the
    paper."""
    return (label
            .replace("Robust MILP (CVaR, $\\lambda_3{=}1$)", "Robust LP")
            .replace("Deterministic MILP", "Deterministic LP")
            .replace("NSGA-II (repr.\\ point)", "NSGA-II")
            .replace("Genetic Algorithm", "GA")
            .replace("Simulated Annealing", "SA")
            .replace("Status quo (no surge)", "Status quo"))


# ---------------------------------------------------------------------------
# E5 + E6: forecast-quality robustness + parameter sweeps
# ---------------------------------------------------------------------------


def _evaluate_under_realised(p, b_peak: "np.ndarray", realised) -> dict[str, float]:
    """Given a chosen allocation ``b_peak`` and a realised single-scenario
    demand path ``realised`` of shape ``(R, H, 1)``, re-solve the LP slave
    against the realised path and return the resulting metrics."""
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
    return sol.b.max(axis=1), p, sol


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
    #
    # Each forecaster's q^{0.9} (or its point prediction, for non-quantile
    # baselines) drives the robust MILP. We report (a) the chosen surge
    # investment (beds), (b) the forecast peak that drove it, (c) the
    # realised peak demand at the same origin, (d) the over- or under-
    # provisioning gap, and (e) the realised unmet under the chosen
    # allocation. At the Delta-peak baseline used here the budget is
    # operationally generous, so realised unmet is typically zero; the
    # paper-meaningful signal is in the forecast-peak / realised-peak gap
    # — a forecaster that over-states peaks spends the budget unnecessarily.
    forecasters = (
        "pinn_gru",
        "arima_per_region",
        "gru_per_region",
        "xgboost_per_region",
        "seasonal_naive",
    )
    print("\n=== E5: forecast-quality robustness ===")
    rows = []
    real_peak_per_region = realised.max(axis=1).flatten()  # (R,)
    real_peak_total = float(real_peak_per_region.sum())
    for fc in forecasters:
        b_peak, p_fc, sol_fc = _solve_robust_get_b(fc)
        forecast_peak_per_region = p_fc.demand[:, :, 2].max(axis=1)  # high scenario peak
        forecast_peak_total = float(forecast_peak_per_region.sum())
        metrics = _evaluate_under_realised(p_fc, b_peak, realised)
        rows.append({
            "forecaster": fc,
            "forecast_peak_total": forecast_peak_total,
            "realised_peak_total": real_peak_total,
            "over_provision_beds": forecast_peak_total - real_peak_total,
            "expected_unmet_at_solve": sol_fc.expected_unmet,
            **metrics,
        })
        print(f"  {fc:25s}  beds={metrics['total_surge_beds']:6.1f}  "
              f"q90 peak total={forecast_peak_total:6.0f}  "
              f"realised peak={real_peak_total:6.0f}  "
              f"realised unmet={metrics['realised_unmet']:5.1f}")
    # Oracle: surge MILP under perfect-foresight demand
    from dataclasses import replace
    realised_3s = np.repeat(realised, 3, axis=2)
    p_oracle = replace(
        p0, demand=realised_3s,
        scenarios=["low", "median", "high"],          # keep canonical labels
        scenario_weights=np.array([0.2, 0.6, 0.2], dtype=float),
    )
    b_oracle = solve_robust(p_oracle, cvar_lambda=0.0).b.max(axis=1)
    metrics_oracle = _evaluate_under_realised(p0, b_oracle, realised)
    rows.append({
        "forecaster": "oracle (y_true)",
        "forecast_peak_total": real_peak_total,
        "realised_peak_total": real_peak_total,
        "over_provision_beds": 0.0,
        "expected_unmet_at_solve": metrics_oracle["realised_unmet"],
        **metrics_oracle,
    })
    print(f"  {'oracle (y_true)':25s}  beds={metrics_oracle['total_surge_beds']:6.1f}  "
          f"q90 peak total={real_peak_total:6.0f}  realised peak={real_peak_total:6.0f}  "
          f"realised unmet={metrics_oracle['realised_unmet']:5.1f}")
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


def build_all_origin_policy_distribution() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate cheap allocation policies at every full-coverage test origin."""
    origins = _full_coverage_origins("pinn_gru")
    if not origins:
        raise RuntimeError("No full-coverage PinnGRU origins found.")

    rows = []
    for origin in origins:
        p = load_allocation_problem(origin=origin)
        for sol in _cheap_policy_solutions(p):
            rows.append(_solution_row(sol, origin=origin))

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["policy", "method_key"])
        .agg(
            n_origins=("origin", "nunique"),
            expected_unmet_mean=("Expected unmet", "mean"),
            expected_unmet_p10=("Expected unmet", lambda x: x.quantile(0.10)),
            expected_unmet_p90=("Expected unmet", lambda x: x.quantile(0.90)),
            worst_case_unmet_mean=("Worst-case unmet", "mean"),
            worst_case_unmet_p90=("Worst-case unmet", lambda x: x.quantile(0.90)),
            transfer_mean=("Transfer burden", "mean"),
            transfer_p90=("Transfer burden", lambda x: x.quantile(0.90)),
            total_surge_mean=("Total surge beds", "mean"),
            runtime_mean_s=("Runtime (s)", "mean"),
        )
        .reset_index()
    )
    detail.to_csv(OUT_DIR / "e7_origin_policy_detail.csv", index=False)
    summary.to_csv(OUT_DIR / "e7_origin_policy_summary.csv", index=False)
    return detail, summary


def build_tighter_budget_policy_tables() -> pd.DataFrame:
    """Run Table-2-style exact and heuristic policies at tighter budgets."""
    rows = []
    for frac in REVISION_BUDGET_FRACTIONS:
        p = load_allocation_problem(budget_fraction=frac)
        for sol in _table2_policy_solutions(p):
            rows.append(_solution_row(sol, origin=p.origin, budget_fraction=frac))
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "e8_budget_policy_comparison.csv", index=False)
    return table


def build_stress_forecast_robustness() -> pd.DataFrame:
    """Re-evaluate forecaster-driven robust allocations under scaled realised demand."""
    p0 = load_allocation_problem()
    forecasts_pq = ROOT / "results" / "forecasting" / "forecasts.parquet"
    realised = realised_demand_at_origin(
        forecasts_pq, DEFAULT_REGION_CODES, DEFAULT_REGION_NAMES,
        origin=p0.origin, horizons=DEFAULT_HORIZONS,
    )

    rows = []
    for scale in (1.0, 1.2, 1.3):
        scaled_realised = realised * scale
        scaled_peak_total = float(scaled_realised.max(axis=1).sum())
        for forecaster in REVISION_FORECASTERS:
            b_peak, p_fc, sol_fc = _solve_robust_get_b(forecaster, origin=p0.origin)
            metrics = _evaluate_under_realised(p_fc, b_peak, scaled_realised)
            forecast_peak_total = float(p_fc.demand[:, :, 2].max(axis=1).sum())
            rows.append({
                "origin": pd.Timestamp(p0.origin).date().isoformat(),
                "forecaster": forecaster,
                "realised_scale": scale,
                "forecast_peak_total": forecast_peak_total,
                "scaled_realised_peak_total": scaled_peak_total,
                "expected_unmet_at_solve": sol_fc.expected_unmet,
                **metrics,
            })

        realised_3s = np.repeat(scaled_realised, 3, axis=2)
        p_oracle = replace(
            p0,
            demand=realised_3s,
            scenarios=["low", "median", "high"],
            scenario_weights=np.array([0.2, 0.6, 0.2], dtype=float),
        )
        b_oracle = solve_robust(p_oracle, cvar_lambda=0.0).b.max(axis=1)
        metrics_oracle = _evaluate_under_realised(p0, b_oracle, scaled_realised)
        rows.append({
            "origin": pd.Timestamp(p0.origin).date().isoformat(),
            "forecaster": "oracle (scaled realised)",
            "realised_scale": scale,
            "forecast_peak_total": scaled_peak_total,
            "scaled_realised_peak_total": scaled_peak_total,
            "expected_unmet_at_solve": metrics_oracle["realised_unmet"],
            **metrics_oracle,
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "e5_stress_forecast_robustness.csv", index=False)
    return table


def run_allocation_revision_main() -> int:
    """Run compact revision analyses requested by the manuscript review."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== E7: all-origin exact/closed-form policy distribution ===")
    detail, summary = build_all_origin_policy_distribution()
    print(f"Wrote {OUT_DIR / 'e7_origin_policy_detail.csv'} ({len(detail):,} rows)")
    print(f"Wrote {OUT_DIR / 'e7_origin_policy_summary.csv'} ({len(summary):,} rows)")
    print(summary.round(2).to_string(index=False))

    print("\n=== E8: tighter-budget policy comparisons ===")
    budget_table = build_tighter_budget_policy_tables()
    print(f"Wrote {OUT_DIR / 'e8_budget_policy_comparison.csv'} ({len(budget_table):,} rows)")

    print("\n=== E5 stress: scaled realised demand ===")
    stress = build_stress_forecast_robustness()
    print(f"Wrote {OUT_DIR / 'e5_stress_forecast_robustness.csv'} ({len(stress):,} rows)")
    print(stress.round(2).to_string(index=False))
    return 0


def build_allocation_figures_main() -> int:
    """Build the single combined paper allocation figure from the saved
    CSVs: (a) per-region surge heatmap and (b) the exact budget
    cost-shortage frontier, merged into one float to fit the 12-page cap.
    The standalone heatmap/budget and NSGA-II Pareto figures are retained
    (commented) for the trust-level journal extension."""
    required = [
        OUT_DIR / "e2_per_region_b.csv",
        OUT_DIR / "e6_budget_sweep.csv",
    ]
    for path in required:
        if not path.exists():
            print(f"Missing input: {path.relative_to(ROOT)}", file=sys.stderr)
            print("Run ukci-run-allocation-e2 / sweeps first.", file=sys.stderr)
            return 1
    out_fig = _figure_alloc_budget()
    # out_heatmap = _figure_allocation_heatmap()   # trust-level extension
    # out_budget = _figure_budget_tradeoff()       # trust-level extension
    # out_pareto = _figure_nsga2_pareto()          # trust-level extension
    print(f"Wrote {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
