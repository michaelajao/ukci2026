"""Run the E2 policy comparison and build allocation figures from saved outputs.

The main experiment compares six policies on the same regional bed-surge problem
(PinnGRU q10/q50/q90 forecasts, Delta-peak baseline capacity,
inter-region transfers with great-circle distance × 1.3 cost, 20% budget):

    1. Status quo (no surge)
    2. Population-proportional
    3. Demand-proportional
    4. Greedy shortage-first
    5. Deterministic LP (median scenario only)
    6. Risk-averse LP (scenario-weighted objective + q90 tail penalty)

Outputs (via ``ukci-run-allocation-e2``):
    results/allocation/table2_allocation.csv      one row per policy
    results/allocation/e2_per_region_b.csv        per-region b allocations

Figures (via ``ukci-build-allocation-figures``):
    figures/fig_alloc_budget.png         allocation heatmap + budget frontier

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
    solve_robust,
    status_quo,
)

OUT_DIR = results_dir("allocation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_LABELS = {
    "status_quo":               "Status quo (no surge)",
    "population_proportional":  "Population-proportional",
    "demand_proportional":      "Demand-proportional",
    "greedy_shortage_first":    "Greedy shortage-first",
    "deterministic_milp":       "Deterministic LP",
    "robust_milp_cvar1":        "Risk-averse LP ($\\lambda_3{=}1$)",
}

REVISION_FORECASTERS = (
    "pinn_gru",
    "arima_per_region",
    "gru_per_region",
    "xgboost_per_region",
    "seasonal_naive",
)

REVISION_BUDGET_FRACTIONS = (0.10, 0.15, 0.20)

# Dense (4 h) vs sparse (2 h) mutual-aid networks for the E9 connectivity
# comparison: at 240 min the 7 regions form 12 undirected links, at 120 min
# only 5, so surge placement stops being fungible.
REVISION_TAU_MINUTES = (240.0, 120.0)

# E10: per-region outbound transfer caps (displaced patients sustained at a
# checkpoint — regional critical-care transfer-service capacity). None is
# the uncapped headline operating point.
REVISION_TRANSFER_CAPS = (None, 20.0, 10.0, 5.0)


def _solution_row(sol, *, origin=None, budget_fraction: float | None = None) -> dict:
    row = {
        "policy": POLICY_LABELS.get(sol.method, sol.method),
        "method_key": sol.method,
        "Scenario-weighted unmet": sol.expected_unmet,
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

    print("Running policies (E2)...\n")
    solutions = [
        status_quo(p),
        population_proportional(p),
        demand_proportional(p),
        greedy_shortage_first(p),
        solve_deterministic(p),
        solve_robust(p),
    ]

    rows = []
    alloc_rows = []
    for sol in solutions:
        label = POLICY_LABELS.get(sol.method, sol.method)
        rows.append({
            "policy": label,
            "method_key": sol.method,
            "Scenario-weighted unmet": sol.expected_unmet,
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

    print("=== Allocation comparison ===")
    print(df.drop(columns=["method_key"]).to_string(
        index=False,
        formatters={
            "Scenario-weighted unmet": "{:6.1f}".format,
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

    print(f"Wrote {OUT_DIR / 'table2_allocation.csv'}")
    print(f"Wrote {OUT_DIR / 'e2_per_region_b.csv'}")
    return 0


# ---------------------------------------------------------------------------
# Figures (consume the CSV outputs above)
# ---------------------------------------------------------------------------

# A policy ordering that puts naive baselines first, exact methods next —
# for a readable heatmap row order.
HEATMAP_POLICY_ORDER = (
    "Status quo (no surge)",
    "Population-proportional",
    "Demand-proportional",
    "Greedy shortage-first",
    "Deterministic LP",
    "Risk-averse LP ($\\lambda_3{=}1$)",
)

# Policies reported at seven-region scale, where the LP is exact.
PAPER_POLICY_ORDER = (
    "Status quo (no surge)",
    "Population-proportional",
    "Demand-proportional",
    "Greedy shortage-first",
    "Deterministic LP",
    "Risk-averse LP ($\\lambda_3{=}1$)",
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
    """Two-panel paper figure (one float, fits the 12-page cap):
    (a) per-region peak surge by policy; (b) exact risk-averse-LP cost-shortage
    frontier vs the surge budget."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from evaluation.figures import apply_paper_style, save_figure

    apply_paper_style()
    alloc = pd.read_csv(OUT_DIR / "e2_per_region_b.csv").set_index("policy")
    # Status quo is omitted: its allocation is identically zero by
    # definition, and it appears in neither Table 2 nor panel (c).
    alloc = alloc.loc[[p for p in PAPER_POLICY_ORDER
                       if p in alloc.index and p != "Status quo (no surge)"]]
    regions = list(alloc.columns)
    values = alloc.to_numpy(dtype=float).T  # (R, P)

    sweep = pd.read_csv(OUT_DIR / "e6_budget_sweep.csv")
    sweep = sweep.sort_values("budget_fraction")
    xb = sweep["budget_fraction"] * 100.0

    # Side-by-side panels keep Figure 3 compact while preserving the two
    # decision messages: where beds go and how shortage falls with budget.
    fig = plt.figure(figsize=(7.2, 2.55), layout="constrained")
    # Under constrained layout, GridSpec wspace/hspace are ignored; spacing
    # between panels must be set on the layout engine.
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.06,
                                wspace=0.10, hspace=0.12)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0])
    axh = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # Heatmap: every cell is annotated with its value, so a colorbar adds
    # nothing and (squeezed between panels) collided with panel (b).
    short = {
        "Status quo (no surge)": "Status quo",
        "Population-proportional": "Pop.",
        "Demand-proportional": "Demand",
        "Greedy shortage-first": "Greedy",
        "Deterministic MILP": "Det. LP",
        "Risk-averse LP ($\\lambda_3{=}1$)": "Risk-averse LP",
    }
    region_short = {
        "East of England": "East Eng.",
        "North East and Yorkshire": "NE & Yorks",
    }
    axh.imshow(values, aspect="auto", cmap="YlGnBu")
    axh.set_xticks(range(len(alloc.index)))
    axh.set_xticklabels([short.get(p, _pretty(p)) for p in alloc.index],
                        fontsize=9, rotation=20, ha="right")
    axh.set_yticks(range(len(regions)))
    axh.set_yticklabels([region_short.get(r, r) for r in regions], fontsize=9)
    axh.set_ylabel("NHS region", fontsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isclose(v, 0.0, atol=0.05):
                annotation = "0"
            elif abs(v) < 1.0:
                annotation = f"{v:.1f}"
            else:
                annotation = f"{v:.0f}"
            axh.text(j, i, annotation, ha="center", va="center", fontsize=9,
                     color="white" if v > values.max() * 0.55 else "black")
    axh.set_title("Additional surge beds by policy", fontsize=11, pad=3)

    axb.plot(xb, sweep["expected_unmet"], marker="o", ms=4.2, color="#0072B2",
             label=r"$U_\pi$")
    axb.plot(xb, sweep["worst_case_unmet"], marker="s", ms=4.2, linestyle="--",
             color="#D55E00", label=r"$u^{\mathrm{worst}}$")
    axb.axvline(20.0, color="0.65", linewidth=0.8, linestyle=":")
    axb.set_xlabel("Surge budget (% of baseline)", fontsize=10, labelpad=2)
    axb.set_ylabel("Unmet (bed-checkpoints)", fontsize=10)
    axb.set_title("Cost-shortage frontier", fontsize=11, pad=3)
    axb.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    axb.grid(True, alpha=0.25)
    axb.tick_params(labelsize=9)

    return save_figure(fig, "fig_alloc_budget", close=True)


def _figure_budget_tradeoff() -> "Path":
    """Exact cost-shortage frontier from the risk-averse-LP budget sweep:
    scenario-weighted and worst-case unmet demand against the surge budget."""
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
            label=r"Scenario-weighted unmet $U_\pi$")
    ax.plot(x, sweep["worst_case_unmet"], marker="s", linestyle="--",
            color="#D55E00", label=r"Worst-case unmet $u^{\mathrm{worst}}$")
    ax.axvline(20.0, color="0.65", linewidth=0.8, linestyle=":")
    ax.text(20.4, ax.get_ylim()[1] * 0.88, "operating point",
            fontsize=7, color="0.4")
    ax.set_xlabel(r"Surge budget $B/\sum_r C_r$ (%)")
    ax.set_ylabel("Unmet demand (beds)")
    ax.set_title("Exact cost-shortage frontier (risk-averse LP)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, "fig_budget_tradeoff", close=True)


def _pretty(label: str) -> str:
    """Shorten policy labels for the heatmap x-axis. The regional model has
    no integer variables, so the exact methods are labelled LP, matching the
    paper."""
    return (label
            .replace("Risk-averse LP ($\\lambda_3{=}1$)", "Risk-averse LP")
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
      e6_lambda_sweep.csv           one row per high-scenario tail weight
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
    # baselines) drives the risk-averse LP. We report (a) the chosen surge
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
              f"U_pi={sol.expected_unmet:6.1f}  WC={sol.worst_case_unmet:6.1f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "e6_budget_sweep.csv", index=False)

    # -------- E6b: high-scenario tail-weight sweep ------------------------
    print("\n=== E6b: high-scenario tail-weight sweep ===")
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
              f"U_pi={sol.expected_unmet:6.1f}  WC={sol.worst_case_unmet:6.1f}")
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
              f"U_pi={sol.expected_unmet:6.1f}  WC={sol.worst_case_unmet:6.1f}  "
              f"transfer={sol.transfer_burden:8.1f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "e6_travel_sweep.csv", index=False)

    print(f"\nWrote {OUT_DIR / 'e5_forecast_robustness.csv'}")
    print(f"Wrote {OUT_DIR / 'e6_budget_sweep.csv'}")
    print(f"Wrote {OUT_DIR / 'e6_lambda_sweep.csv'}")
    print(f"Wrote {OUT_DIR / 'e6_travel_sweep.csv'}")
    return 0


def build_all_origin_policy_distribution() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the manuscript allocation policies at every full-coverage test
    origin. Uses the Table-2 policy set (adds greedy shortage-first to the
    cheap policies) so the all-origin panel reports the same policies as the
    single peak-origin panel; greedy is the slow one (~10 s/origin)."""
    origins = _full_coverage_origins("pinn_gru")
    if not origins:
        raise RuntimeError("No full-coverage PinnGRU origins found.")

    rows = []
    for origin in origins:
        p = load_allocation_problem(origin=origin)
        for sol in _table2_policy_solutions(p):
            rows.append(_solution_row(sol, origin=origin))

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["policy", "method_key"])
        .agg(
            n_origins=("origin", "nunique"),
            expected_unmet_mean=("Scenario-weighted unmet", "mean"),
            expected_unmet_p10=("Scenario-weighted unmet", lambda x: x.quantile(0.10)),
            expected_unmet_p90=("Scenario-weighted unmet", lambda x: x.quantile(0.90)),
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


def build_tau_policy_comparison() -> pd.DataFrame:
    """Run Table-2-style policies under dense (240 min) and sparse (120 min)
    mutual-aid networks across the revision budgets — the E9 check of
    whether surge placement matters once the transfer network thins."""
    rows = []
    for tau in REVISION_TAU_MINUTES:
        for frac in REVISION_BUDGET_FRACTIONS:
            p = load_allocation_problem(
                max_travel_min=float(tau), budget_fraction=frac,
            )
            for sol in _table2_policy_solutions(p):
                row = _solution_row(sol, origin=p.origin, budget_fraction=frac)
                row["tau_min"] = float(tau)
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "e9_tau_policy_comparison.csv", index=False)
    return table


def build_transfer_cap_policy_comparison() -> pd.DataFrame:
    """Run Table-2-style policies under per-region outbound transfer caps
    at the headline operating point (tau=240, B=0.20) — the E10 check of
    whether limited transfer-service capacity breaks placement fungibility
    even on the dense network."""
    rows = []
    for cap in REVISION_TRANSFER_CAPS:
        p = load_allocation_problem(max_transfer_out=cap)
        for sol in _table2_policy_solutions(p):
            row = _solution_row(sol, origin=p.origin, budget_fraction=0.20)
            row["transfer_cap"] = np.nan if cap is None else float(cap)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "e10_transfer_cap_policy_comparison.csv", index=False)
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

    print("\n=== E9: dense vs sparse mutual-aid network policy comparison ===")
    tau_table = build_tau_policy_comparison()
    print(f"Wrote {OUT_DIR / 'e9_tau_policy_comparison.csv'} ({len(tau_table):,} rows)")

    print("\n=== E10: outbound transfer-cap policy comparison ===")
    cap_table = build_transfer_cap_policy_comparison()
    print(f"Wrote {OUT_DIR / 'e10_transfer_cap_policy_comparison.csv'} ({len(cap_table):,} rows)")

    print("\n=== E5 stress: scaled realised demand ===")
    stress = build_stress_forecast_robustness()
    print(f"Wrote {OUT_DIR / 'e5_stress_forecast_robustness.csv'} ({len(stress):,} rows)")
    print(stress.round(2).to_string(index=False))
    return 0


def build_allocation_figures_main() -> int:
    """Build the single combined allocation figure from the saved CSVs:
    (a) per-region surge heatmap and (b) the exact budget cost-shortage
    frontier, merged into one float."""
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
    print(f"Wrote {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
