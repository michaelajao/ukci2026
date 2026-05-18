"""Re-run PINN pre-training only and report final pinn_data / pinn_ode values.

Run from the repo root:
    .\.venv\Scripts\Activate.ps1
    python scripts/check_pinn_convergence.py

Prints one line per region with the actual final values that the paper's
Section 3.2 currently claims are "O(10^-6)". Output is also written to
results/forecasting/pinn_convergence_check.csv for the paper.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from forecasting.pinn_seird import RegionalPINN
from forecasting.train_forecasters import (
    pretrain_pinn_region,
    DEFAULT_SPLIT_DATES,
    DEVICE,
)
from utils import repo_root

TARGET = "mv_beds"
COVAR_H = "hospital_cases"


def main() -> int:
    root = repo_root()
    df = pd.read_csv(
        root / "data" / "processed" / "regional_daily.csv",
        parse_dates=["date"],
    )
    static = pd.read_csv(root / "data" / "processed" / "regional_static.csv")
    pop_by_code = dict(zip(static["region_code"], static["population"]))

    test_start = pd.Timestamp(DEFAULT_SPLIT_DATES["test"][0])
    df_tv = df[df["date"] < test_start].copy()

    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    n_total = len(all_dates)
    date_to_norm = {d: (i / max(n_total - 1, 1)) for i, d in enumerate(all_dates)}

    print(f"\nDevice: {DEVICE}")
    print(f"{'Region':30s}  N (M)   pinn_data   pinn_ode   total")
    print("-" * 75)

    rows = []
    for region_name, group in df_tv.groupby("region_name", observed=True):
        g = group.sort_values("date")
        code = g["region_code"].iloc[0]
        pop = pop_by_code[code]
        h_obs = g[COVAR_H].to_numpy(dtype=float)
        c_obs = g[TARGET].to_numpy(dtype=float)
        t_norm = np.array([date_to_norm[d] for d in g["date"]], dtype=np.float32)

        pinn = RegionalPINN(population=pop, t_min=0.0, t_max=1.0).to(DEVICE)
        parts = pretrain_pinn_region(
            pinn, t_norm, h_obs, c_obs, pop,
            epochs=1500, lr=5e-3, lambda_ode=0.1, n_collocation=256,
        )
        print(f"{region_name:30s}  {pop/1e6:5.2f}  "
              f"{parts['data']:.3e}  {parts['ode']:.3e}  {parts['total']:.3e}")
        rows.append({
            "region": region_name,
            "population_M": pop / 1e6,
            "pinn_data": parts["data"],
            "pinn_ode": parts["ode"],
            "pinn_total": parts["total"],
        })

    out_path = root / "results" / "forecasting" / "pinn_convergence_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())