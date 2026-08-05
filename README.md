# UKCI 2026 - Critical-Care Surge Capacity Planning

**Title:** Physics-Informed Multi-Quantile Forecasting for Risk-Averse NHS Critical-Care Surge Allocation

**Authors:** Michael Ajao-Olarinoye, Abiola Babatunde, AmirHosein Sadeghimanesh,
Fei He, and Matthew England (Centre for Computational Sciences and Mathematical
Modelling, Coventry University)

**Conference:** UKCI 2026, Coventry, 9-11 September 2026

This repository implements the full forecast-to-decision pipeline: per-region
physics-informed neural epidemic forecasting, demand scenario generation, and
robust optimisation for NHS England critical-care surge capacity planning.

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/michaelajao/ukci2026.git
cd ukci2026
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

# 2. Download NHS data (one-off, ~50 MB total)
ukci-download-nhs-data
ukci-download-supporting-data        # ONS populations, geography

# 3. Build the regional tidy dataset and features
ukci-build-regional-dataset
ukci-build-regional-features

# 4. Train forecasters (PINN-SEIRD + baselines)
ukci-train-forecasters

# 5. Rebuild forecast evaluation outputs
ukci-forecast-evaluation all

# 6. Generate scenarios and run the optimisation
ukci-run-allocation-e2
```

Forecast evaluation artifacts are generated from saved outputs:

```bash
ukci-forecast-evaluation sources   # list the source CSVs behind the headline tables
ukci-forecast-evaluation all       # rebuild metrics, headline tables, and forecast figure
```

The point-forecast results live in `results/forecasting/table1_paper.csv`,
the quantile metrics in `results/forecasting/table_quantile_metrics.csv`, and
the allocation results in `results/allocation/table2_allocation.csv`. Detailed
regional metrics remain available for supplementary analysis.

When running checks without activating the environment first, use:

```bash
python -m compileall -q src
```

## Command reference

All `ukci-*` console commands are declared in `pyproject.toml` and become
available after the editable install.

| Command | Purpose |
|---|---|
| `ukci-download-nhs-data` | Download NHS England COVID-19 hospital-activity archives |
| `ukci-download-supporting-data` | Download ONS populations and geography |
| `ukci-build-regional-dataset` | Build the tidy per-region daily dataset |
| `ukci-build-regional-features` | Derive modelling features (lags, slopes, splits) |
| `ukci-train-forecasters` | Train PINN-SEIRD and baseline forecasters |
| `ukci-run-pinn-ablations` | Run the PINN ablation study |
| `ukci-forecast-evaluation` | Rebuild forecast metrics, Table 1, and figures |
| `ukci-run-eda` | Generate exploratory-data-analysis figures |
| `ukci-run-allocation-e2` | Run the core allocation experiment (deterministic + risk-averse LP + baselines) |
| `ukci-run-allocation-sweeps` | Budget / travel-cap / tail-weight sensitivity sweeps |
| `ukci-run-allocation-revision` | Revision-pass allocation re-runs |
| `ukci-build-allocation-figures` | Build allocation figures from saved results |

## Repository layout

Reusable research logic and command entry points live under `src`. Editable
installs expose the `ukci-*` console commands declared in `pyproject.toml`.
Results and figures are committed so the headline tables and figures are
available without re-running the heavy pipeline; only bulky raw NHS data is
gitignored.

```text
ukci2026/
|-- data/
|   |-- raw/                       # NHS XLSX archives (gitignored, downloaded)
|   |-- processed/                 # Tidy regional CSV
|   `-- graphs/                    # NHS region adjacency, distance, correlation
|-- src/                           # Python packages and command entry points
|   |-- data/                      # NHS ingestion, splits, scenarios
|   |-- forecasting/               # PINN-SEIRD, multi-quantile loss, baselines
|   |-- optimization/              # LP, risk-averse LP, heuristics, sensitivity sweeps
|   |-- evaluation/                # Forecast and allocation metrics, EDA
|   `-- utils.py                   # Shared infrastructure helpers
|-- results/                       # Output tables and metrics (committed)
|-- figures/                       # Output figures (committed)
|-- pyproject.toml
`-- README.md
```

## Development workflow

### Branching

- `main` - protected, only via PR
- `forecast/<feature>` - forecasting experiments
- `opt/<feature>` - optimisation experiments
- `data/<task>` - data ingestion and processing

### Commits

Conventional Commits format:

- `feat(forecast): add PINN-SEIRD per-region module`
- `fix(data): handle NHS region renaming in 2022-08 archive`
- `experiment(opt): budget sweep at tighter fractions`
- `docs(method): expand multi-quantile loss derivation`

### Checks

```bash
python -m compileall -q src
ruff check src
```

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).

## Citation

If this work is useful, please cite:

```bibtex
@inproceedings{ajao-olarinoye2026multi,
  title  = {Physics-Informed Multi-Quantile Forecasting for
            Risk-Averse {NHS} Critical-Care Surge Allocation},
  author = {Ajao-Olarinoye, Michael and Babatunde, Abiola and
            Sadeghimanesh, AmirHosein and He, Fei and England, Matthew},
  booktitle = {Proceedings of the 25th UK Workshop on
               Computational Intelligence (UKCI 2026)},
  year   = {2026},
  publisher = {Springer},
}
```
