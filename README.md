# UKCI 2026 â€” Critical-Care Surge Capacity Planning

**Working title:** Physics-Informed ICU Bed Forecasting with Cost-Asymmetric Quantile Loss and Robust Optimisation for NHS Critical-Care Surge Capacity Under Demand Uncertainty

**Authors (planned):** Michael Ajao-Olarinoye, Abiola Babatunde, Vasile Palade
**Conference:** UKCI 2026, Coventry, 9â€“11 September 2026
**Submission deadline:** 31 May 2026 via Microsoft CMT

This repository implements the full forecast-to-decision pipeline described in
`docs/01_RESEARCH_PROGRAMME.md`: per-region physics-informed neural epidemic
forecasting, demand scenario generation, and metaheuristic robust optimisation
for NHS England critical-care surge capacity planning.

## Quickstart

```bash
# 1. Clone and set up environment
git clone <repo-url> ukci2026
cd ukci2026
conda activate pyt_env
python -m pip install -e ".[dev]"

# 2. Download NHS data (one-off, ~50 MB total)
ukci-download-nhs-data

# 3. Build the regional tidy dataset
ukci-build-regional-dataset

# 4. Train forecasters
ukci-train-forecasters

# 5. Rebuild paper-facing forecast outputs
ukci-forecast-evaluation all

# 6. Generate scenarios and run the optimisation
ukci-run-allocation-e2
```

Forecast evaluation artifacts are generated from saved outputs. Use the CSVs as
the internal source of truth for manuscript values, then enter those values into
the LaTeX table directly:

```bash
ukci-forecast-evaluation sources   # list CSVs used as paper source tables
ukci-forecast-evaluation all       # rebuild metrics, Table 1, and forecast figure
```

For the main paper, use `results/forecasting/table1_paper.csv` for the
forecasting table and `results/allocation/table2_allocation.csv` for the
allocation table. Detailed regional metrics can stay in the appendix.

When running checks without activating the environment first, use:

```bash
conda run -n pyt_env python -m compileall -q src
```

## Documentation

| Document | Purpose |
|---|---|
| [`docs/01_RESEARCH_PROGRAMME.md`](docs/01_RESEARCH_PROGRAMME.md) | Strategic plan, paper structure, contributions, risks |
| [`docs/02_METHODOLOGY.md`](docs/02_METHODOLOGY.md) | Mathematical formulation, architecture, hyperparameters |
| [`docs/03_TIMELINE.md`](docs/03_TIMELINE.md) | Day-by-day work plan, gates, experiment matrix |
| [`docs/paper/`](docs/paper/) | Actual UKCI manuscript source and `docs/paper/out/` build artifacts |
| [`docs/ukci_springer_template/`](docs/ukci_springer_template/) | Original UKCI/Springer SVProc template bundle downloaded from the conference website |
| `docs/status/` | Dated status notes recording gate decisions |

## Repository layout

Reusable research logic and command entry points live under `src`. Editable
installs expose the `ukci-*` console commands declared in `pyproject.toml`.

```
ukci2026/
â”œâ”€â”€ docs/                          # Planning and methodology documents
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                       # NHS XLSX archives (gitignored, downloaded)
â”‚   â”œâ”€â”€ processed/                 # Tidy regional CSV
â”‚   â””â”€â”€ graphs/                    # NHS region adjacency, distance, correlation
â”œâ”€â”€ src/                           # Python packages and command entry points
â”‚   â”œâ”€â”€ data/                      # NHS ingestion, splits, scenarios
â”‚   â”œâ”€â”€ forecasting/               # PINN-SEIRD, cost-asymmetric loss, baselines
â”‚   â”œâ”€â”€ optimization/              # MILP, robust MILP, heuristics, metaheuristics, Îµ-constraint
â”‚   â”œâ”€â”€ evaluation/                # Forecast and allocation metrics
â”‚   â””â”€â”€ utils.py                   # Shared infrastructure helpers
â”œâ”€â”€ configs/                       # YAML experiment configs
â”œâ”€â”€ notebooks/                     # EDA and analysis notebooks
â”œâ”€â”€ tests/                         # pytest unit tests
â”œâ”€â”€ results/                       # Output tables and metrics (gitignored)
â”œâ”€â”€ figures/                       # Output figures (gitignored)
â”œâ”€â”€ pyproject.toml
â””â”€â”€ README.md
```

## Development workflow

### Branching

- `main` â€” protected, only via PR
- `paper/draft` â€” paper writing, LaTeX
- `forecast/<feature>` â€” forecasting experiments
- `opt/<feature>` â€” optimisation experiments
- `data/<task>` â€” data ingestion and processing

### Commits

Conventional Commits format:

- `feat(forecast): add PINN-SEIRD per-region module`
- `fix(data): handle NHS region renaming in 2022-08 archive`
- `experiment(opt): NSGA-II on London trust subset`
- `docs(method): expand cost-asymmetric loss derivation`

### Tests

```bash
pytest tests/                       # all
pytest tests/test_data.py           # data harmonisation
pytest -m "not slow"                # skip slow integration tests
```

## License

To be decided before submission. Apache 2.0 or MIT recommended for the paper's
reproducibility statement.

## Citation

If this work is useful, please cite (placeholder until acceptance):

```bibtex
@inproceedings{ajao-olarinoye2026physics,
  title  = {Physics-Informed {ICU} Bed Forecasting with
            Cost-Asymmetric Quantile Loss and Robust Optimisation
            for {NHS} Critical-Care Surge Capacity Under
            Demand Uncertainty},
  author = {Ajao-Olarinoye, Michael and Babatunde, Abiola and Palade, Vasile},
  booktitle = {Proceedings of the 25th UK Workshop on
               Computational Intelligence (UKCI 2026)},
  year   = {2026},
  publisher = {Springer},
}
```
