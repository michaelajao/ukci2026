# UKCI 2026 - Critical-Care Surge Capacity Planning

**Working title:** Physics-Informed ICU Bed Forecasting with Cost-Asymmetric
Quantile Loss and Robust Optimisation for NHS Critical-Care Surge Capacity
Under Demand Uncertainty

**Authors:** Michael Ajao-Olarinoye, Abiola Babatunde, AmirHosein Sadeghimanesh
(Centre for Computational Sciences and Mathematical Modelling, Coventry University)

**Conference:** UKCI 2026, Coventry, 9-11 September 2026

This repository implements the full forecast-to-decision pipeline: per-region
physics-informed neural epidemic forecasting, demand scenario generation, and
robust optimisation for NHS England critical-care surge capacity planning.

## Quickstart

```bash
# 1. Clone and set up environment
git clone <repo-url> ukci2026
cd ukci2026
conda activate pyt_env
python -m pip install -e ".[dev]"

# 2. Download NHS data (one-off, ~50 MB total)
ukci-download-nhs-data
ukci-download-supporting-data        # ONS populations, geography

# 3. Build the regional tidy dataset and features
ukci-build-regional-dataset
ukci-build-regional-features

# 4. Train forecasters (PINN-SEIRD + baselines)
ukci-train-forecasters

# 5. Rebuild paper-facing forecast outputs
ukci-forecast-evaluation all

# 6. Generate scenarios and run the optimisation
ukci-run-allocation-e2
```

Forecast evaluation artifacts are generated from saved outputs. The CSVs are the
internal source of truth for manuscript values; enter those values into the
LaTeX tables directly:

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
| `ukci-run-allocation-e2` | Run the core allocation experiment (deterministic + robust LP + baselines) |
| `ukci-run-allocation-sweeps` | Budget / travel-cap / tail-weight sensitivity sweeps |
| `ukci-run-allocation-revision` | Revision-pass allocation re-runs |
| `ukci-build-allocation-figures` | Build allocation figures from saved results |

## Documentation

| Document | Purpose |
|---|---|
| [`docs/paper/`](docs/paper/) | UKCI conference manuscript source (build with `make`; PDF in `docs/paper/out/`) |
| [`docs/journal/`](docs/journal/) | Journal-extension skeleton (Health Care Management Science target); PDF in `docs/journal/out/` |
| [`docs/ukci_springer_template/`](docs/ukci_springer_template/) | Original UKCI/Springer SVProc template bundle from the conference website |
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | Running status notes and gate decisions |
| [`presentation.md`](presentation.md) | Marp slide deck for the talk (renders to `presentation.pdf`) |

## Repository layout

Reusable research logic and command entry points live under `src`. Editable
installs expose the `ukci-*` console commands declared in `pyproject.toml`.
Results and figures are committed as the shared source of truth so co-authors
get paper-facing tables and figures without re-running the heavy pipeline; only
bulky raw NHS data and LaTeX build artifacts are gitignored.

```text
ukci2026/
|-- docs/                          # Manuscripts, journal extension, status notes
|-- data/
|   |-- raw/                       # NHS XLSX archives (gitignored, downloaded)
|   |-- processed/                 # Tidy regional CSV
|   `-- graphs/                    # NHS region adjacency, distance, correlation
|-- src/                           # Python packages and command entry points
|   |-- data/                      # NHS ingestion, splits, scenarios
|   |-- forecasting/               # PINN-SEIRD, cost-asymmetric loss, baselines
|   |-- optimization/              # LP, robust LP, heuristics, sensitivity sweeps
|   |-- evaluation/                # Forecast and allocation metrics, EDA
|   `-- utils.py                   # Shared infrastructure helpers
|-- configs/                       # YAML experiment configs
|-- notebooks/                     # EDA and analysis notebooks
|-- tests/                         # pytest unit tests
|-- results/                       # Output tables and metrics (committed)
|-- figures/                       # Output figures (committed)
|-- presentation.md                # Marp slide deck
|-- pyproject.toml
`-- README.md
```

## Development workflow

### Branching

- `main` - protected, only via PR
- `paper/draft` - paper writing, LaTeX
- `forecast/<feature>` - forecasting experiments
- `opt/<feature>` - optimisation experiments
- `data/<task>` - data ingestion and processing

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

Released under the MIT License (see [`LICENSE`](LICENSE)).

## Citation

If this work is useful, please cite (placeholder until acceptance):

```bibtex
@inproceedings{ajao-olarinoye2026physics,
  title  = {Physics-Informed {ICU} Bed Forecasting with
            Cost-Asymmetric Quantile Loss and Robust Optimisation
            for {NHS} Critical-Care Surge Capacity Under
            Demand Uncertainty},
  author = {Ajao-Olarinoye, Michael and Babatunde, Abiola and
            Sadeghimanesh, AmirHosein},
  booktitle = {Proceedings of the 25th UK Workshop on
               Computational Intelligence (UKCI 2026)},
  year   = {2026},
  publisher = {Springer},
}
```
