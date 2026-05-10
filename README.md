# UKCI 2026 — Critical-Care Surge Capacity Planning

**Working title:** From Forecasts to Capacity Decisions: A Physics-Informed and Metaheuristic Pipeline for NHS Critical-Care Surge Planning

**Authors (planned):** Michael Ajao-Olarinoye, Abiola Babatunde, Vasile Palade
**Conference:** UKCI 2026, Coventry, 9–11 September 2026
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
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Download NHS data (one-off, ~50 MB total)
python scripts/download_nhs_data.py

# 3. Build the regional tidy dataset
python scripts/build_regional_dataset.py

# 4. Run a baseline forecasting experiment
python -m critical_care_surge.forecasting.train --config configs/baseline_lstm.yaml

# 5. Run the proposed PINN-GRU model
python -m critical_care_surge.forecasting.train --config configs/proposed.yaml

# 6. Generate scenarios and run the optimisation
python -m critical_care_surge.optimization.solve --config configs/milp_regional.yaml
```

## Documentation

| Document | Purpose |
|---|---|
| [`docs/01_RESEARCH_PROGRAMME.md`](docs/01_RESEARCH_PROGRAMME.md) | Strategic plan, paper structure, contributions, risks |
| [`docs/02_METHODOLOGY.md`](docs/02_METHODOLOGY.md) | Mathematical formulation, architecture, hyperparameters |
| [`docs/03_TIMELINE.md`](docs/03_TIMELINE.md) | Day-by-day work plan, gates, experiment matrix |
| `docs/status/` | Dated status notes recording gate decisions |

## Repository layout

```
ukci2026/
├── docs/                          # Planning and methodology documents
├── data/
│   ├── raw/                       # NHS XLSX archives (gitignored, downloaded)
│   ├── processed/                 # Tidy regional CSV
│   └── graphs/                    # NHS region adjacency, distance, correlation
├── scripts/                       # CLI utilities (download, build, etc.)
├── src/critical_care_surge/       # Python package
│   ├── data/                      # NHS ingestion, splits, scenarios
│   ├── forecasting/               # PINN-SEIRD, GRU, baselines, training
│   ├── optimization/              # MILP, robust MILP, GA, NSGA-II, SA
│   ├── evaluation/                # Forecast and allocation metrics
│   └── utils/                     # Shared helpers, logging, seeds
├── configs/                       # YAML experiment configs
├── notebooks/                     # EDA and analysis notebooks
├── tests/                         # pytest unit tests
├── results/                       # Output tables and metrics (gitignored)
├── figures/                       # Output figures (gitignored)
├── pyproject.toml
└── README.md
```

## Development workflow

### Branching

- `main` — protected, only via PR
- `paper/draft` — paper writing, LaTeX
- `forecast/<feature>` — forecasting experiments
- `opt/<feature>` — optimisation experiments
- `data/<task>` — data ingestion and processing

### Commits

Conventional Commits format:

- `feat(forecast): add PINN-SEIRD per-region module`
- `fix(data): handle NHS region renaming in 2022-08 archive`
- `experiment(opt): NSGA-II on London trust subset`
- `docs(method): expand decision-aware loss derivation`

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
@inproceedings{ajao-olarinoye2026forecasts,
  title  = {From Forecasts to Capacity Decisions:
            A Physics-Informed and Metaheuristic Pipeline
            for NHS Critical-Care Surge Planning},
  author = {Ajao-Olarinoye, Michael and Babatunde, Abiola and Palade, Vasile},
  booktitle = {Proceedings of the 25th UK Workshop on
               Computational Intelligence (UKCI 2026)},
  year   = {2026},
  publisher = {Springer},
}
```
