# Graphs and adjacency matrices

Adjacency and travel-time matrices for the seven NHS England regions.

| File | Format | Source |
|---|---|---|
| `nhs_region_adj.txt` | 7×7 binary CSV (no header, no row labels) | Copied verbatim from `data/nhs-adj.txt` (the same matrix used in MSAGAT-Net's NHS-ICUBeds benchmark). |

## Column / row order — to be reconciled

The adjacency was inherited from MSAGAT-Net without an accompanying labels file, so the row/column order is **not yet pinned to NHS region codes**. We must reconcile the order against `data/nhs_timeseries.txt` (whose column order is the same, by construction in the original repo) before relying on this matrix for any per-region computation.

The seven NHS England regions (alphabetical by canonical name):

| Code | Name |
|---|---|
| Y61 | East of England |
| Y56 | London |
| Y60 | Midlands |
| Y63 | North East and Yorkshire |
| Y62 | North West |
| Y59 | South East |
| Y58 | South West |

## Reconciliation tasks (D2)

When `scripts/build_regional_dataset.py` runs and produces `data/processed/regional_daily.csv`, we recover the canonical NHS region naming and ordering from the XLSX archives. At that point:

1. Compute correlation between each column of `data/nhs_timeseries.txt` and each region's MV-bed time series in the fresh `regional_daily.csv`.
2. The column with highest correlation gives the region label for that index.
3. Write the resulting label vector to `data/graphs/nhs_region_labels.csv` (one row: `Y56,Y58,...` in the order matching `nhs_region_adj.txt`).

## What this adjacency is *not* used for

The optimisation module's travel-time matrix `T_ij` (Methodology §0) is **not** derived from this binary adjacency — it is built from region-centroid great-circle distance × 1.3 (detour factor), per Programme §6. This adjacency exists for:

- Optional graph-based forecasting baselines (not implemented in this paper — see plan §3 forecasting baselines for AIIM-overlap rationale).
- Visualisation: NHS regions map figure with neighbour-edges overlay.
- Sanity checks against MSAGAT-Net (same adjacency means the same notion of regional structure).

## Future additions

When the optimisation module is implemented (D8–D13), this directory will also hold:

- `nhs_region_centroids.csv` — lat/long for each region centroid.
- `nhs_region_travel_time.csv` — symmetric 7×7 travel-time matrix `T_ij` in minutes.
- `nhs_region_population.csv` — ONS mid-year population estimates `p_i` per region.
