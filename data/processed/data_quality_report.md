# Regional daily dataset — data quality report

Generated: 2026-05-12 15:52 UTC
Source files: 4 XLSX archive(s) in `data/raw/nhs/`

## Archive coverage

| Archive | rows extracted | min date | max date |
|---|---|---|---|
| `COVID-19-daily-admissions-and-beds-20210406-DQnotes.xlsx` | 6,972 | 2020-08-01 | 2021-04-06 |
| `COVID-19-daily-admissions-and-beds-20211207-20210407-20210930-DQnotes.xlsx` | 4,956 | 2021-04-07 | 2021-09-30 |
| `COVID-19-daily-admissions-and-beds-20220512-211001-220331-v2.xlsx` | 5,096 | 2021-10-01 | 2022-03-31 |
| `COVID-19-daily-admissions-and-beds-20220831-v2_DQnotes.xlsx` | 4,256 | 2022-04-01 | 2022-08-31 |

## Output dataset

- Rows: 5,327
- Date range: 2020-08-01 – 2022-08-31
- Distinct regions: 7
- Distinct dates: 761

### Per-metric summary statistics

| metric | non-null | mean | std | min | max |
|---|---|---|---|---|---|
| admissions | 5,313 | 137.6 | 125.0 | 0.0 | 977.0 |
| hospital_cases | 5,313 | 130.8 | 120.1 | 0.0 | 953.0 |
| occupied_beds | 5,327 | 1204.0 | 1141.5 | 11.0 | 7917.0 |
| mv_beds | 5,327 | 97.4 | 135.4 | 0.0 | 1220.0 |

## Validation

All checks passed: 7 regions, no date gaps, no missing target values.
