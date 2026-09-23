# Data

The input datasets are synthetic teaching datasets and are **not included** in this repository.
Only derived results (in `results/`) are published. To re-run the analyses, place CSVs with the
columns below in this folder.

| File | Rows | Columns |
|---|---|---|
| `gene_expression.csv` | 30 subjects | `ID`, 10 genes (`FKBP9`, `AHR`, `TSPAN5`, `CDCP4`, `KOR`, `MOR`, `DEFB1`, `ERICH3`, `COMT`, `CYP2C9`), `Diagnosis` (Healthy control / Mild / Moderate / Treatment resistant) |
| `ssri_cohort.csv` | 400 patients | `ID`, sociodemographic (`age`, `maritalStatus`, `smokingStatus`, `numChildren`, `seasonalDepression`, `socioEconomicStatus`), `MDD_severity`, biological (`plasmaSerotonin`, `AHR`, `TSPAN5`, `CYP2C19`, `DEFB1`, `SLC6A4`, `COMT`, `CYP2D6`, `ERICH3`), `response` (1 = remission) |
| `ssri_cohort_qids.csv` | 400 patients | row index `""`, all columns of `ssri_cohort.csv` plus 12 baseline QIDS-SR items (`QIDS1.baseline` … `QIDS16.baseline`) and `train_test` (`Train` / `Test`) |
