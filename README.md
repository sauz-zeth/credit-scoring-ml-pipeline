# Credit Scoring | Tabular ML Pipeline

Pet project for a DS / Classic ML portfolio: dataset loading, preprocessing, model comparison, threshold tuning, evaluation, explainability, and reproducible artifacts.

## Why this project

The business goal is not just high ROC-AUC, but a controlled trade-off between catching risky clients and avoiding too many false alarms. This repository is structured as a Python package rather than a notebook-only experiment.

## German Credit dataset

The default dataset is `credit-g` from OpenML (`data_id=31`), also known as the German Credit dataset. It contains mixed numerical and categorical features describing credit applicants. The target is converted to a binary label where:

- `bad` -> positive class `1` (higher credit risk)
- `good` -> negative class `0`

The dataset is downloaded automatically through OpenML. No Kaggle API or manual browser download is required.

## Lending Club dataset

The optional Lending Club configuration uses the `wordsforthewise/lending-club`
Kaggle dataset file `accepted_2007_to_2018Q4.csv.gz`. Configure the Kaggle API
with `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and `KAGGLE_KEY`) or
`~/.kaggle/kaggle.json`; alternatively, put the downloaded file in
`data/raw/lending_club/`. The preprocessing stage uses PySpark and writes
`data/processed/lending_club/loans.parquet`.

The observed source contained **2,260,701 rows**; the status and non-empty
`id`/`loan_amnt` filters retained **1,345,310 rows**. The resulting default
share (`Charged Off`, target 1) was **19.96%**. Only `Fully Paid` (target 0)
and `Charged Off` (target 1) are modeled; other statuses are excluded.

The configured pre-application feature whitelist is:
`loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `sub_grade`,
`emp_length`, `home_ownership`, `annual_inc`, `verification_status`,
`purpose`, `addr_state`, `dti`, `delinq_2yrs`, `earliest_cr_line`,
`fico_range_low`, `fico_range_high`, `inq_last_6mths`, `open_acc`,
`pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `mort_acc`,
`pub_rec_bankruptcies`, `application_type`.

`earliest_cr_line` is converted into `credit_history_months` relative to
`issue_d`; `issue_d` is retained in parquet solely for an optional chronological
split and is never passed to models. `term`, percentage fields, and employment
length are parsed during Spark preparation. The preparation uses this whitelist,
not a blacklist; payment, recovery, post-issuance FICO/credit-pull, hardship,
and settlement fields cannot enter the feature table.

Spark needs Java 17. For the configured 8 GB Spark driver and pandas/sklearn
modeling of 1.35 million rows, **16 GB RAM is a practical minimum; 32 GB is
recommended**. The Random Forest CV and final fit are limited to a deterministic
10,000-row training sample; SHAP explanation uses at most 1,000 test rows.
Other models train on the full training split.

```bash
uv sync
uv run credit-scoring run-all --config configs/lending_club.yaml
```

`split_strategy` may be `stratified` (default) or `time`; the latter trains on
earlier `issue_d` values and tests on later ones. Spark memory and shuffle
partitions, the whitelist, model sampling, and precision floor are set in
`configs/lending_club.yaml`.

## Methodology

The pipeline includes:

- EDA plots: target distribution and numeric correlation heatmap.
- Feature engineering: log credit amount, amount per month, duration-to-age ratio.
- Preprocessing: median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
- Train/test split: stratified split with fixed `random_state`.
- Cross-validation: model comparison using ROC-AUC, PR-AUC, F1, precision, and recall.
- Models: Logistic Regression baseline, Random Forest, LightGBM, CatBoost.
- Threshold tuning: maximize F1 and maximize recall subject to `precision >= precision_floor`.
- Explainability: SHAP when available; fallback to model feature importance or permutation importance.

## Results

German Credit metrics below are existing results from `uv run credit-scoring run-all`
with the default configuration on 2026-05-05. Model selection uses mean
cross-validated ROC-AUC on the training split.

### Cross-validation model comparison

| Model | ROC-AUC mean | ROC-AUC std | PR-AUC mean | F1 mean | Precision mean | Recall mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 0.8006 | 0.0266 | 0.6324 | 0.5586 | 0.5889 | 0.5375 |
| Random Forest | 0.7990 | 0.0380 | 0.6421 | 0.5312 | 0.6388 | 0.4583 |
| LightGBM | 0.7726 | 0.0434 | 0.5945 | 0.5569 | 0.5787 | 0.5417 |
| Logistic Regression | 0.7689 | 0.0511 | 0.5888 | 0.5806 | 0.5125 | 0.6708 |

### Holdout test metrics

Best model: `catboost`

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.8065 |
| PR-AUC | 0.7163 |
| F1 | 0.6565 |
| Precision | 0.6056 |
| Recall | 0.7167 |
| Threshold | 0.4300 |

Confusion matrix at threshold `0.43`:

| Actual \ Predicted | Good risk `0` | Bad risk `1` |
| --- | ---: | ---: |
| Good risk `0` | 112 | 28 |
| Bad risk `1` | 17 | 43 |

### Threshold tuning

| Strategy | Threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Maximize F1 | 0.4300 | 0.6056 | 0.7167 | 0.6565 |
| Maximize recall with precision >= 0.60 | 0.4300 | 0.6056 | 0.7167 | 0.6565 |

### Top feature importance

| Feature | Importance |
| --- | ---: |
| `checking_status_no checking` | 0.5902 |
| `duration` | 0.3319 |
| `credit_history_critical/other existing credit` | 0.3208 |
| `checking_status_<0` | 0.2348 |
| `amount_per_month` | 0.1997 |
| `duration_to_age` | 0.1913 |
| `purpose_used car` | 0.1544 |
| `savings_status_<100` | 0.1513 |
| `employment_4<=X<7` | 0.1326 |
| `property_magnitude_real estate` | 0.1313 |

Generated artifacts:

- `artifacts/metrics.json`
- `artifacts/cv_results.csv`
- `artifacts/test_metrics.csv`
- `artifacts/feature_importance.csv`
- `artifacts/best_model.joblib`
- `artifacts/plots/*.png`

### Lending Club results

Actual full-data run completed **October 7, 2026** with `configs/lending_club.yaml`:
stratified 80/20 train/test, 5-fold stratified CV, `random_state=42`,
`precision_floor=0.35`, configured model parameters, RF CV/final-fit sample
10,000, and SHAP sample cap 1,000. The CV and test ROC-AUC values are below
0.85, so the requested high-AUC leakage stop condition was not triggered.

| Model | CV ROC-AUC mean | CV ROC-AUC std | CV PR-AUC mean |
| --- | ---: | ---: | ---: |
| LightGBM | 0.7248 | 0.0012 | 0.3944 |
| CatBoost | 0.7210 | 0.0014 | 0.3886 |
| Logistic Regression | 0.7131 | 0.0014 | 0.3757 |
| Random Forest* | 0.6782 | 0.0094 | 0.3319 |

Best model: **LightGBM**. Holdout: ROC-AUC **0.7259**, PR-AUC **0.3943**,
F1 **0.4427**, precision **0.3396**, recall **0.6358** at threshold **0.525**.
This run retained 1,345,310 rows; the holdout contained 269,062 rows.
`*` Random Forest CV and fit used a 10,000-row sample; remaining models used
the full training data.

For comparison, a separate **20,000-row training-cap sample run** completed on
October 6, 2026 (3-fold CV; 30 trees/iterations for RF, LightGBM, and CatBoost):
best model CatBoost, CV ROC-AUC **0.7069**, holdout ROC-AUC **0.7019** and
PR-AUC **0.3620**. Its holdout remained the full 269,062 rows. This is a sample
run, not a full-training estimate.

## Quickstart

Install dependencies and run the full pipeline:

```bash
uv sync
uv run credit-scoring download-data
uv run credit-scoring train
uv run credit-scoring evaluate
uv run credit-scoring explain
```

Or run everything in one command:

```bash
uv run credit-scoring run-all
```

Lending Club:

```bash
uv run credit-scoring run-all --config configs/lending_club.yaml
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

## CLI

```bash
uv run credit-scoring download-data
uv run credit-scoring train
uv run credit-scoring evaluate
uv run credit-scoring explain
uv run credit-scoring run-all
```

All commands accept a custom config path:

```bash
uv run credit-scoring train --config configs/default.yaml
```

## Project structure

```text
.
├── artifacts/                 # Generated metrics, models, plots
├── configs/
│   ├── default.yaml            # German Credit configuration
│   └── lending_club.yaml       # Lending Club and Spark settings
├── data/
│   ├── raw/                    # Downloaded OpenML CSV
│   └── processed/              # Deterministic train/test split
├── src/
│   └── credit_scoring/
│       ├── cli.py              # Typer CLI entrypoint
│       ├── config.py           # Typed config loading
│       ├── data.py             # OpenML download and split logic
│       ├── spark_prep.py       # Lending Club Spark cleaning and parquet output
│       ├── evaluate.py         # Test evaluation and plots
│       ├── explain.py          # SHAP / fallback importance
│       ├── features.py         # Feature engineering and preprocessing
│       ├── metrics.py          # Classification metrics
│       ├── models.py           # Model factory
│       ├── plots.py            # EDA and evaluation plots
│       ├── thresholding.py     # Threshold optimization
│       └── train.py            # CV, model selection, artifact saving
├── tests/                      # Unit tests for core ML utilities
├── pyproject.toml              # Python 3.11+, uv-ready package config
└── README.md
```

## Configuration

Main settings live in `configs/default.yaml`:

- `precision_floor`: minimum precision for recall-oriented threshold search.
- `test_size`: holdout test share.
- `cv_folds`: number of stratified CV folds.
- model hyperparameters for Logistic Regression, Random Forest, LightGBM, and CatBoost.
- artifact output paths.

## Next steps

Potential improvements for a stronger portfolio version:

- Add hyperparameter optimization with Optuna.
- Add calibration curves and probability calibration.
- Add data validation with Pandera or Great Expectations.
- Add pre-commit hooks and GitHub Actions CI.
- Package generated metrics into a model card.
