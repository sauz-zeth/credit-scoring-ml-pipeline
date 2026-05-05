# Credit Scoring / Fraud Detection | Tabular ML Pipeline

Production-style pet project for a DS / Classic ML internship portfolio. The project demonstrates an end-to-end tabular classification workflow: dataset loading, EDA, preprocessing, feature engineering, model comparison, threshold tuning, evaluation, explainability, and reproducible artifacts.

## Why this project

Credit scoring and fraud/risk detection are common applied ML tasks: the business goal is not just high ROC-AUC, but a controlled trade-off between catching risky clients and avoiding too many false alarms. This repository is structured as a maintainable Python package rather than a notebook-only experiment, so it is suitable for code review during an ML internship interview.

## Dataset

The default dataset is `credit-g` from OpenML (`data_id=31`), also known as the German Credit dataset. It contains mixed numerical and categorical features describing credit applicants. The target is converted to a binary label where:

- `bad` -> positive class `1` (higher credit risk)
- `good` -> negative class `0`

The dataset is downloaded automatically through OpenML. No Kaggle API or manual browser download is required.

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

This repository does not hard-code or invent metrics. Run the pipeline locally to generate results:

- `artifacts/metrics.json`
- `artifacts/cv_results.csv`
- `artifacts/test_metrics.csv`
- `artifacts/feature_importance.csv`
- `artifacts/best_model.joblib`
- `artifacts/plots/*.png`

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
│   └── default.yaml            # Dataset, training, model, artifact settings
├── data/
│   ├── raw/                    # Downloaded OpenML CSV
│   └── processed/              # Deterministic train/test split
├── src/
│   └── credit_scoring/
│       ├── cli.py              # Typer CLI entrypoint
│       ├── config.py           # Typed config loading
│       ├── data.py             # OpenML download and split logic
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
