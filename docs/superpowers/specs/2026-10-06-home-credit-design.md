# Home Credit Dataset Design

## Goal

Add the Kaggle Home Credit Credit Risk Model Stability training data as a second
config-selected dataset, derive a parquet feature table with PySpark, and
evaluate it with a strictly chronological holdout and cross-validation.

## Existing behavior and required correction

The current German Credit flow downloads OpenML data, creates a stratified
random split, preprocesses inside sklearn pipelines, compares four models, and
saves CSV splits and artifacts. Preserve that behavior for
`configs/default.yaml`.

Currently `train_pipeline()` tunes decision thresholds against `y_test`. This
leaks holdout labels into threshold selection. Both datasets will instead pick
the deployed threshold using out-of-fold predictions from the training data;
the holdout is evaluated only after model and threshold selection.

## Design

### Configuration and data flow

Add `data.dataset` with values `german_credit` and `home_credit`. Keep the
existing config as German Credit and add `configs/home_credit.yaml` with
dataset-specific paths and controls, including `sample_frac`, `holdout_weeks`,
Spark driver memory/shuffle partitions, feature missingness limit, model
imbalance options, and `precision_floor`.

The CLI remains unchanged; each existing command dispatches through the
selected dataset config. German Credit retains OpenML download, current
feature/preprocessing flow, and stratified CV. Home Credit uses the competition
training parquet files only. `download-data` uses Kaggle CLI when available;
when files are absent or Kaggle access is unavailable, fail with instructions
to put parquet files under `data/raw/home_credit/`.

### Home Credit feature table

Find parquet shards by table-name glob rather than fixed filenames. Read
`feature_definitions.csv` and Spark schemas to classify source features:
numeric columns are aggregated with mean/max/min; date columns become day
offsets from `date_decision` before aggregation; categorical columns contribute
distinct counts and deterministic last values. For depth=2 tables, first
aggregate by `(case_id, num_group1)`, then aggregate those group summaries by
`case_id`. Left-join aggregates onto `case_id`, `WEEK_NUM`, `date_decision`,
and `target`.

Sample case IDs before processing dependent tables and joins. Drop columns
above the configured missingness limit and constant columns, cast numeric
features to float32 where applicable, and save
`data/processed/home_credit/features.parquet`. Pandas loads that table for the
existing model pipeline.

### Splitting, fitting, and metrics

Sort distinct `WEEK_NUM` values; reserve the configured number of latest weeks
for holdout. Generate expanding-window CV folds from earlier weeks, ensuring
each validation window follows its training window. No holdout/test rows enter
training or CV. sklearn preprocessing remains inside each estimator pipeline,
so each fold fits transformations only on that fold's training rows.

Use OOF training predictions to select thresholds. Report ROC-AUC, PR-AUC,
F1, precision, recall, confusion matrix, and default rates for train and
holdout. Keep class weighting configurable. Cap Random Forest work with
configured sampling and limit SHAP to a fixed-size sample. Preserve current
German Credit model behavior unless required by the shared threshold-leakage
correction.

### Tests and documentation

Add small local Spark aggregation tests and temporal split tests covering no
case-ID overlap and strictly later holdout weeks. Keep all existing tests
passing and require `uv run ruff check .`.

Update README after a real `sample_frac=0.1` run. Report observed table and row
counts and exact metrics only; if a full run is not completed, state that
plainly. Document Java 17, approximate RAM needs, invocation commands, and
Random Forest sampling. Remove “Fraud Detection” from the title.

## Decisions and constraints

- The competition's unlabeled test split is excluded from feature training,
  CV, threshold selection, and final evaluation.
- Holdout is specified as a count of latest weeks (`holdout_weeks`), not a
  fraction of rows.
- Start Home Credit `precision_floor` at 0.10 as a configurable experimental
  constraint; it is not a guaranteed attainable precision and will not be
  selected using holdout labels.
- Do not add PSI, weekly Gini validation, calibration, or unrelated features.
- Record implementation in small commits by the requested stages: config,
  download, Spark features, splitting, training, and README.
