from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from credit_scoring.config import Config, ensure_directories
from credit_scoring.data import load_raw_data, make_train_test_split, save_split
from credit_scoring.metrics import classification_metrics
from credit_scoring.models import build_estimators
from credit_scoring.plots import save_eda_plots
from credit_scoring.thresholding import tune_thresholds

LOGGER = logging.getLogger(__name__)
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


def _predict_scores(model: Any, x: pd.DataFrame) -> Any:
    return model.predict_proba(x)[:, 1]


def train_pipeline(config: Config) -> dict[str, Any]:
    ensure_directories(config)
    raw_data = load_raw_data(config)
    save_eda_plots(raw_data, config.data.target_column, config.artifacts.plots_dir)

    x_train, x_test, y_train, y_test = make_train_test_split(config)
    if config.training.max_train_rows is not None:
        x_train = x_train.head(config.training.max_train_rows)
        y_train = y_train.head(config.training.max_train_rows)
    save_split(x_train, x_test, y_train, y_test, config.data.processed_dir)

    estimators = build_estimators(
        config.models,
        random_state=config.project.random_state,
        n_jobs=config.training.n_jobs,
    )
    cv = StratifiedKFold(
        n_splits=config.training.cv_folds,
        shuffle=True,
        random_state=config.project.random_state,
    )

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for name, estimator in estimators.items():
        LOGGER.info("Cross-validating %s", name)
        scores = cross_validate(
            estimator,
            x_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            return_train_score=False,
        )
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)

        LOGGER.info("Fitting %s on full training split", name)
        estimator.fit(x_train, y_train)
        fitted_models[name] = estimator

    cv_results = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    cv_results.to_csv(config.artifacts.cv_results_path, index=False)
    best_name = str(cv_results.iloc[0]["model"])
    best_model = fitted_models[best_name]
    joblib.dump(best_model, config.artifacts.best_model_path)

    y_score = _predict_scores(best_model, x_test)
    thresholds = tune_thresholds(
        y_test.to_numpy(),
        y_score,
        precision_floor=config.project.precision_floor,
    )
    chosen_threshold = thresholds["max_f1"].threshold
    test_metrics = classification_metrics(y_test.to_numpy(), y_score, threshold=chosen_threshold)
    test_metrics.update({"model": best_name})

    metrics_payload = {
        "best_model": best_name,
        "selection_metric": "cv roc_auc_mean",
        "cv_best_score": float(cv_results.iloc[0]["roc_auc_mean"]),
        "test_metrics_at_max_f1_threshold": test_metrics,
        "thresholds": {name: value.as_dict() for name, value in thresholds.items()},
    }
    config.artifacts.metrics_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    config.artifacts.threshold_path.write_text(
        json.dumps({name: value.as_dict() for name, value in thresholds.items()}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([test_metrics]).to_csv(config.artifacts.test_metrics_path, index=False)
    LOGGER.info("Best model: %s", best_name)
    return metrics_payload


def load_processed_test(processed_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    x_test = pd.read_csv(processed_dir / "x_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").iloc[:, 0]
    return x_test, y_test
