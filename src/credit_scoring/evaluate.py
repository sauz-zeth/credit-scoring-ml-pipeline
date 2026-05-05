from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import pandas as pd

from credit_scoring.config import Config, ensure_directories
from credit_scoring.data import make_train_test_split, save_split
from credit_scoring.metrics import classification_metrics
from credit_scoring.plots import save_evaluation_plots
from credit_scoring.train import load_processed_test

LOGGER = logging.getLogger(__name__)


def _load_threshold(config: Config) -> float:
    if not config.artifacts.threshold_path.exists():
        return 0.5
    thresholds = json.loads(config.artifacts.threshold_path.read_text(encoding="utf-8"))
    return float(thresholds["max_f1"]["threshold"])


def evaluate_pipeline(config: Config) -> dict[str, Any]:
    ensure_directories(config)
    if not config.artifacts.best_model_path.exists():
        raise FileNotFoundError("Run `uv run credit-scoring train` before evaluate.")
    if not (config.data.processed_dir / "x_test.csv").exists():
        x_train, x_test, y_train, y_test = make_train_test_split(config)
        save_split(x_train, x_test, y_train, y_test, config.data.processed_dir)

    model = joblib.load(config.artifacts.best_model_path)
    x_test, y_test = load_processed_test(config.data.processed_dir)
    y_score = model.predict_proba(x_test)[:, 1]
    threshold = _load_threshold(config)
    metrics = classification_metrics(y_test.to_numpy(), y_score, threshold=threshold)
    pd.DataFrame([metrics]).to_csv(config.artifacts.test_metrics_path, index=False)
    save_evaluation_plots(y_test.to_numpy(), y_score, threshold, config.artifacts.plots_dir)
    LOGGER.info("Saved test metrics to %s", config.artifacts.test_metrics_path)
    return metrics
