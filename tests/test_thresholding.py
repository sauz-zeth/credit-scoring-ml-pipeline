from __future__ import annotations

import numpy as np

from credit_scoring.thresholding import evaluate_threshold, tune_thresholds


def test_evaluate_threshold_computes_binary_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.9])

    result = evaluate_threshold(y_true, y_score, threshold=0.5)

    assert result.threshold == 0.5
    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.f1 > 0


def test_tune_thresholds_respects_precision_floor_when_feasible() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    result = tune_thresholds(y_true, y_score, precision_floor=1.0, grid_size=101)

    constrained = result["max_recall_at_precision_floor"]
    assert constrained.precision >= 1.0
    assert constrained.recall == 1.0
