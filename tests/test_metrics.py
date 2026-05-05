from __future__ import annotations

import numpy as np

from credit_scoring.metrics import classification_metrics


def test_classification_metrics_returns_expected_keys() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = classification_metrics(y_true, y_score, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
