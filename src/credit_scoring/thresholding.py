from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def evaluate_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> ThresholdResult:
    y_pred = (y_score >= threshold).astype(int)
    return ThresholdResult(
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def tune_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    precision_floor: float,
    grid_size: int = 1001,
) -> dict[str, ThresholdResult]:
    thresholds = np.linspace(0.0, 1.0, grid_size)
    results = [evaluate_threshold(y_true, y_score, threshold) for threshold in thresholds]
    best_f1 = max(results, key=lambda item: (item.f1, item.recall, item.precision))
    feasible = [item for item in results if item.precision >= precision_floor]
    best_recall_at_precision = max(
        feasible,
        key=lambda item: (item.recall, item.f1, item.precision),
    ) if feasible else best_f1
    return {
        "max_f1": best_f1,
        "max_recall_at_precision_floor": best_recall_at_precision,
    }
