from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def save_eda_plots(data: pd.DataFrame, target_column: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x=target_column)
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close()

    numeric = data.select_dtypes(include=np.number)
    if not numeric.empty:
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
        plt.title("Numeric feature correlations")
        plt.tight_layout()
        plt.savefig(output_dir / "numeric_correlations.png", dpi=160)
        plt.close()


def save_evaluation_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix @ threshold={threshold:.3f}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def save_feature_importance_plot(
    importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 25,
) -> None:
    top = importance.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(9, max(4, len(top) * 0.28)))
    sns.barplot(data=top, x="importance", y="feature")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
