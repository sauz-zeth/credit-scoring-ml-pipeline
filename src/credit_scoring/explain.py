from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from credit_scoring.config import Config, ensure_directories
from credit_scoring.plots import save_feature_importance_plot
from credit_scoring.train import load_processed_test

LOGGER = logging.getLogger(__name__)


def _feature_names(model: Any) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def _model_feature_importance(model: Any) -> pd.DataFrame | None:
    estimator = model.named_steps["model"]
    feature_names = _feature_names(model)
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        return None
    return pd.DataFrame({"feature": feature_names, "importance": importance})


def _shap_importance(model: Any, x_sample: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import shap

        transformed = model.named_steps["preprocessor"].transform(x_sample)
        estimator = model.named_steps["model"]
        explainer = shap.Explainer(estimator, transformed)
        values = explainer(transformed)
        shap_values = values.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        importance = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({"feature": _feature_names(model), "importance": importance})
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("SHAP failed, falling back to model/permutation importance: %s", error)
        return None


def explain_pipeline(config: Config, sample_size: int = 200) -> pd.DataFrame:
    ensure_directories(config)
    if not config.artifacts.best_model_path.exists():
        raise FileNotFoundError("Run `uv run credit-scoring train` before explain.")
    x_test, y_test = load_processed_test(config.data.processed_dir)
    model = joblib.load(config.artifacts.best_model_path)
    x_sample = x_test.sample(
        n=min(sample_size, len(x_test)),
        random_state=config.project.random_state,
    )

    importance = _shap_importance(model, x_sample)
    if importance is None:
        importance = _model_feature_importance(model)
    if importance is None:
        permutation = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=10,
            random_state=config.project.random_state,
            scoring="roc_auc",
        )
        importance = pd.DataFrame(
            {"feature": x_test.columns, "importance": permutation.importances_mean}
        )

    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    importance.to_csv(config.artifacts.feature_importance_path, index=False)
    save_feature_importance_plot(
        importance,
        config.artifacts.plots_dir / "feature_importance.png",
    )
    LOGGER.info("Saved feature importance to %s", config.artifacts.feature_importance_path)
    return importance
