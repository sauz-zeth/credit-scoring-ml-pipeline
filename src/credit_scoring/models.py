from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_scoring.features import build_preprocessor


def build_estimators(
    model_config: dict[str, dict[str, Any]],
    random_state: int,
    n_jobs: int,
) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        **model_config["logistic_regression"],
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        **model_config["random_forest"],
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LGBMClassifier(
                        **model_config["lightgbm"],
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=n_jobs,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "catboost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    CatBoostClassifier(
                        **model_config["catboost"],
                        auto_class_weights="Balanced",
                        random_seed=random_state,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        ),
    }
