from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Small deterministic feature transformer for credit-risk tabular data."""

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> CreditFeatureEngineer:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        result = x.copy()
        if "credit_amount" in result.columns:
            amount = pd.to_numeric(result["credit_amount"], errors="coerce").clip(lower=0)
            result["credit_amount_log1p"] = np.log1p(amount)
        if {"credit_amount", "duration"}.issubset(result.columns):
            amount = pd.to_numeric(result["credit_amount"], errors="coerce")
            duration = pd.to_numeric(result["duration"], errors="coerce").replace(0, np.nan)
            result["amount_per_month"] = amount / duration
        if {"age", "duration"}.issubset(result.columns):
            age = pd.to_numeric(result["age"], errors="coerce").replace(0, np.nan)
            duration = pd.to_numeric(result["duration"], errors="coerce")
            result["duration_to_age"] = duration / age
        return result


def build_preprocessor() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", CreditFeatureEngineer()),
            ("preprocessing", column_transformer),
        ]
    )
