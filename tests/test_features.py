from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring.features import CreditFeatureEngineer


def test_credit_feature_engineer_adds_expected_features() -> None:
    data = pd.DataFrame(
        {
            "credit_amount": [1000, 0],
            "duration": [10, 0],
            "age": [40, 20],
            "purpose": ["car", "radio/tv"],
        }
    )

    transformed = CreditFeatureEngineer().fit_transform(data)

    assert "credit_amount_log1p" in transformed.columns
    assert "amount_per_month" in transformed.columns
    assert "duration_to_age" in transformed.columns
    assert transformed.loc[0, "amount_per_month"] == 100
    assert np.isclose(transformed.loc[0, "credit_amount_log1p"], np.log1p(1000))
    assert pd.isna(transformed.loc[1, "amount_per_month"])
