from __future__ import annotations

import pandas as pd

from credit_scoring.data import make_binary_target, make_time_split, reduce_memory_usage


def test_make_binary_target_accepts_numeric_lending_labels() -> None:
    target = make_binary_target(pd.Series([0, 1, 0]), positive_label="1")

    assert target.tolist() == [0, 1, 0]


def test_make_time_split_keeps_later_rows_in_test() -> None:
    data = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01", "2015-04-01"]),
            "value": [1, 2, 3, 4],
        }
    )
    target = pd.Series([0, 1, 0, 1])

    x_train, x_test, y_train, y_test = make_time_split(data, target, test_size=0.5)

    assert x_train["value"].tolist() == [1, 2]
    assert x_test["value"].tolist() == [3, 4]
    assert y_train.tolist() == [0, 1]
    assert y_test.tolist() == [0, 1]


def test_reduce_memory_usage_downcasts_numeric_and_categorical_columns() -> None:
    data = pd.DataFrame({"number": [1.0, 2.0], "category": ["a", "b"]})

    reduced = reduce_memory_usage(data)

    assert str(reduced["number"].dtype) == "float32"
    assert str(reduced["category"].dtype) == "category"
