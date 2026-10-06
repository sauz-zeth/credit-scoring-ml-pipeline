from __future__ import annotations

from pyspark.sql import SparkSession

from credit_scoring.spark_prep import prepare_lending_club_dataframe

FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "revol_util",
    "emp_length",
    "earliest_cr_line",
    "grade",
]


def test_prepare_lending_club_filters_statuses_and_parses_values(
    spark: SparkSession,
) -> None:
    data = spark.createDataFrame(
        [
            (
                "a", "Fully Paid", "10000", " 36 months", "13.5%", "40.0%",
                "10+ years", "Jan-2000", "Jan-2015", "B", "1",
            ),
            (
                "b", "Charged Off", "8000", " 60 months", "21.0%", "55.5%",
                "< 1 year", "Jun-2014", "Jun-2016", "D", "1",
            ),
            (
                "c", "Current", "9000", " 36 months", "10.0%", "20.0%",
                "2 years", "Jan-2010", "Jan-2015", "A", "1",
            ),
            (
                "", "Fully Paid", "7000", " 36 months", "10.0%", "20.0%",
                "2 years", "Jan-2010", "Jan-2015", "A", "1",
            ),
            (
                "d", "Fully Paid", "", " 36 months", "10.0%", "20.0%",
                "2 years", "Jan-2010", "Jan-2015", "A", "1",
            ),
        ],
        [
            "id",
            "loan_status",
            "loan_amnt",
            "term",
            "int_rate",
            "revol_util",
            "emp_length",
            "earliest_cr_line",
            "issue_d",
            "grade",
            "total_pymnt",
        ],
    )

    result = prepare_lending_club_dataframe(data, FEATURES)
    rows = result.orderBy("id").collect()

    assert len(rows) == 2
    assert [row.target for row in rows] == [0, 1]
    assert rows[0].term == 36
    assert rows[1].term == 60
    assert rows[0].int_rate == 13.5
    assert rows[0].revol_util == 40.0
    assert rows[0].emp_length == 10
    assert rows[1].emp_length == 0
    assert rows[0].credit_history_months == 180
    assert rows[1].credit_history_months == 24


def test_prepare_lending_club_excludes_leakage_columns(spark: SparkSession) -> None:
    data = spark.createDataFrame(
        [
            ("a", "Fully Paid", "10000", "Jan-2015", "Jan-2000", "10.0"),
        ],
        ["id", "loan_status", "loan_amnt", "issue_d", "earliest_cr_line", "total_rec_prncp"],
    )

    result = prepare_lending_club_dataframe(
        data,
        ["loan_amnt", "earliest_cr_line", "total_rec_prncp"],
    )

    assert "issue_d" in result.columns
    assert "target" in result.columns
    assert "credit_history_months" in result.columns
    assert "total_rec_prncp" not in result.columns
