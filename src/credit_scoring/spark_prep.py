from __future__ import annotations

import logging
from collections.abc import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from credit_scoring.config import Config

LOGGER = logging.getLogger(__name__)

LEAKAGE_PREFIXES = (
    "total_pymnt",
    "total_rec_",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_",
    "next_pymnt_d",
    "out_prncp",
    "last_fico_range_",
    "last_credit_pull_d",
    "debt_settlement_",
    "settlement_",
    "hardship_",
)


def _is_leakage_column(name: str) -> bool:
    return name.startswith(LEAKAGE_PREFIXES)


def _parsed_emp_length(column: str):
    value = F.trim(F.col(column).cast("string"))
    digits = F.regexp_extract(value, r"(\d+)", 1)
    return (
        F.when(value == "< 1 year", F.lit(0))
        .when(F.length(digits) > 0, digits.cast("int"))
        .otherwise(F.lit(None).cast("int"))
    )


def prepare_lending_club_dataframe(
    data: DataFrame,
    feature_whitelist: Sequence[str],
) -> DataFrame:
    required_columns = {"id", "loan_status", "loan_amnt", "issue_d"}
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"Lending Club input is missing required columns: {missing}")

    selected_features = [
        name
        for name in feature_whitelist
        if name != "issue_d" and not _is_leakage_column(name)
    ]
    missing_features = sorted(set(selected_features) - set(data.columns) - {"earliest_cr_line"})
    if missing_features:
        raise ValueError(f"Lending Club whitelist columns are missing: {missing_features}")

    clean = (
        data.filter(F.col("loan_status").isin("Fully Paid", "Charged Off"))
        .filter(F.length(F.trim(F.coalesce(F.col("id").cast("string"), F.lit("")))) > 0)
        .filter(F.col("loan_amnt").cast("double").isNotNull())
    )
    issue_date = F.to_date(F.trim(F.col("issue_d")), "MMM-yyyy")
    expressions = []
    for name in selected_features:
        if name == "earliest_cr_line":
            earliest_date = F.to_date(F.trim(F.col(name)), "MMM-yyyy")
            expressions.append(
                F.floor(F.months_between(issue_date, earliest_date))
                .cast("int")
                .alias("credit_history_months")
            )
        elif name == "term":
            expressions.append(
                F.regexp_replace(F.trim(F.col(name)), r"\s+months$", "")
                .cast("int")
                .alias(name)
            )
        elif name in {"int_rate", "revol_util"}:
            expressions.append(
                F.regexp_replace(F.trim(F.col(name)), "%", "").cast("double").alias(name)
            )
        elif name == "emp_length":
            expressions.append(_parsed_emp_length(name).alias(name))
        else:
            expressions.append(F.col(name))

    return clean.select(
        *expressions,
        issue_date.alias("issue_d"),
        F.when(F.col("loan_status") == "Charged Off", F.lit(1))
        .otherwise(F.lit(0))
        .alias("target"),
    )


def build_spark_session(config: Config) -> SparkSession:
    return (
        SparkSession.builder.appName("credit-scoring-lending-club")
        .master("local[*]")
        .config("spark.driver.memory", config.spark.driver_memory)
        .config("spark.sql.shuffle.partitions", config.spark.shuffle_partitions)
        .getOrCreate()
    )


def prepare_lending_club_file(config: Config) -> dict[str, int | str]:
    if not config.data.raw_path.exists():
        raise FileNotFoundError(f"Raw Lending Club file not found: {config.data.raw_path}")

    output_path = config.data.processed_dir / "loans.parquet"
    spark = build_spark_session(config)
    try:
        raw = (
            spark.read.option("header", True)
            .option("inferSchema", False)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(config.data.raw_path))
        )
        rows_before = raw.count()
        prepared = prepare_lending_club_dataframe(raw, config.data.feature_whitelist).cache()
        rows_after = prepared.count()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.write.mode("overwrite").parquet(str(output_path))
        LOGGER.info(
            "Saved Lending Club parquet to %s: %s rows before, %s after",
            output_path,
            rows_before,
            rows_after,
        )
        return {
            "raw_path": str(config.data.raw_path),
            "processed_path": str(output_path),
            "rows_before": rows_before,
            "rows_after": rows_after,
        }
    finally:
        spark.stop()
