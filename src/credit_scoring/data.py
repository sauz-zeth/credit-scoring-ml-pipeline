from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring.config import Config, ensure_directories
from credit_scoring.spark_prep import prepare_lending_club_file

LOGGER = logging.getLogger(__name__)


def download_openml_dataset(config: Config, force: bool = False) -> Path:
    ensure_directories(config)
    output_path = config.data.raw_path
    if output_path.exists() and not force:
        LOGGER.info("Raw dataset already exists: %s", output_path)
        return output_path

    LOGGER.info(
        "Downloading OpenML dataset %s (%s)",
        config.data.openml_dataset_name,
        config.data.openml_dataset_id,
    )
    dataset = openml.datasets.get_dataset(config.data.openml_dataset_id)
    features, target, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )
    data = features.copy()
    data[config.data.target_column] = target
    data.to_csv(output_path, index=False)
    LOGGER.info("Saved raw dataset to %s with shape %s", output_path, data.shape)
    return output_path


def download_lending_club_dataset(config: Config, force: bool = False) -> Path:
    ensure_directories(config)
    output_path = config.data.raw_path
    if output_path.exists() and not force:
        LOGGER.info("Raw dataset already exists: %s", output_path)
        return output_path

    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    credentials_configured = (
        bool(os.getenv("KAGGLE_API_TOKEN"))
        or bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        or (config_dir / "kaggle.json").exists()
    )
    if not credentials_configured:
        raise RuntimeError(
            "Kaggle API is not configured. Set KAGGLE_API_TOKEN or "
            "KAGGLE_USERNAME/KAGGLE_KEY, or place kaggle.json in ~/.kaggle. "
            f"Otherwise download {config.data.kaggle_filename} manually into "
            f"{output_path.parent}/."
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(
            config.data.kaggle_dataset,
            config.data.kaggle_filename,
            path=str(output_path.parent),
            force=force,
            quiet=False,
        )
    except SystemExit as error:
        raise RuntimeError(
            "Kaggle authentication failed. Configure Kaggle credentials or place "
            f"{config.data.kaggle_filename} manually into {output_path.parent}/."
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"Could not download {config.data.kaggle_dataset}/{config.data.kaggle_filename} "
            f"via Kaggle API. Place the file manually into {output_path.parent}/."
        ) from error

    if not output_path.exists():
        archives = list(output_path.parent.glob("*.zip"))
        if archives:
            with zipfile.ZipFile(archives[-1]) as archive:
                if config.data.kaggle_filename not in archive.namelist():
                    raise RuntimeError(
                        f"Kaggle archive does not contain {config.data.kaggle_filename}."
                    )
                archive.extract(config.data.kaggle_filename, output_path.parent)
    if not output_path.exists():
        raise RuntimeError(
            f"Kaggle download completed but {output_path} was not created. "
            f"Place the file manually into {output_path.parent}/."
        )
    LOGGER.info("Saved raw dataset to %s", output_path)
    return output_path


def download_dataset(config: Config, force: bool = False) -> Path:
    if config.dataset == "german_credit":
        return download_openml_dataset(config, force=force)
    if config.dataset == "lending_club":
        return download_lending_club_dataset(config, force=force)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def load_raw_data(config: Config) -> pd.DataFrame:
    if not config.data.raw_path.exists():
        download_dataset(config)
    return pd.read_csv(config.data.raw_path)


def make_binary_target(target: pd.Series, positive_label: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        return target.fillna(0).astype(int)
    return (target.astype(str) == positive_label).astype(int)


def load_model_data(config: Config) -> tuple[pd.DataFrame, pd.Series]:
    if config.dataset == "german_credit":
        data = load_raw_data(config)
        return data.drop(columns=[config.data.target_column]), make_binary_target(
            data[config.data.target_column], config.project.positive_label
        )

    if config.dataset == "lending_club":
        download_dataset(config)
        parquet_path = config.data.processed_dir / "loans.parquet"
        if not parquet_path.exists():
            prepare_lending_club_file(config)
        data = pd.read_parquet(parquet_path)
        target = data.pop(config.data.target_column)
        if "issue_d" in data:
            data.pop("issue_d")
        return reduce_memory_usage(data), target.astype("int8")

    raise ValueError(f"Unsupported dataset: {config.dataset}")


def make_time_split(
    data: pd.DataFrame,
    target: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ordered = data.sort_values("issue_d")
    target = target.loc[ordered.index].reset_index(drop=True)
    ordered = ordered.reset_index(drop=True)
    split_at = max(1, int(len(ordered) * (1 - test_size)))
    x_train = ordered.iloc[:split_at].drop(columns=["issue_d"])
    x_test = ordered.iloc[split_at:].drop(columns=["issue_d"])
    return x_train, x_test, target.iloc[:split_at], target.iloc[split_at:]


def reduce_memory_usage(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], downcast="float")
        elif pd.api.types.is_integer_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], downcast="integer")
        elif pd.api.types.is_string_dtype(result[column]):
            result[column] = result[column].astype("category")
    return result


def make_train_test_split(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x, y = load_model_data(config)
    if config.dataset == "lending_club" and config.data.split_strategy == "time":
        data = pd.read_parquet(config.data.processed_dir / "loans.parquet")
        x_train, x_test, y_train, y_test = make_time_split(
            data.drop(columns=[config.data.target_column]),
            y,
            config.data.test_size,
        )
        return (
            reduce_memory_usage(x_train),
            reduce_memory_usage(x_test),
            y_train,
            y_test,
        )
    x = reduce_memory_usage(x) if config.dataset == "lending_club" else x
    return train_test_split(
        x,
        y,
        test_size=config.data.test_size,
        random_state=config.project.random_state,
        stratify=y,
    )


def save_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(output_dir / "x_train.csv", index=False)
    x_test.to_csv(output_dir / "x_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
