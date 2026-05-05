from __future__ import annotations

import logging
from pathlib import Path

import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring.config import Config, ensure_directories

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


def load_raw_data(config: Config) -> pd.DataFrame:
    if not config.data.raw_path.exists():
        download_openml_dataset(config)
    return pd.read_csv(config.data.raw_path)


def make_binary_target(target: pd.Series, positive_label: str) -> pd.Series:
    return (target.astype(str) == positive_label).astype(int)


def make_train_test_split(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = load_raw_data(config)
    x = data.drop(columns=[config.data.target_column])
    y = make_binary_target(data[config.data.target_column], config.project.positive_label)
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
