from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    random_state: int
    positive_label: str
    precision_floor: float


@dataclass(frozen=True)
class DataConfig:
    openml_dataset_id: int
    openml_dataset_name: str
    target_column: str
    raw_path: Path
    processed_dir: Path
    test_size: float
    split_strategy: str
    kaggle_dataset: str | None
    kaggle_filename: str | None
    feature_whitelist: tuple[str, ...]


@dataclass(frozen=True)
class SparkConfig:
    driver_memory: str
    shuffle_partitions: int


@dataclass(frozen=True)
class TrainingConfig:
    cv_folds: int
    n_jobs: int
    scoring: str
    max_train_rows: int | None
    random_forest_sample_size: int | None
    shap_sample_size: int


@dataclass(frozen=True)
class ArtifactsConfig:
    dir: Path
    plots_dir: Path
    best_model_path: Path
    metrics_path: Path
    cv_results_path: Path
    test_metrics_path: Path
    threshold_path: Path
    feature_importance_path: Path


@dataclass(frozen=True)
class Config:
    dataset: str
    project: ProjectConfig
    data: DataConfig
    spark: SparkConfig
    training: TrainingConfig
    models: dict[str, dict[str, Any]]
    artifacts: ArtifactsConfig


def _as_path(value: str | Path) -> Path:
    return Path(value)


def load_config(path: Path = Path("configs/default.yaml")) -> Config:
    with path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    return Config(
        dataset=raw.get("dataset", "german_credit"),
        project=ProjectConfig(**raw["project"]),
        data=DataConfig(
            openml_dataset_id=raw["data"]["openml_dataset_id"],
            openml_dataset_name=raw["data"]["openml_dataset_name"],
            target_column=raw["data"]["target_column"],
            raw_path=_as_path(raw["data"]["raw_path"]),
            processed_dir=_as_path(raw["data"]["processed_dir"]),
            test_size=raw["data"]["test_size"],
            split_strategy=raw["data"].get("split_strategy", "stratified"),
            kaggle_dataset=raw["data"].get("kaggle_dataset"),
            kaggle_filename=raw["data"].get("kaggle_filename"),
            feature_whitelist=tuple(raw["data"].get("feature_whitelist", [])),
        ),
        spark=SparkConfig(
            **raw.get("spark", {"driver_memory": "4g", "shuffle_partitions": 8})
        ),
        training=TrainingConfig(
            cv_folds=raw["training"]["cv_folds"],
            n_jobs=raw["training"]["n_jobs"],
            scoring=raw["training"]["scoring"],
            max_train_rows=raw["training"].get("max_train_rows"),
            random_forest_sample_size=raw["training"].get("random_forest_sample_size"),
            shap_sample_size=raw["training"].get("shap_sample_size", 200),
        ),
        models=raw["models"],
        artifacts=ArtifactsConfig(
            dir=_as_path(raw["artifacts"]["dir"]),
            plots_dir=_as_path(raw["artifacts"]["plots_dir"]),
            best_model_path=_as_path(raw["artifacts"]["best_model_path"]),
            metrics_path=_as_path(raw["artifacts"]["metrics_path"]),
            cv_results_path=_as_path(raw["artifacts"]["cv_results_path"]),
            test_metrics_path=_as_path(raw["artifacts"]["test_metrics_path"]),
            threshold_path=_as_path(raw["artifacts"]["threshold_path"]),
            feature_importance_path=_as_path(raw["artifacts"]["feature_importance_path"]),
        ),
    )


def ensure_directories(config: Config) -> None:
    config.data.raw_path.parent.mkdir(parents=True, exist_ok=True)
    config.data.processed_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts.dir.mkdir(parents=True, exist_ok=True)
    config.artifacts.plots_dir.mkdir(parents=True, exist_ok=True)
