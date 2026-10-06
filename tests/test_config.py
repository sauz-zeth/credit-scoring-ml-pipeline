from __future__ import annotations

from pathlib import Path

from credit_scoring.config import load_config


def test_dataset_configs_load_independent_settings() -> None:
    german = load_config(Path("configs/default.yaml"))
    lending = load_config(Path("configs/lending_club.yaml"))

    assert german.dataset == "german_credit"
    assert german.data.split_strategy == "stratified"
    assert lending.dataset == "lending_club"
    assert lending.project.precision_floor == 0.35
    assert lending.data.split_strategy == "stratified"
    assert "earliest_cr_line" in lending.data.feature_whitelist
    assert not any(
        name.startswith(("total_pymnt", "total_rec_", "hardship_", "settlement_"))
        for name in lending.data.feature_whitelist
    )
