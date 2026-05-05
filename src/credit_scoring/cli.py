from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from credit_scoring.config import load_config
from credit_scoring.data import download_openml_dataset
from credit_scoring.evaluate import evaluate_pipeline
from credit_scoring.explain import explain_pipeline
from credit_scoring.logging_utils import setup_logging
from credit_scoring.train import train_pipeline

app = typer.Typer(help="Credit scoring / fraud detection tabular ML pipeline.")
ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to YAML config.")]


@app.callback()
def callback() -> None:
    setup_logging()


@app.command("download-data")
def download_data(
    config_path: ConfigOption = Path("configs/default.yaml"),
    force: Annotated[bool, typer.Option(help="Re-download even if raw CSV exists.")] = False,
) -> None:
    config = load_config(config_path)
    path = download_openml_dataset(config, force=force)
    typer.echo(f"Saved dataset: {path}")


@app.command("train")
def train(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    metrics = train_pipeline(load_config(config_path))
    typer.echo(f"Best model: {metrics['best_model']}")


@app.command("evaluate")
def evaluate(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    metrics = evaluate_pipeline(load_config(config_path))
    typer.echo(metrics)


@app.command("explain")
def explain(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    importance = explain_pipeline(load_config(config_path))
    typer.echo(importance.head(10).to_string(index=False))


@app.command("run-all")
def run_all(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    config = load_config(config_path)
    download_openml_dataset(config)
    train_pipeline(config)
    evaluate_pipeline(config)
    explain_pipeline(config)
    typer.echo("Pipeline completed. See artifacts/ for outputs.")


if __name__ == "__main__":
    app()
