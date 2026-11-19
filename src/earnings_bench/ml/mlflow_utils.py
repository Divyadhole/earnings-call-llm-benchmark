"""MLflow setup helpers — consistent experiment naming and tags across models."""

from __future__ import annotations

from contextlib import contextmanager

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

from earnings_bench.config import MLflowConfig


def configure_mlflow(config: MLflowConfig | None = None) -> None:
    if mlflow is None:
        raise ImportError("Install the `ml` extras to get mlflow")
    cfg = config or MLflowConfig.from_env()
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


@contextmanager
def start_run(run_name: str, model_tag: str, **tags: str):
    """Start an MLflow run with standard tags. Yields the active run."""
    configure_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model", model_tag)
        mlflow.set_tag("project", "earnings-bench")
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        yield run
