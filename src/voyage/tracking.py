"""Thin MLflow helpers so training code stays readable."""
from __future__ import annotations

import logging

import mlflow

from . import config

log = logging.getLogger(__name__)


def setup() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def log_sklearn_model(model, registered_name: str | None = None, signature=None, input_example=None):
    """Log a model, tolerating the artifact_path -> name rename between MLflow 2.x and 3.x."""
    import mlflow.sklearn as ms
    kwargs = dict(sk_model=model, registered_model_name=registered_name,
                  signature=signature, input_example=input_example,
                  serialization_format="cloudpickle")  # skops (MLflow 3 default) rejects XGBoost types
    try:
        return ms.log_model(name="model", **kwargs)
    except TypeError:
        return ms.log_model(artifact_path="model", **kwargs)


def save_partial(part: str, metrics: dict) -> None:
    """Persist one model's metrics so Airflow tasks can train models independently."""
    import json
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODEL_DIR / f"metrics_{part}.json").write_text(json.dumps(metrics, indent=2))


def merge_partials() -> dict:
    import json
    out = {}
    for part in ("flight_price", "gender", "recommender"):
        f = config.MODEL_DIR / f"metrics_{part}.json"
        if f.exists():
            out[part] = json.loads(f.read_text())
    return out
