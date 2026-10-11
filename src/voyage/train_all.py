"""Run the full training pipeline: python -m voyage.train_all  (or `make train`)."""
from __future__ import annotations

import argparse
import json
import logging

import mlflow

from . import config, recommender, tracking, train_gender, train_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voyage.train_all")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-register", action="store_true", help="skip MLflow Model Registry")
    ap.add_argument("--metrics-only", action="store_true",
                    help="merge per-model metrics files into models/metrics.json (used by Airflow)")
    args = ap.parse_args(argv)
    if args.metrics_only:
        out = tracking.merge_partials()
        config.METRICS_PATH.write_text(json.dumps(out, indent=2))
        log.info("Merged metrics for: %s", list(out))
        return out
    reg = not args.no_register

    out = {"flight_price": train_regression.train(register=reg),
           "gender": train_gender.train(register=reg)}
    rec_metrics = recommender.train()
    tracking.setup()
    with mlflow.start_run(run_name="hotel_recommender"):
        mlflow.log_param("algorithm", "hybrid item-CF + popularity + destination boost")
        for k, v in rec_metrics.items():
            if isinstance(v, dict):
                for who, val in v.items():
                    mlflow.log_metric(f"{k}_{who}".replace("@", "_at_"), val)
        mlflow.log_metric("n_eval_users", rec_metrics["n_eval_users"])
    out["recommender"] = rec_metrics

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(out, indent=2))
    log.info("Training complete. Metrics -> %s", config.METRICS_PATH)
    return out


if __name__ == "__main__":
    main()
