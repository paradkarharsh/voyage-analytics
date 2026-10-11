"""Objective 5 – Apache Airflow: weekly retraining pipeline for Voyage Analytics.

validate_data -> [train_flight_price, train_gender, train_recommender] -> quality_gate
              -> build_docker_image -> deploy_to_kubernetes

Set env VOYAGE_HOME to the project root on the Airflow worker (default /opt/airflow/voyage-analytics).
Compatible with Airflow 2.x and 3.x.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator

HOME = os.getenv("VOYAGE_HOME", "/opt/airflow/voyage-analytics")
ENV = f"cd {HOME} && export PYTHONPATH={HOME}/src"

default_args = {
    "owner": "voyage-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

with DAG(
    dag_id="voyage_training_pipeline",
    description="Validate data, retrain models, gate on quality, rebuild + roll out the API",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "voyage", "travel"],
) as dag:

    validate_data = BashOperator(task_id="validate_data",
                                 bash_command=f"{ENV} && python -m voyage.validate data")

    # One training entrypoint logs all three models to MLflow; separate tasks keep the DAG
    # readable and let you re-run pieces. Each task trains only its own model.
    train_flight_price = BashOperator(
        task_id="train_flight_price",
        bash_command=f"{ENV} && python -c \"from voyage import train_regression as t; print(t.train()['best'])\"")

    train_gender = BashOperator(
        task_id="train_gender",
        bash_command=f"{ENV} && python -c \"from voyage import train_gender as t; print(t.train()['best'])\"")

    train_recommender = BashOperator(
        task_id="train_recommender",
        bash_command=f"{ENV} && python -c \"from voyage import recommender as r; print(r.train())\"")

    # Aggregate the three results into models/metrics.json for the gate + API /metadata
    write_metrics = BashOperator(
        task_id="write_metrics",
        bash_command=f"{ENV} && python -m voyage.train_all --metrics-only",
    )

    quality_gate = BashOperator(task_id="quality_gate",
                                bash_command=f"{ENV} && python -m voyage.validate gate")

    build_docker_image = BashOperator(
        task_id="build_docker_image",
        bash_command=f"cd {HOME} && docker build -t voyage-api:{{{{ ds_nodash }}}} -t voyage-api:latest .")

    deploy_to_kubernetes = BashOperator(
        task_id="deploy_to_kubernetes",
        bash_command=(f"cd {HOME} && kubectl apply -f k8s/ && "
                      "kubectl -n voyage rollout restart deployment/voyage-api && "
                      "kubectl -n voyage rollout status deployment/voyage-api --timeout=180s"))

    validate_data >> [train_flight_price, train_gender, train_recommender] >> write_metrics
    write_metrics >> quality_gate >> build_docker_image >> deploy_to_kubernetes
