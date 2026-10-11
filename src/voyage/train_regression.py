"""Objective 1 – flight price regression (+ MLflow tracking, Objective 7)."""
from __future__ import annotations

import json
import logging
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from . import config, data, tracking
from .tracking import save_partial

log = logging.getLogger(__name__)

CV_SAMPLE = 20_000


def make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), config.FLIGHT_CATEGORICAL),
         ("num", "passthrough", config.FLIGHT_NUMERIC)]
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def candidate_models() -> dict:
    rs = config.RANDOM_STATE
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=40, max_depth=14, min_samples_leaf=5,
                                               n_jobs=-1, random_state=rs),
        "xgboost": XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.15, subsample=0.9,
                                n_jobs=-1, random_state=rs, tree_method="hist"),
    }


def evaluate(model, X, y) -> dict:
    p = model.predict(X)
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def train(register: bool = True) -> dict:
    flights = data.load_flights()
    X, y = flights[config.FLIGHT_FEATURES], flights["price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE,
                                          random_state=config.RANDOM_STATE)
    tracking.setup()
    results, fitted = {}, {}
    with mlflow.start_run(run_name="flight_price_regression"):
        mlflow.log_params({"n_rows": len(flights), "test_size": config.TEST_SIZE,
                           "features": ",".join(config.FLIGHT_FEATURES)})
        for name, est in candidate_models().items():
            with mlflow.start_run(run_name=name, nested=True):
                pipe = make_pipeline(est)
                t0 = time.time()
                # CV on a subsample keeps runtime sane; the final fit uses the full train split.
                sub = Xtr.sample(n=min(CV_SAMPLE, len(Xtr)), random_state=config.RANDOM_STATE).index
                cv_r2 = cross_val_score(pipe, Xtr.loc[sub], ytr.loc[sub], cv=3, scoring="r2").mean()
                pipe.fit(Xtr, ytr)
                fit_s = time.time() - t0
                m = evaluate(pipe, Xte, yte)
                m.update(cv_r2=float(cv_r2), train_seconds=fit_s)
                mlflow.log_params({"algorithm": name, **{k: v for k, v in est.get_params().items()
                                                         if isinstance(v, (int, float, str, bool))}})
                mlflow.log_metrics(m)
                results[name], fitted[name] = m, pipe
                log.info("%s -> %s", name, {k: round(v, 4) for k, v in m.items()})

        best = min(results, key=lambda k: results[k]["rmse"])
        mlflow.log_param("best_model", best)
        mlflow.log_metrics({f"best_{k}": v for k, v in results[best].items()})
        with mlflow.start_run(run_name=f"champion_{best}", nested=True):
            mlflow.log_params({"algorithm": best, "role": "champion"})
            mlflow.log_metrics(results[best])
            tracking.log_sklearn_model(fitted[best],
                                       "flight-price-regressor" if register else None,
                                       input_example=Xte.head(3))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], config.FLIGHT_MODEL_PATH, compress=3)
    config.ROUTE_TABLE_PATH.write_text(json.dumps(data.build_route_table(flights), indent=2))
    out = {"best": best, "results": results}
    save_partial("flight_price", out)
    return out
