"""Objective 8 – gender classification.

NOTE (honest reporting): on this dataset the target is essentially unpredictable from the
available features; accuracy sits at/below the majority-class baseline. We report that
plainly (see docs/PROJECT_REPORT.md) and expose calibrated probabilities from the API.
"""
from __future__ import annotations

import logging

import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, data, tracking
from .tracking import save_partial

log = logging.getLogger(__name__)

CATEGORICAL = ["company"]
NUMERIC = ["age", "n_flights", "avg_flight_price", "avg_distance", "pct_first_class",
           "pct_economic", "n_agencies", "n_destinations", "n_hotel_stays", "avg_stay_days",
           "avg_hotel_spend", "n_hotels"]
FEATURES = CATEGORICAL + NUMERIC
CLASSES = ["female", "male", "none"]


def make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                             ("num", StandardScaler(), NUMERIC)])
    return Pipeline([("pre", pre), ("model", estimator)])


def candidate_models() -> dict:
    rs = config.RANDOM_STATE
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10,
                                                class_weight="balanced", random_state=rs, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=rs),
    }


def train(register: bool = True) -> dict:
    users, flights, hotels = data.load_users(), data.load_flights(), data.load_hotels()
    df = data.user_behaviour_features(users, flights, hotels)
    X, y = df[FEATURES], df["gender"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                                          random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    tracking.setup()
    results, fitted = {}, {}
    with mlflow.start_run(run_name="gender_classification"):
        mlflow.log_params({"n_users": len(df), "classes": ",".join(CLASSES)})
        for name, est in candidate_models().items():
            with mlflow.start_run(run_name=name, nested=True):
                pipe = make_pipeline(est)
                cvr = cross_validate(pipe, Xtr, ytr, cv=cv, n_jobs=1,
                                     scoring=["accuracy", "f1_macro"])
                pipe.fit(Xtr, ytr)
                p = pipe.predict(Xte)
                m = {"cv_accuracy": float(cvr["test_accuracy"].mean()),
                     "cv_f1_macro": float(cvr["test_f1_macro"].mean()),
                     "test_accuracy": float(accuracy_score(yte, p)),
                     "test_f1_macro": float(f1_score(yte, p, average="macro"))}
                mlflow.log_param("algorithm", name)
                mlflow.log_metrics(m)
                results[name], fitted[name] = m, pipe
                log.info("%s -> %s", name, {k: round(v, 3) for k, v in m.items()})

        contenders = {k: v for k, v in results.items() if k != "baseline_majority"}
        best = max(contenders, key=lambda k: contenders[k]["cv_f1_macro"])
        beats = results[best]["test_accuracy"] > results["baseline_majority"]["test_accuracy"]
        mlflow.log_params({"best_model": best, "beats_majority_baseline": beats})
        with mlflow.start_run(run_name=f"champion_{best}", nested=True):
            mlflow.log_metrics(results[best])
            tracking.log_sklearn_model(fitted[best],
                                       "gender-classifier" if register else None,
                                       input_example=Xte.head(3))

    joblib.dump(fitted[best], config.GENDER_MODEL_PATH, compress=3)
    out = {"best": best, "beats_baseline": bool(beats), "results": results}
    save_partial("gender", out)
    return out
