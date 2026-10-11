"""Flask REST API serving the three Voyage Analytics models (Objective 2)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from voyage import config
from voyage import recommender as _rec  # noqa: F401  (needed to unpickle HotelRecommender)
from voyage.train_gender import CLASSES, FEATURES as GENDER_FEATURES

log = logging.getLogger("voyage.api")

FLIGHT_TYPES = ["economic", "premium", "firstClass"]
AGENCIES = ["CloudFy", "FlyingDrops", "Rainbow"]
COMPANIES = ["4You", "Acme Factory", "Monsters CYA", "Umbrella LTDA", "Wonka Company"]


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def _json_body() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError("Request body must be a JSON object.")
    return body


def _require(body: dict, *names: str):
    missing = [n for n in names if n not in body or body[n] in (None, "")]
    if missing:
        raise ApiError(f"Missing required field(s): {', '.join(missing)}")


def create_app(model_dir: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    mdir = Path(model_dir) if model_dir else config.MODEL_DIR
    state: dict = {"started": time.time(), "requests": 0}

    def load(name):
        p = mdir / name
        if not p.exists():
            raise FileNotFoundError(f"{p} missing – run `make train` first.")
        return joblib.load(p)

    state["flight"] = load(config.FLIGHT_MODEL_PATH.name)
    state["gender"] = load(config.GENDER_MODEL_PATH.name)
    state["recommender"] = load(config.RECOMMENDER_PATH.name)
    state["routes"] = json.loads((mdir / config.ROUTE_TABLE_PATH.name).read_text())
    mp = mdir / config.METRICS_PATH.name
    state["metrics"] = json.loads(mp.read_text()) if mp.exists() else {}
    app.config["STATE"] = state

    @app.before_request
    def _count():
        state["requests"] += 1

    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        return jsonify(error=e.message), e.status

    @app.errorhandler(404)
    def _404(_):
        return jsonify(error="Not found"), 404

    @app.errorhandler(405)
    def _405(_):
        return jsonify(error="Method not allowed"), 405

    @app.errorhandler(Exception)
    def _500(e):
        log.exception("Unhandled error")
        return jsonify(error="Internal server error"), 500

    # ------------------------------------------------------------------ meta
    @app.get("/health")
    def health():
        return jsonify(status="ok", uptime_s=round(time.time() - state["started"], 1))

    @app.get("/ready")
    def ready():
        ok = all(k in state for k in ("flight", "gender", "recommender"))
        return (jsonify(ready=ok), 200 if ok else 503)

    @app.get("/stats")
    def stats():
        return jsonify(requests=state["requests"], uptime_s=round(time.time() - state["started"], 1))

    @app.get("/metadata")
    def metadata():
        cities = sorted({k.split("|")[0] for k in state["routes"]})
        rec = state["recommender"]
        return jsonify(cities=cities, flight_types=FLIGHT_TYPES, agencies=AGENCIES,
                       companies=COMPANIES, hotels=rec.hotel_info_.reset_index().to_dict("records"),
                       metrics=state["metrics"])

    # ----------------------------------------------------------- flight price
    @app.post("/predict/price")
    def predict_price():
        b = _json_body()
        _require(b, "from", "to", "flightType", "agency")
        if b["flightType"] not in FLIGHT_TYPES:
            raise ApiError(f"flightType must be one of {FLIGHT_TYPES}")
        if b["agency"] not in AGENCIES:
            raise ApiError(f"agency must be one of {AGENCIES}")
        route = state["routes"].get(f"{b['from']}|{b['to']}")
        if route is None:
            raise ApiError(f"Unknown route {b['from']} -> {b['to']}. See /metadata for valid cities.")
        X = pd.DataFrame([{"from": b["from"], "to": b["to"], "flightType": b["flightType"],
                           "agency": b["agency"], "distance": route["distance"]}])
        price = float(state["flight"].predict(X)[0])
        return jsonify(predicted_price=round(price, 2), currency="BRL",
                       distance_km=route["distance"], flight_time_h=route["time"])

    # ---------------------------------------------------------------- gender
    @app.post("/predict/gender")
    def predict_gender():
        b = _json_body()
        _require(b, "age", "company")
        try:
            age = float(b["age"])
        except (TypeError, ValueError):
            raise ApiError("age must be a number")
        if not 0 < age < 120:
            raise ApiError("age must be between 1 and 119")
        if b["company"] not in COMPANIES:
            raise ApiError(f"company must be one of {COMPANIES}")
        row = {f: 0.0 for f in GENDER_FEATURES}
        row.update({k: b[k] for k in GENDER_FEATURES if k in b})
        row["age"], row["company"] = age, b["company"]
        proba = state["gender"].predict_proba(pd.DataFrame([row]))[0]
        classes = list(state["gender"].classes_)
        probs = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        label = max(probs, key=probs.get)
        return jsonify(predicted_gender=label, probabilities=probs,
                       confidence=round(probs[label], 4),
                       warning="Model is near chance level on this dataset (see docs/PROJECT_REPORT.md); "
                               "do not use for decisions about individuals.")

    # --------------------------------------------------------- recommendation
    @app.post("/recommend")
    def recommend():
        b = _json_body() if request.data else {}
        k = int(b.get("k", 3))
        if not 1 <= k <= 9:
            raise ApiError("k must be between 1 and 9")
        user = b.get("userCode")
        if user is not None:
            try:
                user = int(user)
            except (TypeError, ValueError):
                raise ApiError("userCode must be an integer")
        recs = state["recommender"].recommend(
            user_code=user, k=k, destination=b.get("destination"),
            exclude_visited=bool(b.get("excludeVisited", False)), max_price=b.get("maxPrice"))
        known = user in state["recommender"].user_index_ if user is not None else False
        return jsonify(recommendations=recs, cold_start=not known)

    @app.get("/similar/<hotel>")
    def similar(hotel):
        try:
            return jsonify(hotel=hotel, similar=state["recommender"].similar_hotels(
                hotel, int(request.args.get("k", 3))))
        except KeyError as e:
            raise ApiError(str(e.args[0]), 404)

    return app


def get_app() -> Flask:  # gunicorn entrypoint: gunicorn "api.app:get_app()"
    logging.basicConfig(level=logging.INFO)
    return create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="0.0.0.0", port=5000)
