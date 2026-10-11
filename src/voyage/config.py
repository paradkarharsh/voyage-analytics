"""Central configuration. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("VOYAGE_DATA_DIR", ROOT / "data" / "raw"))
MODEL_DIR = Path(os.getenv("VOYAGE_MODEL_DIR", ROOT / "models"))

FLIGHTS_CSV = DATA_DIR / "flights.csv"
HOTELS_CSV = DATA_DIR / "hotels.csv"
USERS_CSV = DATA_DIR / "users.csv"

FLIGHT_MODEL_PATH = MODEL_DIR / "flight_price_model.joblib"
GENDER_MODEL_PATH = MODEL_DIR / "gender_model.joblib"
RECOMMENDER_PATH = MODEL_DIR / "hotel_recommender.joblib"
ROUTE_TABLE_PATH = MODEL_DIR / "route_table.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

# MLflow: a local SQLite store by default so the Model Registry works with no server.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "voyage-analytics")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Flight model schema (the API contract)
FLIGHT_CATEGORICAL = ["from", "to", "flightType", "agency"]
FLIGHT_NUMERIC = ["distance"]
FLIGHT_FEATURES = FLIGHT_CATEGORICAL + FLIGHT_NUMERIC
