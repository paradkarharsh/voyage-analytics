"""Data validation + post-training quality gate (used by Airflow, Jenkins and CI)."""
from __future__ import annotations

import json
import logging
import sys

from . import config, data

log = logging.getLogger("voyage.validate")

# Thresholds a candidate model must clear before it is allowed to ship.
MIN_FLIGHT_R2 = 0.95
MIN_HIT3_VS_RANDOM = 1.0   # hybrid recommender must beat random top-3


def validate_data() -> None:
    """Raise AssertionError with a readable message if the raw data looks wrong."""
    users, flights, hotels = data.load_users(), data.load_flights(), data.load_hotels()
    assert len(flights) > 1000, "flights.csv suspiciously small"
    assert flights["price"].gt(0).all(), "non-positive flight prices"
    assert flights[["from", "to", "flightType", "agency", "distance"]].notna().all().all(), \
        "nulls in flight feature columns"
    assert set(flights["flightType"]) <= {"economic", "premium", "firstClass"}, "unknown flightType"
    assert hotels["days"].between(1, 60).all(), "hotel 'days' out of range"
    assert (hotels["total"] - hotels["days"] * hotels["price"]).abs().max() < 0.05, \
        "hotel total != days * price"
    assert set(flights["userCode"]) <= set(users["code"]), "flights reference unknown users"
    assert set(hotels["userCode"]) <= set(users["code"]), "hotels reference unknown users"
    log.info("Data validation passed (%d users, %d flights, %d hotel stays)",
             len(users), len(flights), len(hotels))


def quality_gate() -> None:
    m = json.loads(config.METRICS_PATH.read_text())
    best = m["flight_price"]["best"]
    r2 = m["flight_price"]["results"][best]["r2"]
    assert r2 >= MIN_FLIGHT_R2, f"flight R2 {r2:.4f} < {MIN_FLIGHT_R2}"
    h3 = m["recommender"]["hit@3"]
    assert h3["hybrid"] > h3["random"] * MIN_HIT3_VS_RANDOM, \
        f"recommender hit@3 {h3['hybrid']} does not beat random {h3['random']:.3f}"
    log.info("Quality gate passed (flight R2=%.4f, hit@3=%.3f)", r2, h3["hybrid"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    which = sys.argv[1] if len(sys.argv) > 1 else "data"
    try:
        {"data": validate_data, "gate": quality_gate}[which]()
    except AssertionError as e:
        log.error("FAILED: %s", e)
        sys.exit(1)
