"""Data loading + cleaning with basic validation."""
from __future__ import annotations

import logging

import pandas as pd

from . import config

log = logging.getLogger(__name__)

DATE_FMT = "%m/%d/%Y"


def _read(path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put users.csv, hotels.csv, flights.csv in {config.DATA_DIR} "
            "or set VOYAGE_DATA_DIR."
        )
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df


def load_flights() -> pd.DataFrame:
    df = _read(config.FLIGHTS_CSV, {"travelCode", "userCode", "from", "to", "flightType",
                                    "price", "time", "distance", "agency", "date"})
    df["date"] = pd.to_datetime(df["date"], format=DATE_FMT)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_hotels() -> pd.DataFrame:
    df = _read(config.HOTELS_CSV, {"travelCode", "userCode", "name", "place", "days",
                                   "price", "total", "date"})
    df["date"] = pd.to_datetime(df["date"], format=DATE_FMT)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_users() -> pd.DataFrame:
    df = _read(config.USERS_CSV, {"code", "company", "name", "gender", "age"})
    return df.drop_duplicates().reset_index(drop=True)


def build_route_table(flights: pd.DataFrame) -> dict:
    """Route -> distance / typical duration. Lets the API accept just from/to."""
    g = flights.groupby(["from", "to"]).agg(distance=("distance", "mean"), time=("time", "mean"))
    return {f"{a}|{b}": {"distance": round(float(r.distance), 2), "time": round(float(r.time), 2)}
            for (a, b), r in g.iterrows()}


def user_behaviour_features(users: pd.DataFrame, flights: pd.DataFrame,
                            hotels: pd.DataFrame) -> pd.DataFrame:
    """One row per user with demographics + aggregated travel behaviour."""
    fl = flights.groupby("userCode").agg(
        n_flights=("price", "size"),
        avg_flight_price=("price", "mean"),
        avg_distance=("distance", "mean"),
        pct_first_class=("flightType", lambda s: (s == "firstClass").mean()),
        pct_economic=("flightType", lambda s: (s == "economic").mean()),
        n_agencies=("agency", "nunique"),
        n_destinations=("to", "nunique"),
    )
    ho = hotels.groupby("userCode").agg(
        n_hotel_stays=("total", "size"),
        avg_stay_days=("days", "mean"),
        avg_hotel_spend=("total", "mean"),
        n_hotels=("name", "nunique"),
    )
    df = (users.rename(columns={"code": "userCode"})
                .merge(fl, on="userCode", how="left")
                .merge(ho, on="userCode", how="left"))
    num = df.select_dtypes("number").columns.difference(["userCode", "age"])
    df[num] = df[num].fillna(0)
    return df
