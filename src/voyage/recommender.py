"""Objective 9 – hotel recommendation (hybrid item-CF + popularity + destination boost)."""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from . import config, data
from .tracking import save_partial

log = logging.getLogger(__name__)


class HotelRecommender:
    """Recommends hotels for a user.

    score(h) = w_cf * cosine-CF(h | user history) + w_pop * popularity(h)
               (+ destination boost if the user's next city is known)

    Cold-start users (no history) get popularity, optionally narrowed by destination / budget.
    """

    def __init__(self, w_cf: float = 0.7, w_pop: float = 0.3, dest_boost: float = 10.0):
        self.w_cf, self.w_pop, self.dest_boost = w_cf, w_pop, dest_boost

    # ------------------------------------------------------------------ fit
    def fit(self, hotels: pd.DataFrame) -> "HotelRecommender":
        info = (hotels.groupby("name")
                      .agg(place=("place", "first"), price_per_night=("price", "first"),
                           bookings=("total", "size"), avg_days=("days", "mean"))
                      .sort_index())
        self.hotel_info_ = info
        self.hotels_ = list(info.index)
        idx = {h: i for i, h in enumerate(self.hotels_)}

        # user x hotel booking-count matrix
        m = hotels.groupby(["userCode", "name"]).size().unstack(fill_value=0)
        m = m.reindex(columns=self.hotels_, fill_value=0)
        self.user_index_ = {u: i for i, u in enumerate(m.index)}
        self.matrix_ = m.to_numpy(dtype=float)

        # item-item cosine similarity
        M = self.matrix_
        norms = np.linalg.norm(M, axis=0)
        norms[norms == 0] = 1
        self.sim_ = (M.T @ M) / np.outer(norms, norms)
        np.fill_diagonal(self.sim_, 0.0)

        pop = info["bookings"].to_numpy(dtype=float)
        self.popularity_ = pop / pop.sum()
        self.place_to_hotel_ = {p: h for h, p in info["place"].items()}
        return self

    # ------------------------------------------------------------ inference
    def _scores(self, user_code: int | None, destination: str | None) -> np.ndarray:
        pop = self.popularity_ / self.popularity_.max()
        if user_code is not None and user_code in self.user_index_:
            u = self.matrix_[self.user_index_[user_code]]
            u = u / u.sum() if u.sum() else u
            cf = self.sim_ @ u
            cf = cf / cf.max() if cf.max() > 0 else cf
            s = self.w_cf * cf + self.w_pop * pop
            # blend in the user's own affinity so favourites still rank high
            s = s + 0.5 * (u / u.max() if u.max() > 0 else u)
        else:
            s = pop.copy()
        if destination and destination in self.place_to_hotel_:
            s[self.hotels_.index(self.place_to_hotel_[destination])] += self.dest_boost
        return s

    def recommend(self, user_code: int | None = None, k: int = 3, destination: str | None = None,
                  exclude_visited: bool = False, max_price: float | None = None) -> list[dict]:
        s = self._scores(user_code, destination)
        info = self.hotel_info_
        mask = np.ones(len(s), dtype=bool)
        if max_price is not None:
            mask &= (info["price_per_night"].to_numpy() <= max_price)
        if exclude_visited and user_code in self.user_index_:
            mask &= (self.matrix_[self.user_index_[user_code]] == 0)
        s = np.where(mask, s, -np.inf)
        order = [i for i in np.argsort(-s) if np.isfinite(s[i])][:k]
        return [{"hotel": self.hotels_[i], "place": info.iloc[i]["place"],
                 "price_per_night": round(float(info.iloc[i]["price_per_night"]), 2),
                 "score": round(float(s[i]), 4)} for i in order]

    def similar_hotels(self, hotel: str, k: int = 3) -> list[dict]:
        if hotel not in self.hotels_:
            raise KeyError(f"Unknown hotel: {hotel}")
        i = self.hotels_.index(hotel)
        order = np.argsort(-self.sim_[i])[:k]
        return [{"hotel": self.hotels_[j], "place": self.hotel_info_.iloc[j]["place"],
                 "similarity": round(float(self.sim_[i, j]), 4)} for j in order]


# ---------------------------------------------------------------------- eval
def evaluate_leave_last_out(hotels: pd.DataFrame, ks=(1, 3, 5), min_stays: int = 4) -> dict:
    """Hold out each user's most recent stay; can we rank it in the top-k?

    Compared against popularity and random baselines.
    """
    h = hotels.sort_values(["userCode", "date"])
    counts = h.groupby("userCode").size()
    eligible = counts[counts >= min_stays].index
    last_idx = h[h.userCode.isin(eligible)].groupby("userCode").tail(1).index
    train, test = h.drop(last_idx), h.loc[last_idx]

    rec = HotelRecommender().fit(train)
    pop_rank = list(rec.hotel_info_.sort_values("bookings", ascending=False).index)
    n_hotels = len(rec.hotels_)
    out = {f"hit@{k}": {"hybrid": 0.0, "popularity": 0.0, "random": k / n_hotels} for k in ks}
    for _, r in test.iterrows():
        ranked = [x["hotel"] for x in rec.recommend(r.userCode, k=max(ks))]
        for k in ks:
            out[f"hit@{k}"]["hybrid"] += r["name"] in ranked[:k]
            out[f"hit@{k}"]["popularity"] += r["name"] in pop_rank[:k]
    for k in ks:
        for key in ("hybrid", "popularity"):
            out[f"hit@{k}"][key] = round(out[f"hit@{k}"][key] / len(test), 4)
    out["n_eval_users"] = int(len(test))
    return out


def train() -> dict:
    hotels = data.load_hotels()
    metrics = evaluate_leave_last_out(hotels)
    rec = HotelRecommender().fit(hotels)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rec, config.RECOMMENDER_PATH, compress=3)
    save_partial("recommender", metrics)
    return metrics
