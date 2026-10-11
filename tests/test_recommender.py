import joblib

from voyage import config
from voyage import recommender as r


def test_recommender_roundtrip_and_scores():
    rec = joblib.load(config.RECOMMENDER_PATH)
    out = rec.recommend(user_code=1, k=9)
    assert len(out) == 9 and out == sorted(out, key=lambda x: -x["score"])


def test_exclude_visited():
    rec = joblib.load(config.RECOMMENDER_PATH)
    uid = next(u for u, i in rec.user_index_.items() if (rec.matrix_[i] == 0).any())
    seen = {rec.hotels_[j] for j in range(len(rec.hotels_)) if rec.matrix_[rec.user_index_[uid], j] > 0}
    assert not seen & {x["hotel"] for x in rec.recommend(uid, k=9, exclude_visited=True)}


def test_hybrid_not_worse_than_random():
    m = joblib.load(config.RECOMMENDER_PATH) and __import__("json").loads(config.METRICS_PATH.read_text())
    h3 = m["recommender"]["hit@3"]
    assert h3["hybrid"] >= h3["random"]
