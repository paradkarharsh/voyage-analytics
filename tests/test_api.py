import pytest

from api.app import create_app


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


ROUTE = {"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "economic", "agency": "CloudFy"}


def test_health_and_ready(client):
    assert client.get("/health").json["status"] == "ok"
    assert client.get("/ready").json["ready"] is True


def test_price_prediction_is_sensible(client):
    r = client.post("/predict/price", json=ROUTE)
    assert r.status_code == 200
    body = r.get_json()
    assert 300 < body["predicted_price"] < 1800
    assert body["distance_km"] > 0


def test_first_class_costs_more_than_economic(client):
    eco = client.post("/predict/price", json=ROUTE).json["predicted_price"]
    first = client.post("/predict/price", json={**ROUTE, "flightType": "firstClass"}).json["predicted_price"]
    assert first > eco


@pytest.mark.parametrize("patch,msg", [
    ({"flightType": "luxury"}, "flightType"),
    ({"agency": "Nope"}, "agency"),
    ({"to": "Atlantis"}, "Unknown route"),
])
def test_price_validation(client, patch, msg):
    r = client.post("/predict/price", json={**ROUTE, **patch})
    assert r.status_code == 400 and msg in r.json["error"]


def test_price_missing_fields(client):
    r = client.post("/predict/price", json={"from": "Recife (PE)"})
    assert r.status_code == 400 and "Missing" in r.json["error"]


def test_bad_body(client):
    assert client.post("/predict/price", data="not json", content_type="text/plain").status_code == 400


def test_gender(client):
    r = client.post("/predict/gender", json={"age": 30, "company": "4You"})
    assert r.status_code == 200
    assert set(r.json["probabilities"]) == {"female", "male", "none"}
    assert abs(sum(r.json["probabilities"].values()) - 1) < 1e-3
    assert client.post("/predict/gender", json={"age": 500, "company": "4You"}).status_code == 400


def test_recommend_existing_and_cold_start(client):
    r = client.post("/recommend", json={"userCode": 0, "k": 3}).json
    assert len(r["recommendations"]) == 3 and r["cold_start"] is False
    cold = client.post("/recommend", json={"k": 2}).json
    assert cold["cold_start"] is True and len(cold["recommendations"]) == 2


def test_recommend_destination_boost(client):
    r = client.post("/recommend", json={"userCode": 5, "destination": "Salvador (BH)", "k": 1}).json
    assert r["recommendations"][0]["place"] == "Salvador (BH)"


def test_recommend_budget_filter(client):
    r = client.post("/recommend", json={"k": 9, "maxPrice": 150}).json
    assert r["recommendations"] and all(x["price_per_night"] <= 150 for x in r["recommendations"])


def test_similar_hotels(client):
    assert len(client.get("/similar/Hotel A?k=2").json["similar"]) == 2
    assert client.get("/similar/Nope").status_code == 404
