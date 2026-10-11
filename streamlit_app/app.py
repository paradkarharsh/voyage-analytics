"""Objective 9 – Streamlit dashboard on top of the deployed Voyage Analytics API."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:5000").rstrip("/")
DATA_DIR = Path(os.getenv("VOYAGE_DATA_DIR", Path(__file__).resolve().parents[1] / "data" / "raw"))

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")


# ------------------------------------------------------------------ helpers
def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=15, **kw)
        data = r.json()
        if r.status_code >= 400:
            st.error(data.get("error", f"API error {r.status_code}"))
            return None
        return data
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API_URL}. Start it with `make serve` or docker compose. ({e})")
        return None


@st.cache_data(ttl=300)
def meta():
    return api("GET", "/metadata")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


m = meta()
st.sidebar.title("✈️ Voyage Analytics")
page = st.sidebar.radio("Page", ["Flight price", "Hotel recommender", "Travel insights",
                                 "Gender classifier", "Model & MLOps"])
st.sidebar.caption(f"API: {API_URL}")
if not m:
    st.stop()

# ------------------------------------------------------------ flight price
if page == "Flight price":
    st.header("Flight price prediction")
    c1, c2 = st.columns(2)
    origin = c1.selectbox("From", m["cities"], index=0)
    dest = c2.selectbox("To", [c for c in m["cities"] if c != origin], index=0)
    c3, c4 = st.columns(2)
    ftype = c3.selectbox("Class", m["flight_types"])
    agency = c4.selectbox("Agency", m["agencies"])
    if st.button("Predict price", type="primary"):
        res = api("POST", "/predict/price", json={"from": origin, "to": dest,
                                                  "flightType": ftype, "agency": agency})
        if res:
            a, b, c = st.columns(3)
            a.metric("Predicted price", f"R$ {res['predicted_price']:,.2f}")
            b.metric("Distance", f"{res['distance_km']:.0f} km")
            c.metric("Flight time", f"{res['flight_time_h']:.2f} h")
            rows = []
            for t in m["flight_types"]:
                r = api("POST", "/predict/price", json={"from": origin, "to": dest,
                                                        "flightType": t, "agency": agency})
                if r:
                    rows.append({"class": t, "price": r["predicted_price"]})
            if rows:
                st.plotly_chart(px.bar(pd.DataFrame(rows), x="class", y="price", text_auto=".0f",
                                       title="Same route & agency, by class"), width="stretch")

# ------------------------------------------------------ hotel recommender
elif page == "Hotel recommender":
    st.header("Hotel recommendations")
    users = load_csv("users")
    c1, c2, c3 = st.columns(3)
    mode = c1.radio("Traveller", ["Existing user", "New user (cold start)"])
    user = None
    if mode == "Existing user":
        user = c1.number_input("User code", 0, int(users["code"].max()) if users is not None else 1339, 0)
    dest = c2.selectbox("Next destination (optional)", [None] + sorted({h["place"] for h in m["hotels"]}))
    k = c3.slider("How many?", 1, 9, 3)
    budget = c3.number_input("Max price / night (0 = any)", 0.0, 1000.0, 0.0)
    unvisited = c2.checkbox("Only hotels never visited by this user", value=False)
    if st.button("Recommend", type="primary"):
        body = {"k": k, "destination": dest, "excludeVisited": unvisited}
        if user is not None:
            body["userCode"] = int(user)
        if budget > 0:
            body["maxPrice"] = budget
        res = api("POST", "/recommend", json=body)
        if res:
            if res["cold_start"]:
                st.info("No history for this traveller – ranking by overall popularity.")
            df = pd.DataFrame(res["recommendations"])
            if df.empty:
                st.warning("No hotels match those filters.")
            else:
                st.dataframe(df, width="stretch", hide_index=True)
                st.plotly_chart(px.bar(df, x="hotel", y="score", color="place", title="Recommendation scores"),
                                width="stretch")
    st.subheader("Hotel catalogue")
    st.dataframe(pd.DataFrame(m["hotels"]), width="stretch", hide_index=True)
    h = st.selectbox("Find hotels similar to…", [x["name"] for x in m["hotels"]])
    sim = api("GET", f"/similar/{h}")
    if sim:
        st.dataframe(pd.DataFrame(sim["similar"]), hide_index=True)

# --------------------------------------------------------- travel insights
elif page == "Travel insights":
    st.header("Travel insights")
    flights, hotels, users = load_csv("flights"), load_csv("hotels"), load_csv("users")
    if flights is None or hotels is None:
        st.warning(f"Put flights.csv / hotels.csv in {DATA_DIR} to see charts.")
        st.stop()
    t1, t2, t3 = st.tabs(["Flights", "Hotels", "Demand over time"])
    with t1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.box(flights, x="flightType", y="price", color="agency",
                               title="Price by class and agency"), width="stretch")
        top = flights.groupby("to").size().sort_values(ascending=False).reset_index(name="flights")
        c2.plotly_chart(px.bar(top, x="to", y="flights", title="Most popular destinations"),
                        width="stretch")
        c1.plotly_chart(px.scatter(flights.sample(5000, random_state=1), x="distance", y="price",
                                   color="flightType", opacity=.5, title="Price vs distance"),
                        width="stretch")
    with t2:
        c1, c2 = st.columns(2)
        b = hotels.groupby("name").agg(bookings=("total", "size"), price=("price", "first")).reset_index()
        c1.plotly_chart(px.bar(b, x="name", y="bookings", color="price", title="Bookings per hotel"),
                        width="stretch")
        c2.plotly_chart(px.histogram(hotels, x="days", title="Length of stay (nights)"),
                        width="stretch")
    with t3:
        mth = (flights.assign(month=flights["date"].dt.to_period("M").dt.to_timestamp())
                      .groupby("month").size().reset_index(name="flights"))
        st.plotly_chart(px.line(mth, x="month", y="flights", title="Flights per month"),
                        width="stretch")

# ------------------------------------------------------ gender classifier
elif page == "Gender classifier":
    st.header("Gender classification")
    st.warning("On this dataset the model performs at roughly chance level – treat outputs as a "
               "demonstration of the deployment path, not as a reliable prediction.")
    c1, c2 = st.columns(2)
    age = c1.slider("Age", 18, 70, 35)
    company = c2.selectbox("Company", m["companies"])
    if st.button("Classify", type="primary"):
        res = api("POST", "/predict/gender", json={"age": age, "company": company})
        if res:
            st.metric("Predicted", res["predicted_gender"], f"confidence {res['confidence']:.0%}")
            st.plotly_chart(px.bar(x=list(res["probabilities"]), y=list(res["probabilities"].values()),
                                   labels={"x": "class", "y": "probability"}), width="stretch")

# --------------------------------------------------------------- MLOps page
else:
    st.header("Model performance & pipeline")
    fm = m["metrics"].get("flight_price")
    if fm:
        st.subheader(f"Flight price – champion: {fm['best']}")
        st.dataframe(pd.DataFrame(fm["results"]).T.round(4))
    gm = m["metrics"].get("gender")
    if gm:
        st.subheader(f"Gender – champion: {gm['best']}  (beats majority baseline: {gm['beats_baseline']})")
        st.dataframe(pd.DataFrame(gm["results"]).T.round(4))
    rm = m["metrics"].get("recommender")
    if rm:
        st.subheader("Recommender – hit rate (leave-last-out)")
        rows = [{"k": k, **v} for k, v in rm.items() if isinstance(v, dict)]
        st.dataframe(pd.DataFrame(rows).set_index("k"))
    st.caption("Experiments and the model registry are in the MLflow UI (`make mlflow-ui`).")
