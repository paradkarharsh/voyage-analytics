# Voyage Analytics: Integrating MLOps in Travel – Project Report

## 1. Abstract
This project builds and productionises machine-learning models on three travel datasets (users, flights, hotels): a flight-price regressor, a gender classifier and a hotel recommender. The models are trained with experiment tracking (MLflow), served through a Flask REST API, containerised with Docker, deployed on Kubernetes with autoscaling, retrained by an Apache Airflow DAG and shipped through a Jenkins CI/CD pipeline. A Streamlit dashboard exposes the deployed models. Beyond the engineering, the project's main analytical finding is *what the data does and does not support*.

## 2. Data
| Table | Rows | Key columns |
|---|---|---|
| users | 1,340 | code, company (5), gender (male 452 / female 448 / none 440), age 21–65 |
| flights | 271,888 | from, to (9 cities), flightType (3), agency (3), price, time, distance, date |
| hotels | 40,552 | name (9 hotels), place, days (1–4), price/night, total, date |

No missing values or duplicates. Validation (`voyage.validate`) additionally checks schema, ranges, referential integrity (all flights/hotels map to a known user) and `total = days × price`.

**Structural findings (EDA):** (1) `time` ≡ constant × `distance` (r = 1.00); (2) one distance per route; (3) flight price has **zero variance within each of the 490 (route, class, agency) combinations**; (4) each hotel has exactly one city and one price, and the hotel city equals the flight destination of the same trip in 100% of trips; (5) FlyingDrops sells only first-class tickets, so its higher average price is a product-mix effect; within first class agencies differ by under 1%.

## 3. Methods and results
### 3.1 Flight price regression (Objective 1)
Features: from, to, flightType, agency, distance (one-hot + passthrough in a scikit-learn Pipeline). 80/20 split.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear regression | 103.17 | 81.29 | 0.919 |
| Random forest | 5.70 | 1.55 | 0.9998 |
| **XGBoost (champion)** | **0.0085** | 0.0063 | 1.000 |

The near-perfect fit is explained by the deterministic price table – not leakage (no target-derived feature). Ablation shows flightType and agency dominate; date features add nothing.

### 3.2 Gender classification (Objective 8)
One row per user; age, company and aggregated flight/hotel behaviour (13 features). Stratified 5-fold CV + 20% hold-out.

| Model | CV accuracy | CV macro-F1 | Test accuracy |
|---|---|---|---|
| Majority baseline | 0.338 | 0.168 | 0.336 |
| Logistic regression | 0.329 | 0.326 | 0.358 |
| Random forest | 0.324 | 0.321 | 0.358 |
| **HistGradientBoosting (selected by CV macro-F1)** | 0.332 | 0.330 | 0.343 |

A permutation test gives p = 0.76 (real-label CV accuracy 0.318 vs 0.329 with shuffled labels): **the models perform at chance level.** Hypothesis tests in the EDA agree (gender vs spend p = 0.26, gender vs company p = 0.37). The model is nonetheless deployed to demonstrate the pipeline, with probabilities and an explicit warning returned by the API. Predicting gender from behaviour is also an ethically sensitive practice; production systems should rely on self-declared data.

### 3.3 Hotel recommender (Objective 9)
Hybrid of item-based collaborative filtering (cosine similarity over user × hotel booking counts), popularity, user affinity, and a destination boost; cold-start falls back to popularity; budget and exclude-visited filters. Evaluation: hold out each user's latest stay (users with ≥ 4 stays, n = 1,245).

| | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| Hybrid | 0.120 | 0.368 | 0.612 |
| Popularity | 0.122 | 0.349 | 0.598 |
| Random | 0.111 | 0.333 | 0.556 |

The hybrid−popularity difference at hit@3 is +0.018 with a 95% bootstrap CI of [−0.014, +0.051] – **not statistically established**. Users have tried ~7.4 of the 9 hotels, so history carries little signal. The genuinely useful behaviour is destination-aware recommendation (hotel city ↔ flight destination is one-to-one), the budget filter and similar-hotel lookup.

## 4. MLOps architecture
* **Tracking (Obj. 7):** every candidate model is a nested MLflow run (params, CV and test metrics, timings); champions are logged with an input example and registered (`flight-price-regressor`, `gender-classifier`). Local SQLite backend by default; `MLFLOW_TRACKING_URI` points it at a server.
* **API (Obj. 2):** Flask endpoints `/predict/price`, `/predict/gender`, `/recommend`, `/similar/<hotel>`, `/metadata`, `/health`, `/ready`, `/stats` with validation and JSON errors; served by gunicorn. The price API needs only from/to/class/agency – distance is looked up from the route table.
* **Docker (Obj. 3):** slim Python image, non-root user, health check; compose file starts API, dashboard and an MLflow server.
* **Kubernetes (Obj. 4):** 2-replica Deployment with readiness/liveness probes and resource limits, NodePort Service, and an HPA scaling 2–8 pods at 60% CPU.
* **Airflow (Obj. 5):** weekly DAG – validate data → train three models in parallel → merge metrics → quality gate → docker build → rolling deploy.
* **CI/CD (Obj. 6):** Jenkins pipeline (setup → validate/test → train → gate → build → container smoke test → deploy on `main`); GitHub Actions runs a lighter CI.
* **Quality gate:** flight R² ≥ 0.95 and recommender hit@3 above random; the gender model is excluded from gating because it has no signal to protect.

## 5. Limitations and honest caveats
1. Prices are a fixed table, so R² ≈ 1 says little about generalisation to a real airline.
2. The data window and demand pattern (peak in 2020) look synthetic; avoid real-world claims.
3. Gender is not predictable from the available features.
4. The recommender's offline lift over popularity is within noise.
5. The Docker, Kubernetes, Airflow and Jenkins artefacts were not executed in the authoring environment; test and screenshot them in your own.

## 6. Future work
Drift monitoring (Evidently/Prometheus) on request features; a real model-serving registry hook (load the `Production` alias from MLflow); Helm chart; authentication and rate limiting on the API; richer recommender features if real preference data becomes available.
