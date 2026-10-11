# Voyage Analytics – Integrating MLOps in Travel
*Productionization of ML Systems* – capstone project (team of 3)

Three datasets (users, flights, hotels) → three ML models → a production-style MLOps stack.

```
 CSV data ─► validate ─► train (sklearn/XGBoost) ─► MLflow tracking + registry ─► models/*.joblib
                                                                                   │
   Airflow DAG (weekly) ── orchestrates ──────────────────────────────────────────┤
   Jenkins / GitHub Actions ── test → train → gate → docker build → kubectl ──────┤
                                                                                   ▼
   Streamlit dashboard  ◄── HTTP ──  Flask REST API (gunicorn) in Docker ── Kubernetes (2–8 pods, HPA)
```

## Objectives → where they live
| # | Objective | Location |
|---|-----------|----------|
| 1 | Flight-price regression | `src/voyage/train_regression.py`, `notebooks/02_*` |
| 2 | Flask REST API | `api/app.py` |
| 3 | Docker | `Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml` |
| 4 | Kubernetes | `k8s/` (namespace, deployment, service, HPA, dashboard) |
| 5 | Airflow workflow* | `airflow/dags/voyage_training_dag.py` |
| 6 | CI/CD (Jenkins)* | `Jenkinsfile`, `.github/workflows/ci.yml` |
| 7 | MLflow tracking | `src/voyage/tracking.py` (used by every trainer) |
| 8 | Gender classification | `src/voyage/train_gender.py`, `notebooks/03_*` |
| 9 | Recommendation + Streamlit | `src/voyage/recommender.py`, `streamlit_app/app.py`, `notebooks/04_*` |

\*The deck screenshots cut off objectives 5–6; these are implemented as Airflow + Jenkins. Adjust if your brief differs.

## Quick start
Requires **Python 3.11+**. Library versions are pinned so the shipped `models/*.joblib` load correctly; `make train` regenerates them anyway.
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put users.csv, hotels.csv, flights.csv in data/raw/  (already included in the zip)
export PYTHONPATH=src                                  # Windows PowerShell: $env:PYTHONPATH="src"
make train        # trains 3 models, logs to MLflow, writes models/, runs the quality gate (~3-5 min on 1 CPU)
make test         # 19 tests
make serve        # API  → http://localhost:5000
make ui           # dashboard → http://localhost:8501   (new terminal)
make mlflow-ui    # experiments + registry → http://localhost:5001
```
No `make`? Run the commands inside the Makefile directly.

### API examples
```bash
curl -X POST localhost:5000/predict/price -H 'Content-Type: application/json' \
  -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"firstClass","agency":"FlyingDrops"}'
curl -X POST localhost:5000/recommend -H 'Content-Type: application/json' \
  -d '{"userCode":7,"k":3,"destination":"Natal (RN)","maxPrice":300}'
curl -X POST localhost:5000/predict/gender -H 'Content-Type: application/json' -d '{"age":35,"company":"4You"}'
curl localhost:5000/similar/Hotel%20A ; curl localhost:5000/metadata ; curl localhost:5000/health
```

### Docker
```bash
make train                       # models/ must exist before building the API image
docker compose up --build        # API :5000, dashboard :8501, MLflow :5001
```

### Kubernetes (minikube or Docker Desktop)
```bash
make docker-build
minikube image load voyage-api:latest voyage-dashboard:latest   # minikube only
kubectl apply -f k8s/
kubectl -n voyage get pods,svc,hpa
minikube service -n voyage voyage-api --url                     # or http://localhost:30500 on Docker Desktop
# load test to watch the autoscaler (needs metrics-server: `minikube addons enable metrics-server`)
kubectl -n voyage run load --rm -it --image=busybox -- sh -c 'while true; do wget -q -O- --post-data="{\"from\":\"Recife (PE)\",\"to\":\"Florianopolis (SC)\",\"flightType\":\"economic\",\"agency\":\"CloudFy\"}" --header="Content-Type: application/json" http://voyage-api:5000/predict/price; done'
```

### Airflow
Copy/mount the project at `/opt/airflow/voyage-analytics` (or set `VOYAGE_HOME`) and put `airflow/dags/voyage_training_dag.py` in your DAGs folder. Trigger `voyage_training_pipeline`.

### Jenkins
Create a Pipeline job from SCM pointing at this repo. Needs Docker, kubectl and Python 3.11 on the agent, and a `docker-registry-url` credential (or remove that line from the `environment` block).

## Models & results (see `docs/PROJECT_REPORT.md`)
| Model | Champion | Held-out result |
|---|---|---|
| Flight price | XGBoost | RMSE ≈ 0.01, R² = 1.000 (price is a deterministic table of route × class × agency) |
| Gender | HistGradientBoosting | CV accuracy ≈ 33% – **at chance**; reported honestly |
| Hotel recs | Hybrid item-CF + popularity + destination boost | hit@3 0.368 vs popularity 0.349 vs random 0.333 |

## Layout
```
api/ src/voyage/ streamlit_app/ tests/ notebooks/ k8s/ airflow/ docs/ scripts/ models/ data/raw/
```

## Verification status
Verified in development: training pipeline, quality gate, 19 unit/API tests, gunicorn serving, Streamlit pages.
**Not executed in the authoring sandbox** (no Docker/kubectl/Airflow available): Dockerfiles, compose, k8s manifests, the Airflow DAG and the Jenkinsfile – written to standard patterns and syntax-checked only. Run them on your machine and record screenshots for the report.
