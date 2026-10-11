# Suggested split of work (team of 3)

| Member | Owns | Deliverables to demo / explain in the viva |
|---|---|---|
| **A – Data & Modelling** | Objectives 1, 8, 9 | EDA notebook, regression, gender classifier (and why it is at chance), recommender + offline evaluation |
| **B – Serving & Containers** | Objectives 2, 3, 4, 9 (UI) | Flask API, Dockerfiles/compose, Kubernetes deployment + autoscaling demo, Streamlit app |
| **C – MLOps Automation** | Objectives 5, 6, 7 | MLflow tracking/registry walkthrough, Airflow DAG run, Jenkins/GitHub Actions pipeline, data validation + quality gate |

Everyone should be able to explain the full architecture diagram in the README and the three "honest findings" in the report
(deterministic prices, gender at chance, small recommender lift). Screenshots to capture: MLflow runs table + registered models,
`kubectl get pods,hpa`, Airflow graph view, Jenkins stage view, Streamlit pages.
