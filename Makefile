.PHONY: install data-check train test serve ui mlflow-ui docker-build docker-up k8s-up k8s-down clean
export PYTHONPATH := src

install:        ## install python deps
	pip install -r requirements.txt
data-check:     ## validate the raw CSVs
	python -m voyage.validate data
train:          ## train all models, log to MLflow, write models/*.joblib
	python -m voyage.train_all
	python -m voyage.validate gate
test:
	pytest -q tests
serve:          ## dev API on :5000  (gunicorn in Docker)
	python -m api.app
ui:             ## Streamlit dashboard on :8501 (API must be running)
	API_URL=http://localhost:5000 streamlit run streamlit_app/app.py
mlflow-ui:      ## experiments + model registry on :5001
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
docker-build:
	docker build -t voyage-api:latest . && docker build -f Dockerfile.streamlit -t voyage-dashboard:latest .
docker-up:
	docker compose up --build
k8s-up:         ## needs minikube / docker-desktop k8s with images already built
	kubectl apply -f k8s/
k8s-down:
	kubectl delete -f k8s/ --ignore-not-found
clean:
	rm -rf mlruns mlflow.db models/*.joblib models/*.json __pycache__ .pytest_cache
