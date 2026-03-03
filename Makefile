PYTHON=python
PIP=pip

export PYTHONPATH=src

.PHONY: install download-data train run-api test mlflow-ui lint docker-build docker-up docker-down

install:
	$(PIP) install -r requirements.txt

download-data:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) scripts/train_model.py

run-api:
	uvicorn churn_mlops.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000

docker-build:
	docker build -t saas-churn-mlops:latest .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
