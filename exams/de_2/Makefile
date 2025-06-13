VENV_NAME=.venv
PYTHON=$(VENV_NAME)/bin/python
PIP=$(VENV_NAME)/bin/pip
AIRFLOW=$(VENV_NAME)/bin/airflow

init:
	python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(AIRFLOW) db init

requirements:
	$(PIP) freeze > requirements.txt

run-download:
	$(PYTHON) etl/download.py

run-preprocess:
	$(PYTHON) etl/preprocess.py

run-train:
	$(PYTHON) etl/train.py

run-evaluate:
	$(PYTHON) etl/evaluate.py

clean:
	rm -rf logs/*.log results/* clean_data/*

run-all: run-download run-preprocess run-train run-evaluate

start-airflow:
	$(AIRFLOW) webserver -p 8080 & \
	$(AIRFLOW) scheduler
