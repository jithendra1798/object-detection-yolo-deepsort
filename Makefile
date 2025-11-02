.PHONY: help venv install dev-install run clean

VENV=.venv
PY=${VENV}/bin/python
PIP=${VENV}/bin/pip

help:
	@echo "Targets: venv, install, dev-install, run, clean"

venv:
	python3 -m venv ${VENV}

install: venv
	${PIP} install -U pip
	${PIP} install -r requirements.txt

dev-install: install
	${PIP} install -r requirements-dev.txt

run: install
	${PY} webapp.py

docker-build:
	docker build -t object-tracker:local .

docker-run:
	docker run --rm -p 5000:5000 -v $(pwd)/uploads:/app/uploads -v $(pwd)/processed:/app/processed --name object-tracker-local object-tracker:local

clean:
	rm -rf ${VENV} __pycache__ uploads processed
