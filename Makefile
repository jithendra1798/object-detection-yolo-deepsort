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

clean:
	rm -rf ${VENV} __pycache__ uploads processed
