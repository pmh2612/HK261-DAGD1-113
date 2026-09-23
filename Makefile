.PHONY: help install neo4j-up neo4j-down lint format test check-neo4j

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install runtime + dev dependencies
	test -d $(VENV) || python3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

neo4j-up:  ## Start Neo4j in the background
	docker compose up -d

neo4j-down:  ## Stop Neo4j (the named volumes keep the data)
	docker compose down

lint:  ## Check code style
	$(VENV)/bin/ruff check .

format:  ## Auto-format and fix what ruff can fix
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

test:  ## Run the test suite
	$(VENV)/bin/pytest

check-neo4j:  ## Verify the app can connect to Neo4j
	$(PY) scripts/check_neo4j.py
