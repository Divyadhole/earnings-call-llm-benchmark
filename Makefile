.PHONY: install install-all test lint format dbt-run dbt-test universe clean

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,dbt,llm,agent,ml]"

test:
	pytest -v

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

universe:
	python -m earnings_bench.ingest.universe

dbt-run:
	cd dbt_project && dbt seed && dbt run

dbt-test:
	cd dbt_project && dbt test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf dbt_project/target dbt_project/dbt_packages dbt_project/logs
