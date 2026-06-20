.PHONY: install dev test lint typecheck clean setup-ui run-api run-orch bench

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -v tests/

lint:
	ruff check anvil/ tests/

typecheck:
	mypy anvil/

clean:
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

setup-ui:
	pip install -e ".[ui]"

run-api:
	uvicorn anvil.api.router:app --host 127.0.0.1 --port 8080 --reload

run-orch:
	python -m anvil.core.orchestrator

bench:
	pytest tests/benchmarks/ -v --benchmark-only

setup: install
	./scripts/anvil_setup.sh
