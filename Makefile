.PHONY: infra infra-down dev worker test lint eval seed migrate clean

infra:
	docker compose up -d

infra-down:
	docker compose down

dev: infra
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	uv run arq app.worker.WorkerSettings

test:
	uv run pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy app/ cli/

eval:
	uv run corpus eval run --dataset evals/golden.jsonl --strategy hybrid_rerank

seed:
	uv run corpus admin seed

migrate:
	uv run alembic upgrade head

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
