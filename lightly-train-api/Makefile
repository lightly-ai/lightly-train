.PHONY: install
install:
	uv sync

.PHONY: run
run:
	uv run python -m lightly_train_api

.PHONY: test
test:
	uv run pytest

.PHONY: check
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src tests
