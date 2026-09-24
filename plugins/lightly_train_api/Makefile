.PHONY: install format type-check

install:
	uv venv
	uv pip install -r ../../dev-requirements.txt
	uv pip install -e .

format:
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	uv run mypy --config-file ../../mypy.ini .
