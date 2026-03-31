.PHONY: all install install-dev test format

all: install test slow-tests

install:
	uv python install 3.12
	uv venv --python 3.12
	uv pip install -e .

install-dev:
	uv python install 3.12
	uv venv --python 3.12
	uv pip install -e .[dev]

install-full:
	uv python install 3.12
	uv venv --python 3.12
	uv pip install -e .[full]

test:
	uv run pytest tests -m "not slow"

format:
	uv run ruff format
	uv run ruff check --fix
