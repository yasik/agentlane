.PHONY: init sync sync-upgrade format lint lint-python lint-static tests tree

init:
	uv sync --all-extras

sync:
	uv sync --all-extras

sync-upgrade:
	uv lock --upgrade
	uv sync --all-extras

format:
	uv run isort src packages tests
	uv run black src packages tests

lint: lint-python lint-static

lint-python:
	uv run isort --check-only src packages tests
	uv run black --check src packages tests
	uv run ruff check src packages tests
	uv run pyright

lint-static:
	uv run yamllint -c .yamllint.yaml .
	@if command -v markdownlint >/dev/null 2>&1; then \
		markdownlint "**/*.md" --config .markdownlint.yaml --ignore docs/plans/**; \
	else \
		echo "markdownlint not installed; skipping markdown lint"; \
	fi

tests:
	uv run pytest

tree:
	find . -maxdepth 4 -type d | sort
