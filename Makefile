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

.PHONY: mypy
mypy:
	uv run mypy .

.PHONY: pyright
pyright:
	uv run pyright --project pyrightconfig.json

.PHONY: typecheck
typecheck:
	@set -eu; \
	mypy_pid=''; \
	pyright_pid=''; \
	trap 'test -n "$$mypy_pid" && kill $$mypy_pid 2>/dev/null || true; test -n "$$pyright_pid" && kill $$pyright_pid 2>/dev/null || true' EXIT INT TERM; \
	echo "Running make mypy and make pyright in parallel..."; \
	$(MAKE) mypy & mypy_pid=$$!; \
	$(MAKE) pyright & pyright_pid=$$!; \
	wait $$mypy_pid; \
	wait $$pyright_pid; \
	trap - EXIT

tests:
	uv run pytest

tree:
	find . -maxdepth 4 -type d | sort
