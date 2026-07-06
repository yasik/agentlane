.PHONY: init sync sync-upgrade
.PHONY: format format-python format-ts
.PHONY: lint lint-python lint-static lint-ts
.PHONY: mypy pyright typecheck typecheck-python typecheck-ts
.PHONY: tests tests-python test-ts build-ts check-ts tree

TS_BRIDGE_DIR := packages/process_bridge_ts

init:
	$(MAKE) sync

sync:
	uv sync --all-extras
	cd $(TS_BRIDGE_DIR) && bun install

sync-upgrade:
	uv lock --upgrade
	uv sync --all-extras
	cd $(TS_BRIDGE_DIR) && bun update

format: format-python format-ts

format-python:
	uv run isort src packages tests .agents/skills/release/scripts
	uv run black src packages tests .agents/skills/release/scripts

format-ts:
	cd $(TS_BRIDGE_DIR) && bun run format

lint: lint-python lint-ts lint-static

lint-python:
	uv run isort --check-only src packages tests .agents/skills/release/scripts
	uv run black --check src packages tests .agents/skills/release/scripts
	uv run ruff check src packages tests .agents/skills/release/scripts
	uv run pyright

lint-static:
	uv run yamllint -c .yamllint.yaml .
	@if command -v markdownlint >/dev/null 2>&1; then \
		markdownlint "**/*.md" --config .markdownlint.yaml --ignore docs/plans/**; \
	else \
		echo "markdownlint not installed; skipping markdown lint"; \
	fi

lint-ts:
	cd $(TS_BRIDGE_DIR) && bun run lint

mypy:
	uv run mypy src packages tests

pyright:
	uv run pyright --project pyrightconfig.json

typecheck: typecheck-python typecheck-ts

typecheck-python:
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

typecheck-ts:
	cd $(TS_BRIDGE_DIR) && bun run typecheck

tests: tests-python test-ts

tests-python:
	uv run pytest

test-ts:
	cd $(TS_BRIDGE_DIR) && bun run test

build-ts:
	cd $(TS_BRIDGE_DIR) && bun run build

check-ts: lint-ts typecheck-ts test-ts build-ts

tree:
	find . -maxdepth 4 -type d | sort
