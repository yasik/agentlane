.PHONY: init sync sync-upgrade format lint lint-python lint-static tests tree release

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

release:
	@set -eu; \
		branch=$$(git branch --show-current); \
		if [ "$$branch" != "main" ]; then \
			echo "Release must be cut from main. Current branch: $$branch"; \
			exit 1; \
		fi; \
		if [ -n "$$(git status --short)" ]; then \
			echo "Release requires a clean working tree."; \
			git status --short; \
			exit 1; \
		fi; \
		if ! command -v gh >/dev/null 2>&1; then \
			echo "GitHub CLI (gh) is required for make release."; \
			exit 1; \
		fi; \
		if [ -n "$${TAG:-}" ]; then \
			tag="$$TAG"; \
		else \
			tag=$$(git tag --list --sort=-creatordate | head -n 1); \
			if [ -z "$$tag" ]; then \
				echo "No local tags found. Pass TAG=vX.Y.Z."; \
				exit 1; \
			fi; \
		fi; \
		notes_file="docs/releases/$${tag}.md"; \
		if ! git rev-parse "$$tag" >/dev/null 2>&1; then \
			echo "Local tag not found: $$tag"; \
			exit 1; \
		fi; \
		if [ ! -f "$$notes_file" ]; then \
			echo "Release notes file not found: $$notes_file"; \
			exit 1; \
		fi; \
		if gh release view "$$tag" >/dev/null 2>&1; then \
			echo "GitHub release already exists for $$tag"; \
			exit 1; \
		fi; \
		echo "Pushing tag $$tag to origin..."; \
		git push origin "$$tag"; \
		echo "Creating GitHub release $$tag from $$notes_file..."; \
		gh release create "$$tag" \
			--verify-tag \
			--title "$$tag" \
			--notes-file "$$notes_file"
