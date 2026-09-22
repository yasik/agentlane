.DEFAULT_GOAL := help
.PHONY: help install dev build preview format lint typecheck test check clean

help:
	@echo "install dev build preview format lint typecheck test check clean"

install:
	pnpm install --frozen-lockfile

dev:
	pnpm run dev --host 127.0.0.1

build:
	pnpm run build

preview:
	pnpm run preview --host 127.0.0.1

format:
	pnpm run format

lint:
	pnpm run lint

typecheck:
	pnpm run typecheck

test: build
	pnpm test

check: lint typecheck test

clean:
	rm -rf dist .astro
