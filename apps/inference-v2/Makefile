.PHONY: help install install-dev test test-unit test-integration lint typecheck format check all clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install production dependencies"
	@echo "  install-dev      - Install all dependencies including dev and test"
	@echo "  test             - Run all tests (unit + integration)"
	@echo "  test-unit        - Run unit tests only (fast, mocked)"
	@echo "  test-integration - Run integration tests (slow, real server)"
	@echo "  lint             - Run ruff linter"
	@echo "  typecheck        - Run pyright type checker"
	@echo "  format           - Format code with ruff"
	@echo "  check            - Run all checks (lint + typecheck)"
	@echo "  all              - Install deps, run checks, and run tests"
	@echo "  clean            - Clean up temporary files"

# Installation targets
install:
	uv sync

install-dev:
	uv sync --group dev --group test

# Testing targets
test: test-unit test-integration

test-unit:
	@echo "Running unit tests..."
	uv run pytest tests/ -k "not integration" -v

test-integration:
	@echo "Running integration tests (this may take several minutes)..."
	@echo "Starting server and loading model..."
	uv run pytest tests/test_chat_completion_integration.py -v -s --tb=short

# Code quality targets
lint:
	@echo "Running ruff linter..."
	uv run ruff check .

typecheck:
	@echo "Running pyright type checker..."
	uv run pyright .

format:
	@echo "Formatting code with ruff..."
	uv run ruff format .

# Combined check target
check: lint typecheck

# Run everything
all: install-dev check test

# Cleanup
clean:
	@echo "Cleaning up temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
