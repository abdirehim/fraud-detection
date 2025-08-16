# Makefile for fraud detection project development

.PHONY: help install install-dev test test-unit test-integration test-coverage test-performance lint format type-check security clean run-pipeline

# Default target
help:
	@echo "Available targets:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies and setup pre-commit"
	@echo "  test             Run all tests with coverage"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  test-coverage    Run tests and generate coverage report"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security         Run security checks"
	@echo "  clean            Clean up generated files"
	@echo "  run-pipeline     Run the fraud detection pipeline"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Testing targets
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/ -v -m "unit or not integration" --cov=src

test-integration:
	pytest tests/ -v -m "integration" --cov=src

test-performance:
	pytest tests/ -v -m "performance" --cov=src

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=xml --cov-fail-under=90
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality targets
lint:
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports --no-strict-optional

format:
	black src/ tests/ --line-length=88
	isort src/ tests/ --profile=black --line-length=88

type-check:
	mypy src/ --ignore-missing-imports --no-strict-optional

security:
	bandit -r src/ -f json -o bandit-report.json
	safety check

# Cleanup targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -f bandit-report.json

# Pipeline execution
run-pipeline:
	python run.py --verbose

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make format' to format code"
	@echo "Run 'make lint' to check code quality"
