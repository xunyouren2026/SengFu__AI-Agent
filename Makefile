# =============================================================================
# AGI Unified Framework - Makefile
# =============================================================================
# Common development commands for the project
# Usage: make <target>
# =============================================================================

.PHONY: help install install-dev install-all clean test lint format \
        docker-build docker-run docker-stop serve train evaluate \
        docs check type-check security-audit benchmark \
        k8s-deploy k8s-rollback k8s-logs

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
PYTHON       ?= python3
PIP          ?= pip3
DOCKER       ?= docker
DOCKER_COMPOSE ?= docker-compose
KUBECTL      ?= kubectl
IMAGE_NAME   ?= agi-unified-framework
IMAGE_TAG    ?= latest
PORT         ?= 8000

# Colors for output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
NC     := \033[0m

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help: ## Show this help message
	@echo "$(BLUE)AGI Unified Framework - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install: ## Install the package with core dependencies
	@echo "$(GREEN)Installing AGI Unified Framework...$(NC)"
	$(PIP) install -e .

install-dev: ## Install with development dependencies (linting, formatting, testing)
	@echo "$(GREEN)Installing with development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"

install-all: ## Install with all optional dependencies
	@echo "$(GREEN)Installing with all dependencies...$(NC)"
	$(PIP) install -e ".[all]"

install-vision: ## Install with vision dependencies
	$(PIP) install -e ".[vision]"

install-audio: ## Install with audio dependencies
	$(PIP) install -e ".[audio]"

install-rag: ## Install with RAG dependencies
	$(PIP) install -e ".[rag]"

install-federated: ## Install with federated learning dependencies
	$(PIP) install -e ".[federated]"

install-monitoring: ## Install with monitoring dependencies
	$(PIP) install -e ".[monitoring]"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ testing/ -v --tb=short

test-fast: ## Run tests excluding slow and integration tests
	$(PYTHON) -m pytest tests/ testing/ -v -m "not slow and not integration and not e2e" --tb=short

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/ -v --tb=short

test-integration: ## Run integration tests
	$(PYTHON) -m pytest testing/integration/ -v --tb=short

test-e2e: ## Run end-to-end tests
	$(PYTHON) -m pytest testing/e2e/ -v --tb=short

test-gpu: ## Run GPU-specific tests
	$(PYTHON) -m pytest tests/ testing/ -v -m gpu --tb=short

test-coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ testing/ -v --cov=agi_unified_framework \
		--cov-report=html --cov-report=term-missing --cov-fail-under=60

test-benchmark: ## Run performance benchmarks
	$(PYTHON) -m pytest testing/performance/ -v --benchmark-only

# -----------------------------------------------------------------------------
# Linting & Formatting
# -----------------------------------------------------------------------------
lint: ## Run linter (ruff)
	@echo "$(GREEN)Running linter...$(NC)"
	$(PYTHON) -m ruff check agi_unified_framework/ tests/ testing/

lint-fix: ## Run linter and auto-fix issues
	$(PYTHON) -m ruff check --fix agi_unified_framework/ tests/ testing/

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	$(PYTHON) -m black agi_unified_framework/ tests/ testing/
	$(PYTHON) -m isort agi_unified_framework/ tests/ testing/

format-check: ## Check code formatting without modifying
	$(PYTHON) -m black --check agi_unified_framework/ tests/ testing/
	$(PYTHON) -m isort --check-only agi_unified_framework/ tests/ testing/

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checker...$(NC)"
	$(PYTHON) -m mypy agi_unified_framework/ --ignore-missing-imports

check: lint format-check type-check ## Run all checks (lint + format + type)

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f deployment/docker/Dockerfile .

docker-build-prod: ## Build production Docker image
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f deployment/docker/Dockerfile \
		--target production .

docker-run: ## Run Docker container
	@echo "$(GREEN)Starting Docker container...$(NC)"
	$(DOCKER) run --gpus all -p $(PORT):8000 \
		-v $(PWD)/models:/app/models:ro \
		-v $(PWD)/config:/app/config:ro \
		-v $(PWD)/logs:/app/logs \
		--env-file .env \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-stop: ## Stop all running containers
	$(DOCKER) compose -f deployment/docker/docker-compose.yml down

docker-up: ## Start services with docker-compose
	$(DOCKER_COMPOSE) -f deployment/docker/docker-compose.yml up -d

docker-down: ## Stop services with docker-compose
	$(DOCKER_COMPOSE) -f deployment/docker/docker-compose.yml down

docker-logs: ## View Docker container logs
	$(DOCKER_COMPOSE) -f deployment/docker/docker-compose.yml logs -f

docker-restart: ## Restart Docker containers
	$(DOCKER_COMPOSE) -f deployment/docker/docker-compose.yml restart

# -----------------------------------------------------------------------------
# Kubernetes
# -----------------------------------------------------------------------------
k8s-deploy: ## Deploy to Kubernetes
	@echo "$(GREEN)Deploying to Kubernetes...$(NC)"
	$(KUBECTL) apply -f deployment/kubernetes/namespace.yaml
	$(KUBECTL) apply -f deployment/kubernetes/configmap.yaml
	$(KUBECTL) apply -f deployment/kubernetes/secret.yaml
	$(KUBECTL) apply -f deployment/kubernetes/deployment.yaml
	$(KUBECTL) apply -f deployment/kubernetes/service.yaml
	$(KUBECTL) apply -f deployment/kubernetes/hpa.yaml
	$(KUBECTL) apply -f deployment/kubernetes/ingress.yaml

k8s-rollback: ## Rollback Kubernetes deployment
	$(KUBECTL) rollout undo deployment/agi-unified-api -n agi-framework

k8s-restart: ## Restart Kubernetes deployment
	$(KUBECTL) rollout restart deployment/agi-unified-api -n agi-framework

k8s-status: ## Check Kubernetes deployment status
	$(KUBECTL) get all -n agi-framework

k8s-logs: ## View Kubernetes pod logs
	$(KUBECTL) logs -f deployment/agi-unified-api -n agi-framework

k8s-delete: ## Delete all Kubernetes resources
	$(KUBECTL) delete -f deployment/kubernetes/ --ignore-not-found

# -----------------------------------------------------------------------------
# Serve & Run
# -----------------------------------------------------------------------------
serve: ## Start the API server (development mode)
	@echo "$(GREEN)Starting API server on port $(PORT)...$(NC)"
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --reload

serve-prod: ## Start the API server (production mode)
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --workers 4

train: ## Run training script
	$(PYTHON) scripts/train.py

evaluate: ## Run evaluation script
	$(PYTHON) scripts/evaluate.py

benchmark: ## Run benchmark script
	$(PYTHON) scripts/benchmark.py

download-models: ## Download pre-trained models
	$(PYTHON) scripts/download_models.py

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	$(PYTHON) -m sphinx docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# -----------------------------------------------------------------------------
# Security
# -----------------------------------------------------------------------------
security-audit: ## Run security audit on dependencies
	$(PIP) install pip-audit && $(PYTHON) -m pip_audit

security-scan: ## Run security scan on source code
	$(PYTHON) -m bandit -r agi_unified_framework/ -ll

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean: ## Remove all generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "$(GREEN)Clean complete.$(NC)"

clean-all: clean ## Deep clean (also remove logs, outputs, and cache)
	rm -rf logs/ output/ outputs/ results/ .cache/ 2>/dev/null || true
	@echo "$(GREEN)Deep clean complete.$(NC)"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup-env: ## Create .env file from example
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)Created .env from .env.example$(NC)"; \
	else \
		echo "$(YELLOW).env already exists, skipping.$(NC)"; \
	fi

# Default target
.DEFAULT_GOAL := help
