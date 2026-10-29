.PHONY: install test lint format dev up down logs

install:  ## Install the package with dev tools into the active virtualenv
	pip install -e ".[dev]"

test:  ## Run the test suite (Postgres tests run when SWARMOPS_TEST_DATABASE_URL is set)
	pytest -q -rs

lint:  ## Lint and check formatting
	ruff check .
	ruff format --check .

format:  ## Auto-fix lint issues and format
	ruff check --fix .
	ruff format .

dev:  ## Run the API locally with auto-reload
	uvicorn swarmops.api.app:app --reload

up:  ## Build and start API + Postgres in Docker
	docker compose up -d --build --wait

down:  ## Stop the stack and delete its volumes
	docker compose down -v

logs:  ## Follow API logs
	docker compose logs -f api
