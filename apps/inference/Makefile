.PHONY: test-verbose coverage format test check-format check-ci install install-dev all run-server install-prerelease-tlens

# Variables
PYTHON = python
POETRY = poetry
PYTEST = $(POETRY) run pytest
COVERAGE = $(POETRY) run coverage
RUFF = $(POETRY) run ruff

# Targets
test:
	$(POETRY) run $(PYTEST) -v --cov=neuronpedia_inference/ --cov-report=term-missing --cov-branch tests

test-verbose:
	$(POETRY) run $(PYTEST) -vv --no-header

format:
	$(RUFF) format .
	$(RUFF) check --fix-only .

check-format:
	$(RUFF) check .
	$(RUFF) format --check .

check-type:
	$(POETRY) run pyright .

check-ci: check-format check-type test

install:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install

install-dev:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install --with dev

all: install-dev format check-format test coverage

run-server:
	$(POETRY) run python start.py

# TODO: Remove install-prerelease-tlens once we have a stable version of transformer-lens that is compatible with sae-lens
# We first install the packages from pyproject.toml. Then, we install sae-lens' dependencies separately since we are using the --no-deps flag.
# Finally, we install sae-lens with the --no-deps flag.
install-prerelease-tlens:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install	
	$(POETRY) run pip install babe datasets nltk plotly plotly-express pyyaml safetensors simple-parsing tenacity
	$(POETRY) run pip install sae-lens==6.6.4 --no-deps

	