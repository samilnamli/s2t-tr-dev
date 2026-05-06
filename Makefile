PROJECT_NAME = s2t-tr-dev
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run python


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint with ruff
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format


# ----------------------------------------------------------------------------
# Run an experiment
#   make run CONFIG=main_results_ami
#   make run CONFIG=main_results_ami OVERRIDES="trainer.max_epochs=2 data.batch_size=8"
# ----------------------------------------------------------------------------

CONFIG ?=
OVERRIDES ?=

## Run an experiment by name (CONFIG=<name>, optional OVERRIDES="k=v ...")
.PHONY: run
run:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Usage: make run CONFIG=<name> [OVERRIDES=\"key=val ...\"]"; \
		echo "Available: $$(ls configs/experiment | sed 's/.yaml$$//' | tr '\n' ' ')"; \
		exit 1; \
	fi
	$(PYTHON_INTERPRETER) run.py experiment=$(CONFIG) $(OVERRIDES)


# ----------------------------------------------------------------------------
# Local smoke test (runtime-generated synthetic parquet, max_epochs=1)
# ----------------------------------------------------------------------------

## Run the local end-to-end smoke test
.PHONY: smoke
smoke:
	uv run pytest tests/test_smoke.py -v


# ----------------------------------------------------------------------------
# Materialize a dataset's parquet explicitly
#   make prepare_data DATASET=ami
#   make prepare_data DATASET=voxpopuli
#   make prepare_data DATASET=synthetic
# ----------------------------------------------------------------------------

DATASET ?=

## Materialize a dataset's parquet (DATASET=ami|voxpopuli|synthetic)
.PHONY: prepare_data
prepare_data:
	@if [ -z "$(DATASET)" ]; then \
		echo "Usage: make prepare_data DATASET=<ami|voxpopuli|synthetic>"; exit 1; \
	fi
	$(PYTHON_INTERPRETER) -m src.data.prepare --dataset $(DATASET)


# ----------------------------------------------------------------------------
# Reproduce a logged MLflow run (checkout commit + apply patch + re-run)
#   make reproduce RUN_ID=<mlflow_run_id>
#   make reproduce RUN_ID=<mlflow_run_id> FORCE=1   # allow dirty tree
# ----------------------------------------------------------------------------

RUN_ID ?=
FORCE ?=

## Reproduce a logged MLflow parent run by id
.PHONY: reproduce
reproduce:
	@if [ -z "$(RUN_ID)" ]; then echo "Usage: make reproduce RUN_ID=<id> [FORCE=1]"; exit 1; fi
	$(PYTHON_INTERPRETER) -m src.experiments.reproduce $(RUN_ID) $(if $(FORCE),--force,)


# ----------------------------------------------------------------------------
# Self-documenting help
# ----------------------------------------------------------------------------

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
lines = '\n'.join([line for line in sys.stdin])
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines)
print('Available rules:\n')
print('\n'.join(['{:25}{}'.format(*reversed(m)) for m in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
