#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = s2t-tr-dev
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format


#################################################################################
# DATA                                                                          #
#################################################################################

## Download processed AMI dataset from Google Drive
.PHONY: download_ami
download_ami:
	uv run python -m src.data.get_processed -d ami

## Download processed VoxPopuli dataset from Google Drive (parked deliverable)
.PHONY: download_voxpopuli
download_voxpopuli:
	uv run python -m src.data.get_processed -d voxpopuli


#################################################################################
# EXPERIMENTS                                                                   #
#################################################################################
#
# Generic runner. The experiment YAML is the SSOT (see configs/README.md).
# Usage:
#   make run_experiment EXPERIMENT=ablation_loss
#   make run_experiment EXPERIMENT=ablation_architecture
#   make run_experiment EXPERIMENT=synthetic
#   make run_experiment EXPERIMENT=main_results_ami
#

EXPERIMENT ?=

## Run any experiment by name (set EXPERIMENT=<name>)
.PHONY: run_experiment
run_experiment:
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "ERROR: EXPERIMENT is unset. Usage: make run_experiment EXPERIMENT=ablation_loss"; \
		exit 1; \
	fi
	uv run python -m src.experiments.run experiments=$(EXPERIMENT)


## Render manuscript table for a finished experiment (set EXPERIMENT=<name>)
.PHONY: render_table
render_table:
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "ERROR: EXPERIMENT is unset."; exit 1; \
	fi
	uv run python -m src.reporting.tables render \
		--results reports/main_results/$(EXPERIMENT)/main_results.json \
		--output-dir reports/manuscript/figures/auto/$(EXPERIMENT)


## Render manuscript figures for a finished experiment (set EXPERIMENT=<name>)
.PHONY: render_figures
render_figures:
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "ERROR: EXPERIMENT is unset."; exit 1; \
	fi
	uv run python -m src.reporting.figures render \
		--results reports/main_results/$(EXPERIMENT)/main_results.json \
		--output-dir reports/manuscript/figures/auto/$(EXPERIMENT)


#################################################################################
# DEPRECATED ALIASES (kept for back-compat with parked notebooks; remove later) #
#################################################################################

.PHONY: run_main_results_ami
run_main_results_ami:
	$(MAKE) run_experiment EXPERIMENT=main_results_ami

.PHONY: run_main_results_voxpopuli
run_main_results_voxpopuli:
	$(MAKE) run_experiment EXPERIMENT=main_results_voxpopuli

.PHONY: run_main_results_voxpopuli_v2
run_main_results_voxpopuli_v2:
	$(MAKE) run_experiment EXPERIMENT=main_results_voxpopuli_v2

.PHONY: run_main_results_voxpopuli_v3
run_main_results_voxpopuli_v3:
	$(MAKE) run_experiment EXPERIMENT=main_results_voxpopuli_v3

.PHONY: run_main_results_voxpopuli_v4
run_main_results_voxpopuli_v4:
	$(MAKE) run_experiment EXPERIMENT=main_results_voxpopuli_v4


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
