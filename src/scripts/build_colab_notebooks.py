"""Build the four Colab reproducibility-package notebooks from a spec.

The notebooks under ``notebooks/colab/`` are not editable scratchpads;
they are the reproducibility package for each manuscript deliverable.
Per the project protocol they consist of:

    1. A title + ``experiment_metadata`` markdown header mirroring the
       YAML config.
    2. A clone + uv setup cell.
    3. A secrets-loading cell (``WANDB_API_KEY``, ``HF_TOKEN``).
    4. A data-fetch cell (``!make download_*`` or
       ``!uv run python -m src.data.synthetic ...``).
    5. The experiment-run cell:
       ``!uv run python -m src.experiments.run experiments=<name>``.
    6. The rendering cell that produces the manuscript table + figure
       and pushes them to W&B as an artifact.
    7. A final markdown cell embedding the rendered deliverable.

Generating the notebooks programmatically keeps them in lock-step with
the active config set and removes drift between hand-edited cells.

Usage:
    uv run python -m src.scripts.build_colab_notebooks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Notebook spec dataclass + cell builders
# --------------------------------------------------------------------------- #


@dataclass
class NotebookSpec:
    """Per-notebook information consumed by :func:`build_notebook`."""

    title: str
    subtitle: str
    experiment: str
    data_setup: List[str]
    extra_cells: List[dict]


def _markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# --------------------------------------------------------------------------- #
# Reusable cell snippets
# --------------------------------------------------------------------------- #


def _header_cell(spec: NotebookSpec) -> dict:
    body = (
        f"# {spec.title}\n"
        f"\n"
        f"{spec.subtitle}\n"
        f"\n"
        f"This notebook is a **reproducibility package**, not an exploratory\n"
        f"scratchpad. Cells contain only shell calls (`!uv run python -m ...` /\n"
        f"`!make ...`) and one final rendering cell that emits the\n"
        f"manuscript-grade table/figure for this experiment from `src.reporting`.\n"
        f"All artifacts are also pushed to W&B.\n"
        f"\n"
        f"Experiment config: [`configs/experiments/{spec.experiment}.yaml`](../"
        f"../configs/experiments/{spec.experiment}.yaml). The full operating\n"
        f"protocol is in [`PROJECT_STATE.md`](../../PROJECT_STATE.md).\n"
    )
    return _markdown(body)


def _setup_cells() -> List[dict]:
    """Cells that clone, install, and authenticate."""
    return [
        _markdown("## 1. Setup — clone, `uv` env, secrets\n"),
        _code(
            "%%bash\n"
            "set -euo pipefail\n"
            "if [ ! -d s2t-tr-dev ]; then\n"
            "  git clone https://github.com/huseyin-karaca/s2t-tr-dev\n"
            "fi\n"
            "cd s2t-tr-dev\n"
            "if ! command -v uv >/dev/null 2>&1; then\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  export PATH=\"$HOME/.cargo/bin:$PATH\"\n"
            "fi\n"
            "uv venv --python 3.10 --no-managed-python\n"
            "uv sync\n"
        ),
        _code("%cd s2t-tr-dev\n"),
        _code(
            "from src.scripts.colab_helpers import load_colab_secrets\n"
            "print(load_colab_secrets())\n"
        ),
    ]


def _run_cells(experiment: str, data_setup: List[str]) -> List[dict]:
    """Cells that fetch data and run the experiment via the generic runner."""
    cells: List[dict] = [_markdown("## 2. Data\n")]
    for cmd in data_setup:
        cells.append(_code(cmd))

    cells += [
        _markdown(
            f"## 3. Run the experiment\n"
            f"\n"
            f"All knobs live in `configs/experiments/{experiment}.yaml`. Per the\n"
            f"SSOT rule, do not pass overrides on the CLI for manuscript runs;\n"
            f"if a parameter must change, create `<name>_v3.yaml` (or higher).\n"
        ),
        _code(f"!uv run python -m src.experiments.run experiments={experiment}\n"),
    ]
    return cells


def _reporting_cells(experiment: str) -> List[dict]:
    """Cells that render the manuscript-grade table+figure and push to W&B."""
    return [
        _markdown(
            "## 4. Render manuscript deliverables\n"
            "\n"
            "Builds Markdown + LaTeX tables and the companion figures from\n"
            "the `main_results.json` aggregator written by step 3, then pushes\n"
            "everything to Weights & Biases as a `results-table` /\n"
            "`results-figures` artifact tied to the current git commit.\n"
        ),
        _code(
            f"!uv run python -m src.reporting.tables render \\\n"
            f"    --results reports/main_results/{experiment}/main_results.json \\\n"
            f"    --output-dir reports/manuscript/figures/auto/{experiment} \\\n"
            f"    --push-wandb\n"
        ),
        _code(
            f"!uv run python -m src.reporting.figures render \\\n"
            f"    --results reports/main_results/{experiment}/main_results.json \\\n"
            f"    --output-dir reports/manuscript/figures/auto/{experiment} \\\n"
            f"    --push-wandb\n"
        ),
        _markdown("## 5. Display the rendered deliverables\n"),
        _code(
            f"from src.scripts.colab_helpers import display_deliverables\n"
            f"display_deliverables('{experiment}')\n"
        ),
    ]


def _shutdown_cell() -> dict:
    return _code(
        "# Optional: free the Colab GPU once finished.\n"
        "# from google.colab import runtime\n"
        "# runtime.unassign()\n"
    )


# --------------------------------------------------------------------------- #
# Notebook-level builder
# --------------------------------------------------------------------------- #


def build_notebook(spec: NotebookSpec) -> dict:
    """Assemble one Jupyter notebook (nbformat 4) from a spec."""
    cells: List[dict] = [_header_cell(spec)]
    cells += _setup_cells()
    cells += _run_cells(spec.experiment, spec.data_setup)
    cells += spec.extra_cells
    cells += _reporting_cells(spec.experiment)
    cells.append(_shutdown_cell())

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# Specs for the four active deliverables
# --------------------------------------------------------------------------- #


SPECS: Sequence[NotebookSpec] = [
    NotebookSpec(
        title="Synthetic Regime-Switch Experiment",
        subtitle=(
            "Manuscript Section 4.1 — proves the temporal-modeling claim by "
            "training the proposed hierarchical transformer router and the "
            "MLP-pool baseline on a controlled regime-switch dataset where the "
            "best expert depends on the *ordered pair* of regimes within the clip."
        ),
        experiment="synthetic_v2",
        data_setup=[
            "!mkdir -p data/processed/synthetic_regime_switch\n"
            "!uv run python -m src.data.synthetic \\\n"
            "    --output-path data/processed/synthetic_regime_switch/combined_features.parquet \\\n"
            "    --num-samples 10000 --frame-length 128 --num-regimes 4 \\\n"
            "    --regime-dim 32 --noise-std 0.5 --feature-dtype float16 \\\n"
            "    --write-batch-size 500 --seed 42\n",
        ],
        extra_cells=[
            _markdown(
                "### 3a. (Optional) Sweep over $R$\n"
                "\n"
                "Generates the manuscript Fig. `synthetic_sweep` by re-running\n"
                "the pipeline at $R \\in \\{2,3,4,6,8\\}$. Comment in if you want\n"
                "the full sweep; defaults are tuned for a Colab T4.\n"
            ),
            _code(
                "# !uv run python -m src.experiments.sweep run \\\n"
                "#     --output-dir reports/sweeps/synthetic_R \\\n"
                "#     --r-values 2,3,4,6,8 \\\n"
                "#     --num-samples 5000 --frame-length 128 \\\n"
                "#     --max-epochs 30 --batch-size 32 --seed 42\n"
                "# !uv run python -m src.experiments.sweep figure \\\n"
                "#     --results reports/sweeps/synthetic_R/sweep_results.json \\\n"
                "#     --output-path reports/manuscript/figures/auto/synthetic_v2/synthetic_sweep.pdf\n"
            ),
        ],
    ),
    NotebookSpec(
        title="Loss Ablation (AMI)",
        subtitle=(
            "Manuscript Tab. III — sweeps the loss-coefficient triple "
            "$(\\lambda_{wer}, \\lambda_{hard}, \\lambda_{soft})$ on the AMI "
            "dataset to confirm that the proposed `WER + Soft CE` mix "
            "($\\tau=0.1$) wins."
        ),
        experiment="ablation_loss_v2",
        data_setup=["!make download_ami\n"],
        extra_cells=[],
    ),
    NotebookSpec(
        title="Architecture Ablation (AMI)",
        subtitle=(
            "Manuscript Tab. IV — toggles the cross-attention bridge, the "
            "Stage-1 weight sharing, and sweeps Stage-1/Stage-2 depth and "
            "$d_{model}$ to justify the default architecture."
        ),
        experiment="ablation_architecture_v2",
        data_setup=["!make download_ami\n"],
        extra_cells=[],
    ),
    NotebookSpec(
        title="Main Results — AMI",
        subtitle=(
            "Manuscript Tab. II (AMI column) — proposed router vs. MLP-pool "
            "baseline against training-free baselines (random / weighted / "
            "single-expert / oracle) and ROVER variants on the AMI dataset, "
            "averaged over 3 seeds."
        ),
        experiment="main_results_ami",
        data_setup=["!make download_ami\n"],
        extra_cells=[],
    ),
]


# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #


_FILENAMES = {
    "synthetic_v2": "s2t_tr_dev_synthetic_experiment.ipynb",
    "ablation_loss_v2": "s2t_tr_dev_ablation_loss.ipynb",
    "ablation_architecture_v2": "s2t_tr_dev_ablation_architecture.ipynb",
    "main_results_ami": "s2t_tr_dev_main_results_ami.ipynb",
}


def main(output_dir: Optional[Path] = None) -> List[Path]:
    """Build every spec to ``notebooks/colab/<name>.ipynb`` and return the paths."""
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = output_dir or repo_root / "notebooks" / "colab"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for spec in SPECS:
        nb = build_notebook(spec)
        target = out_dir / _FILENAMES[spec.experiment]
        target.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        written.append(target)
        print(f"wrote {target}")
    return written


if __name__ == "__main__":
    main()
