"""Render manuscript-grade figures from a ``main_results.json`` aggregator.

For experiments where the manuscript ships a figure rather than (or
in addition to) a table, this module produces:

* ``selection_frequency.{pdf,png}`` — a grouped bar chart of per-expert
  selection frequency, one bar group per method.
* ``method_wer_bars.{pdf,png}``     — a horizontal bar chart of test WER
  per method, with per-seed error bars when multiple seeds are present.

The synthetic R-sweep figure stays in :mod:`src.experiments.sweep` (it
needs the cross-R sweep file, not a single ``main_results.json``).

Usage:
    uv run python -m src.reporting.figures render \\
        --results reports/main_results/ablation_loss/main_results.json \\
        --output-dir reports/manuscript/figures/auto/ablation_loss \\
        [--push-wandb]
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import typer

from src.utils.logging import setup_unified_logging

app = typer.Typer(help="Render manuscript-grade figures from a main_results.json aggregator.")

_SEED_RE = re.compile(r"-seed(\d+)$")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _build_method_rows(
    data: Dict[str, Any],
) -> List[Tuple[str, float, float, int]]:
    """Group ``-seed<n>`` runs and return ``(base_name, mean, std, n)`` rows."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    for name, payload in (data.get("methods") or {}).items():
        wer = payload.get("test", {}).get("selected_wer")
        if wer is None:
            wer = payload.get("test", {}).get("wer")
        if wer is None:
            continue
        base = _SEED_RE.sub("", name)
        grouped[base].append(float(wer))

    rows = []
    for base, wers in grouped.items():
        mean = float(np.mean(wers))
        std = float(np.std(wers, ddof=1)) if len(wers) > 1 else 0.0
        rows.append((base, mean, std, len(wers)))
    rows.sort(key=lambda r: r[1])
    return rows


def render_method_wer_bars(
    data: Dict[str, Any],
    output_path: Path,
) -> Optional[Path]:
    rows = _build_method_rows(data)
    if not rows:
        logger.warning("render_method_wer_bars: no method rows in JSON.")
        return None

    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.45 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, means, xerr=stds, color="#1f77b4", alpha=0.85, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Test WER")
    ax.set_title(f"Test WER per method — {data.get('config_name', '')}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)
    return output_path


def render_selection_frequency(
    data: Dict[str, Any],
    output_path: Path,
) -> Optional[Path]:
    """Bar chart of ``freq_<expert>`` test metrics, one bar group per method."""
    methods = data.get("methods") or {}
    if not methods:
        return None
    freq_keys: List[str] = []
    for payload in methods.values():
        freq_keys = sorted({k for k in payload.get("test", {}).keys() if k.startswith("freq_")})
        if freq_keys:
            break
    if not freq_keys:
        logger.warning("render_selection_frequency: no freq_* keys in test results.")
        return None

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for name, payload in methods.items():
        base = _SEED_RE.sub("", name)
        for k in freq_keys:
            v = payload.get("test", {}).get(k)
            if v is not None:
                grouped[base][k].append(float(v))

    bases = list(grouped.keys())
    K = len(freq_keys)
    width = 0.8 / max(len(bases), 1)
    x = np.arange(K)
    palette = plt.cm.tab10(np.arange(len(bases)) % 10)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, base in enumerate(bases):
        means = [float(np.mean(grouped[base][k])) if grouped[base][k] else 0.0 for k in freq_keys]
        stds = [
            float(np.std(grouped[base][k], ddof=1)) if len(grouped[base][k]) > 1 else 0.0
            for k in freq_keys
        ]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            means,
            yerr=stds,
            width=width,
            color=palette[i],
            label=base,
            capsize=2,
        )
    labels = [k.replace("freq_", "") for k in freq_keys]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Selection frequency on test split")
    ax.set_title(f"Selection frequency per expert — {data.get('config_name', '')}")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)
    return output_path


@app.command("render")
def render_cli(
    results: str = typer.Option(..., "--results", "-r"),
    output_dir: str = typer.Option(..., "--output-dir", "-o"),
    push_wandb: bool = typer.Option(False, "--push-wandb/--no-push-wandb"),
    wandb_project: str = typer.Option("s2t-tr-dev", "--wandb-project"),
):
    """Render the figure pair (WER bars + selection frequency) from a results JSON."""
    setup_unified_logging(level="INFO")
    data = json.loads(Path(results).read_text())
    out = Path(output_dir)
    paths = [
        render_method_wer_bars(data, out / "method_wer_bars.pdf"),
        render_selection_frequency(data, out / "selection_frequency.pdf"),
    ]
    paths = [p for p in paths if p is not None]
    for p in paths:
        logger.info("Rendered figure: {}", p)

    if push_wandb and paths:
        if not os.environ.get("WANDB_API_KEY"):
            logger.warning("--push-wandb requested but WANDB_API_KEY not set; skipping.")
            return
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                name=f"reporting-figures-{Path(results).parent.name}",
                job_type="reporting",
                reinit=True,
            )
            artifact = wandb.Artifact(
                name=f"results-figures-{Path(results).parent.name}",
                type="results-figures",
            )
            for p in paths:
                artifact.add_file(str(p))
                if p.suffix == ".pdf":
                    png = p.with_suffix(".png")
                    if png.exists():
                        artifact.add_file(str(png))
            run.log_artifact(artifact)
            run.finish()
            logger.info("Pushed figures artifact to W&B project={}", wandb_project)
        except Exception as exc:  # pragma: no cover - W&B may be offline
            logger.warning("Failed to push W&B artifact: {}", exc)


if __name__ == "__main__":
    app()
