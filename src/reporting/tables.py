"""Render manuscript-grade tables from a ``main_results.json`` aggregator.

The aggregator file is produced by :mod:`src.experiments.run` and has
the shape::

    {
        "config_name": "ablation_loss",
        "experiment_metadata": {...},
        "training_free": {...},   # may be None if skip_training_free=true
        "rover":         {...},   # may be None if skip_rover=true
        "methods": {
            "<method_name>-seed42": {"test": {"selected_wer": 0.319, ...}, "config_overrides": {...}},
            "<method_name>-seed2":  {"test": {...}, ...},
            ...
        }
    }

This module groups multi-seed runs (any name suffixed ``-seed<n>``),
computes mean ± std, and emits both a Markdown table (for terminal /
notebook viewing) and a LaTeX ``booktabs``-style table (for
``\\input{...}`` into ``main.tex``).

Optionally, the rendered files are also pushed as W&B artifacts.

Usage:
    uv run python -m src.reporting.tables render \\
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
import numpy as np
import typer

from src.utils.logging import setup_unified_logging

app = typer.Typer(help="Render Markdown + LaTeX tables from main_results.json.")

_SEED_RE = re.compile(r"-seed(\d+)$")


# --------------------------------------------------------------------------- #
# Public helper for src.experiments.run to call
# --------------------------------------------------------------------------- #


def render_main_results_table(
    json_path: Path,
    output_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Render Markdown + LaTeX tables next to a ``main_results.json``.

    Used by :mod:`src.experiments.run` after a pipeline finishes; safe
    to call when reporting fails (caller catches and warns).

    Returns:
        ``(markdown_path, latex_path)`` — either may be ``None`` on failure.
    """
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        logger.warning("render_main_results_table: {} does not exist.", json_path)
        return None, None

    data = json.loads(json_path.read_text())
    rows = _build_rows(data)
    if not rows:
        logger.warning("render_main_results_table: no rows to render.")
        return None, None

    md_path = output_dir / "results.md"
    tex_path = output_dir / "results.tex"
    md_path.write_text(_to_markdown(rows, data.get("config_name", "")))
    tex_path.write_text(_to_latex(rows, data.get("config_name", "")))
    logger.info("Rendered table to {} and {}", md_path, tex_path)
    return md_path, tex_path


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def _flat_wer(d: Dict[str, Any]) -> Optional[float]:
    """Pull ``selected_wer`` (preferred) or ``wer`` from a flattened test dict."""
    if "selected_wer" in d:
        return float(d["selected_wer"])
    if "wer" in d:
        return float(d["wer"])
    return None


def _group_methods_by_base_name(
    methods: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Group ``<base>-seed<n>`` keys back together for mean-±-std reporting."""
    grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for name, payload in methods.items():
        wer = _flat_wer(payload.get("test", {}))
        if wer is None:
            continue
        base = _SEED_RE.sub("", name)
        grouped[base].append((name, wer))
    return grouped


def _build_rows(data: Dict[str, Any]) -> List[Tuple[str, float, float, int]]:
    """Build ``(name, mean_wer, std_wer, n)`` rows ready for table rendering.

    Order:
        1. Training-free baselines (single-shot per name).
        2. ROVER and weighted ROVER (single-shot per name).
        3. Trained methods, grouped by base name (mean ± std over seeds).
        4. Within each block, sorted ascending by WER for legibility.
    """
    rows: List[Tuple[str, float, float, int]] = []

    tf = data.get("training_free")
    if tf:
        for k, v in (tf.get("results", {}).get("test", tf) or {}).items():
            if isinstance(v, dict) and "wer_mean" in v:
                rows.append((f"baseline:{k}", float(v["wer_mean"]), 0.0, int(v.get("n", 1))))
            elif isinstance(v, float) and "wer" in k.lower():
                rows.append((f"baseline:{k}", float(v), 0.0, 1))

    rv = data.get("rover")
    if rv:
        rover_section = rv.get("results", {}).get("test", rv) or {}
        for k, v in rover_section.items():
            if isinstance(v, dict) and "wer_mean" in v:
                rows.append((f"rover:{k}", float(v["wer_mean"]), 0.0, int(v.get("n", 1))))

    grouped = _group_methods_by_base_name(data.get("methods") or {})
    for base, items in grouped.items():
        wers = [w for _, w in items]
        mean = float(np.mean(wers))
        std = float(np.std(wers, ddof=1)) if len(wers) > 1 else 0.0
        rows.append((f"method:{base}", mean, std, len(wers)))

    rows.sort(key=lambda r: r[1])
    return rows


# --------------------------------------------------------------------------- #
# Markdown / LaTeX renderers
# --------------------------------------------------------------------------- #


def _fmt_wer(mean: float, std: float, n: int) -> str:
    if n > 1:
        return f"{mean:.4f} ± {std:.4f} (n={n})"
    return f"{mean:.4f}"


def _to_markdown(rows: List[Tuple[str, float, float, int]], title: str) -> str:
    head = f"# Results — `{title}`\n\n" if title else ""
    out = [head, "| Method | Test WER |", "| --- | --- |"]
    for name, mean, std, n in rows:
        out.append(f"| `{name}` | {_fmt_wer(mean, std, n)} |")
    return "\n".join(out) + "\n"


def _to_latex(rows: List[Tuple[str, float, float, int]], title: str) -> str:
    out = [
        "% Auto-generated by src.reporting.tables — do not edit by hand.",
        f"% Source: main_results.json for config '{title}'.",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Method & Test WER \\\\",
        "\\midrule",
    ]
    for name, mean, std, n in rows:
        clean = name.replace("_", "\\_")
        if n > 1:
            cell = f"{mean * 100:.2f} $\\pm$ {std * 100:.2f} (n={n})"
        else:
            cell = f"{mean * 100:.2f}"
        out.append(f"{clean} & {cell} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Typer CLI
# --------------------------------------------------------------------------- #


@app.command("render")
def render_cli(
    results: str = typer.Option(..., "--results", "-r", help="Path to main_results.json."),
    output_dir: str = typer.Option(..., "--output-dir", "-o"),
    push_wandb: bool = typer.Option(False, "--push-wandb/--no-push-wandb"),
    wandb_project: str = typer.Option("s2t-tr-dev", "--wandb-project"),
    wandb_run_name: Optional[str] = typer.Option(None, "--wandb-run-name"),
):
    """Render Markdown + LaTeX tables and (optionally) push to W&B."""
    setup_unified_logging(level="INFO")
    md, tex = render_main_results_table(Path(results), Path(output_dir))

    if push_wandb and md and tex:
        if not os.environ.get("WANDB_API_KEY"):
            logger.warning("--push-wandb requested but WANDB_API_KEY not set; skipping.")
            return
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"reporting-tables-{Path(results).parent.name}",
                job_type="reporting",
                reinit=True,
            )
            artifact = wandb.Artifact(
                name=f"results-table-{Path(results).parent.name}",
                type="results-table",
            )
            artifact.add_file(str(md))
            artifact.add_file(str(tex))
            artifact.add_file(results)
            run.log_artifact(artifact)
            run.finish()
            logger.info("Pushed table artifact to W&B project={}", wandb_project)
        except Exception as exc:  # pragma: no cover - W&B may be offline
            logger.warning("Failed to push W&B artifact: {}", exc)


if __name__ == "__main__":
    app()
