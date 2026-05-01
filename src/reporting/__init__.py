"""Generic reporting layer.

Consumes the per-experiment ``main_results.json`` produced by
:mod:`src.experiments.run` and emits the manuscript-grade
deliverables — Markdown / LaTeX tables and matplotlib figures — used
both by the Colab notebooks (final cell) and by the manuscript via
``\\input{auto/...}``.

Public entrypoints are typer apps so they can be invoked directly
from a notebook cell:

    !uv run python -m src.reporting.tables render \\
        --results reports/main_results/<name>/main_results.json \\
        --output-dir reports/manuscript/figures/auto/<name>
"""
