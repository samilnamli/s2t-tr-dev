"""Tiny helpers used by Colab notebooks.

The notebooks under ``notebooks/colab/`` are reproducibility packages
that may only contain shell calls (``!uv run python -m ...`` /
``!make ...``) and a small amount of platform glue. Anything that
would otherwise be raw inline Python (loading Colab secrets,
displaying a rendered manuscript table, etc.) lives here so the
notebook itself stays minimal and the helper functions are
independently testable from ``src/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

DEFAULT_SECRETS: tuple[str, ...] = ("WANDB_API_KEY", "WANDB_ENTITY", "HF_TOKEN")


def load_colab_secrets(names: Iterable[str] = DEFAULT_SECRETS) -> dict[str, bool]:
    """Copy Colab Secrets into ``os.environ``.

    Returns a ``{name: was_loaded}`` dict so the calling cell can print a
    short status line. Silently no-ops outside Colab (so the same
    notebook still runs locally with pre-set environment variables).
    """
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except Exception:
        return {n: bool(os.environ.get(n)) for n in names}

    status: dict[str, bool] = {}
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = value
            status[name] = True
        else:
            status[name] = bool(os.environ.get(name))
    return status


def display_deliverables(experiment: str, base_dir: str = "reports/manuscript/figures/auto") -> None:
    """Render a Markdown ``results.md`` and every ``*.png`` figure for an experiment.

    Defers IPython imports so this module is importable in a non-IPython
    environment for testing.
    """
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        print("display_deliverables: IPython not available; skipping inline render.")
        return

    out_dir = Path(base_dir) / experiment
    md = out_dir / "results.md"
    if md.exists():
        display(Markdown(md.read_text()))
    else:
        print(f"display_deliverables: no results.md found at {md}")

    pngs = sorted(out_dir.glob("*.png"))
    for png in pngs:
        display(Markdown(f"### `{png.stem}`"))
        display(Image(filename=str(png)))
    if not pngs:
        print(f"display_deliverables: no .png figures under {out_dir}")
