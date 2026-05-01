"""Shared subprocess + config helpers for ``src/experiments/*``.

Single source of truth for the orchestrator-side plumbing that used to
be duplicated across ``main_results.py``, ``run_ablation.py``, and
``synthetic_sweep.py``. Keeps each high-level runner short and lets us
fix bugs (e.g. how booleans are translated to flags) in one place.

Public functions:
    :func:`python_module`    — build a ``[python, -m, name, *args]`` invocation.
    :func:`run`              — synchronous subprocess runner with logging.
    :func:`flags_from_dict`  — dict to ``--flag value`` list (typer style).
    :func:`hydra_overrides`  — dict to ``key=value`` list (hydra style).
    :func:`expand_seeds`     — duplicate a method dict over a list of seeds.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict, List

from loguru import logger


def python_module(name: str, *args: str) -> List[str]:
    """Build ``[python, -m, name, *args]`` using the active interpreter.

    We resolve the current ``sys.executable`` rather than relying on
    ``PATH`` lookup so that ``uv run`` / virtualenvs / Colab kernels
    all execute the same Python the orchestrator is running in.
    """
    return [sys.executable, "-m", name, *args]


def run(cmd: List[str], description: str) -> None:
    """Run ``cmd`` synchronously, streaming output, raising on non-zero exit.

    Args:
        cmd: argv-style list (already shell-escaped by Python).
        description: short label for the log line and error message.

    Raises:
        RuntimeError: if the subprocess exits non-zero.
    """
    logger.info(">>> {} — {}", description, " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed (exit {completed.returncode}): {' '.join(cmd)}")


def flags_from_dict(d: Dict[str, Any]) -> List[str]:
    """Convert a config dict to typer-compatible ``--flag value`` pairs.

    Booleans become bare ``--flag`` toggles (or are skipped when False).
    ``None`` values are skipped. All other values are stringified.
    """
    out: List[str] = []
    for k, v in d.items():
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            out.append(flag)
        elif v is False or v is None:
            continue
        else:
            out.extend([flag, str(v)])
    return out


def hydra_overrides(d: Dict[str, Any]) -> List[str]:
    """Convert a config dict to Hydra ``key=value`` overrides.

    ``None`` values are skipped (Hydra treats them as removal). Booleans
    are lower-cased per Hydra's YAML interpretation rules.
    """
    out: List[str] = []
    for k, v in d.items():
        if v is None:
            continue
        if v is True:
            out.append(f"{k}=true")
        elif v is False:
            out.append(f"{k}=false")
        else:
            out.append(f"{k}={v}")
    return out


def expand_seeds(methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand each method whose ``seed`` is a list into one entry per seed.

    The expanded entry's ``name`` is suffixed ``-seed<n>`` so log dirs
    and W&B run names are unique. A scalar seed (or absent seed) is
    passed through unchanged.
    """
    out: List[Dict[str, Any]] = []
    for method in methods:
        seeds = method.get("seed")
        if isinstance(seeds, (list, tuple)):
            for s in seeds:
                expanded = dict(method)
                expanded["seed"] = s
                expanded["name"] = f"{method['name']}-seed{s}"
                out.append(expanded)
        else:
            out.append(dict(method))
    return out
