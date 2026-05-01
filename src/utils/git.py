"""Capture the current git state for reproducibility tracking.

We log the git SHA + dirty-state into the W&B run config so that every
metric is unambiguously traceable to a code state. The functions here
are dependency-free (no GitPython) and tolerate non-git environments
(e.g. Colab clones in degraded modes) by returning ``None`` rather
than raising.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional


def _git(*args: str) -> Optional[str]:
    """Run ``git <args>`` and return stdout stripped, or ``None`` on error."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.decode("utf-8").strip() or None


def git_state() -> dict[str, Optional[str]]:
    """Return ``{commit, branch, dirty, remote}`` for the current repo.

    ``dirty`` is the literal string ``"true"`` / ``"false"`` (or ``None``
    if git is unavailable) so it is JSON-serializable for W&B.
    """
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    remote = _git("config", "--get", "remote.origin.url")

    dirty: Optional[str]
    if status is None:
        dirty = None
    else:
        dirty = "true" if status.strip() else "false"

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "remote": remote,
    }
