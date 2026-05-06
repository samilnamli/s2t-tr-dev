"""Capture git state for reproducibility tracking.

Functions are dependency-free (no GitPython) and tolerate non-git
environments by returning ``None`` rather than raising.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> Optional[str]:
    """Run ``git <args>`` and return stdout, or ``None`` on error."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            cwd=REPO_ROOT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.decode("utf-8").rstrip("\n") or None


def git_state() -> dict[str, Optional[str]]:
    """Return ``{commit, branch, dirty, remote}`` for the current repo."""
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


def git_diff_patch() -> Optional[str]:
    """Combined staged + unstaged diff against HEAD as a single patch.

    Returns ``None`` for a clean tree or non-git environment. Includes
    untracked files via ``git diff --no-index`` to /dev/null is too
    fragile; we capture only tracked-file changes here. Untracked
    additions show up in ``git_state()['dirty']`` but not in the patch.
    """
    diff = _git("diff", "HEAD", "--binary")
    return diff if diff else None


def working_tree_clean() -> bool:
    """True iff working tree has no modifications. Non-git → False."""
    state = git_state()
    return state["dirty"] == "false"
