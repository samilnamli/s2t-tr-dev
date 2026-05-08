"""HuggingFace Hub push / pull for extraction artifacts.

Layout convention for one HF dataset repo per evaluation dataset, e.g.
``huseyin-karaca/s2t-tr-earnings22``:

    interim/<extractor_name>/features.parquet
    interim/<extractor_name>/manifest.yaml
    processed/combined_features.parquet
    processed/manifest.yaml

Push policy:

* Refuse to push when the working tree is dirty unless the caller
  explicitly opts in via ``allow_dirty=True``.  When dirty pushes are
  allowed, the diff patch is uploaded alongside the artifact under
  ``<stage>/git_dirty.patch`` so the snapshot remains reproducible.
* The Hub commit message encodes the short git SHA + config hash so
  the Hub history doubles as a provenance trail.

Pull policy:

* :func:`pull_artifact` always reads the manifest first and verifies
  it matches the caller's expected manifest before trusting the local
  files. A mismatch raises :class:`ManifestMismatchError`.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from src.data.extraction.manifest import (
    DEFAULT_MATCH_IGNORE,
    MANIFEST_FILENAME,
    Manifest,
)
from src.utils.git import git_diff_patch, git_state

logger = logging.getLogger(__name__)


class ManifestMismatchError(RuntimeError):
    """Raised when a pulled artifact's manifest disagrees with the expected one."""

    def __init__(self, repo_id: str, remote_path: str, diff: dict):
        super().__init__(
            f"Manifest mismatch for {repo_id}/{remote_path}. Disagreeing fields: "
            f"{sorted(diff.keys())}"
        )
        self.repo_id = repo_id
        self.remote_path = remote_path
        self.diff = diff


@dataclass
class HubConfig:
    """Hydra-friendly Hub config — the pieces of state needed by push/pull.

    ``repo_id`` is a Python format-string with one optional placeholder:
    ``{dataset_key}``. This lets one config drive multiple datasets:

        repo_id: huseyin-karaca/s2t-tr-{dataset_key}
    """

    repo_id: str
    repo_type: str = "dataset"
    revision: Optional[str] = None       # branch/tag/commit; ``None`` = ``main``
    private: bool = False
    push: bool = True                    # global kill-switch (CI / dry runs)
    pull: bool = True
    allow_dirty: bool = False
    token: Optional[str] = None          # ``None`` ⟶ environment ``HF_TOKEN``

    def resolved_repo_id(self, dataset_key: str) -> str:
        try:
            return self.repo_id.format(dataset_key=dataset_key)
        except KeyError:
            return self.repo_id


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


def push_artifact(
    local_dir: str | Path,
    *,
    repo_id: str,
    remote_subdir: str,
    config: HubConfig,
    extra_files: Iterable[str | Path] = (),
) -> Optional[str]:
    """Upload everything under ``local_dir`` to ``{repo_id}/{remote_subdir}``.

    Returns the resulting commit SHA, or ``None`` when ``config.push``
    is False (caller asked us not to push, e.g. CI).
    """
    if not config.push:
        logger.info("Hub push disabled (config.push=false); skipping %s.", remote_subdir)
        return None

    state = git_state()
    if state.get("dirty") == "true" and not config.allow_dirty:
        raise RuntimeError(
            "Refusing to push: working tree is dirty. Commit your changes or set "
            "storage.allow_dirty=true (which will also push the diff patch)."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=config.token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=config.repo_type,
        exist_ok=True,
        private=config.private,
    )

    local_dir = Path(local_dir)

    # Stage extras into the local dir so a single upload_folder call
    # captures them. ``git_dirty.patch`` is only relevant when
    # ``allow_dirty=True``; emit it conditionally.
    if state.get("dirty") == "true" and config.allow_dirty:
        patch = git_diff_patch()
        if patch:
            (local_dir / "git_dirty.patch").write_text(patch, encoding="utf-8")

    allow_patterns: List[str] = ["*"]
    commit_message = _commit_message(remote_subdir, state)

    logger.info(
        "Pushing %s → %s (path=%s)",
        local_dir,
        repo_id,
        remote_subdir,
    )
    info = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=config.repo_type,
        path_in_repo=remote_subdir,
        commit_message=commit_message,
        allow_patterns=allow_patterns,
        revision=config.revision,
    )
    logger.info("Pushed; commit: %s", getattr(info, "oid", "<unknown>"))
    for extra in extra_files:
        logger.info("(noted extra file %s — already inside local_dir)", extra)
    return getattr(info, "oid", None)


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


def pull_artifact(
    *,
    repo_id: str,
    remote_subdir: str,
    local_dir: str | Path,
    expected_manifest: Optional[Manifest] = None,
    config: HubConfig,
    ignore_for_match: Iterable[str] = DEFAULT_MATCH_IGNORE,
) -> bool:
    """Download ``{repo_id}/{remote_subdir}`` into ``local_dir``.

    Returns ``True`` when the artifact was fetched and its manifest
    matched ``expected_manifest`` (or no expectation was set), ``False``
    when ``config.pull`` is False or the remote is missing the
    artifact. Raises :class:`ManifestMismatchError` when the artifact
    exists but disagrees with ``expected_manifest``.
    """
    if not config.pull:
        logger.info("Hub pull disabled (config.pull=false); skipping %s.", remote_subdir)
        return False

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=config.repo_type,
            revision=config.revision,
            local_dir=str(local_dir.parent),
            allow_patterns=[f"{remote_subdir}/*"],
            token=config.token,
        )
    except (RepositoryNotFoundError, EntryNotFoundError) as e:
        logger.info("Pull miss for %s/%s: %s", repo_id, remote_subdir, e)
        return False
    except Exception as e:  # noqa: BLE001 — surface as miss rather than crash prepare_data
        logger.warning("Pull failed for %s/%s (%s); falling back to local extraction.", repo_id, remote_subdir, e)
        return False

    # Verify manifest (if expected).
    if expected_manifest is None:
        return True

    found_path = local_dir / MANIFEST_FILENAME
    if not found_path.exists():
        logger.warning("Pulled artifact missing %s — treating as miss.", found_path)
        return False

    found = Manifest.read_yaml(found_path)
    if expected_manifest.matches(found, ignore=ignore_for_match):
        return True
    raise ManifestMismatchError(
        repo_id=repo_id,
        remote_path=remote_subdir,
        diff=expected_manifest.diff(found),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _commit_message(remote_subdir: str, state: dict) -> str:
    commit = (state.get("commit") or "no-git")[:7]
    branch = state.get("branch") or "no-branch"
    dirty = "+dirty" if state.get("dirty") == "true" else ""
    return f"extraction: stage={remote_subdir} git={commit}{dirty}@{branch}"
