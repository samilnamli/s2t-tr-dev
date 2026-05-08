"""Per-stage provenance manifests for the ASR feature pipeline.

A :class:`Manifest` describes everything needed to *reproduce* a parquet
artifact: source dataset (HF id + revision), extractor models (HF id +
revision), git state at write time, the resolved Hydra config hash,
runtime versions, and a wall-clock timestamp.

The manifest lives in two places, deliberately redundant:

1. As ``manifest.yaml`` next to the parquet on disk (and in the HF Hub
   commit). Human-readable; greppable; useful when the parquet has been
   moved by hand.
2. Embedded in the parquet's Arrow schema metadata under the
   ``extraction_manifest`` (UTF-8 JSON) key. Makes a parquet
   self-describing — you can recover provenance from the file alone,
   even if it was renamed or copied without its sidecar.

Comparing two manifests answers "are these artifacts equivalent for my
purposes?" — :meth:`Manifest.matches` exposes a configurable subset of
fields to compare, since e.g. ``created_at`` always differs but doesn't
affect reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, List, Mapping, Optional

import pyarrow as pa
import yaml

from src.utils.git import git_state

MANIFEST_FILENAME = "manifest.yaml"
MANIFEST_METADATA_KEY = b"extraction_manifest"
MANIFEST_SCHEMA_VERSION = 1

# Default fields ignored when comparing manifests for equivalence.
# ``created_at`` and ``created_by`` are environmental drift; ``git`` and
# ``versions`` are useful audit info but a strict equivalence check on
# them prevents pulling artifacts produced from a different commit even
# when the artifact itself is byte-identical.
DEFAULT_MATCH_IGNORE = ("created_at", "created_by", "git", "versions")


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable spec of the source dataset for a manifest."""

    hf_id: str
    revision: Optional[str] = None        # HF dataset commit SHA — pin in production
    split: Optional[str] = None
    config: Optional[str] = None          # HF datasets config name (e.g. "en")
    num_examples: Optional[int] = None
    chunking: Optional[Mapping[str, Any]] = None  # method/seconds/overlap/etc.

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ExtractorEntry:
    """Manifest entry for one frozen ASR model used in extraction."""

    name: str                     # logical column-name prefix
    hf_model_id: str
    hf_revision: Optional[str] = None
    embedding_dim: Optional[int] = None
    sample_rate: int = 16000

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Manifest:
    """Provenance for one parquet artifact."""

    stage: str                              # "raw" | "interim" | "processed"
    dataset: DatasetSpec
    extractors: List[ExtractorEntry] = field(default_factory=list)
    config_hash: Optional[str] = None       # SHA256 of resolved Hydra config
    git: Mapping[str, Any] = field(default_factory=dict)
    versions: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    created_by: str = ""
    schema_version: int = MANIFEST_SCHEMA_VERSION

    # ----- factory ----------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        stage: str,
        dataset: DatasetSpec,
        extractors: Iterable[ExtractorEntry] = (),
        config_hash: Optional[str] = None,
    ) -> "Manifest":
        """Construct a manifest snapshotting the current environment."""
        return cls(
            stage=stage,
            dataset=dataset,
            extractors=list(extractors),
            config_hash=config_hash,
            git=git_state(),
            versions=_runtime_versions(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            created_by=_safe_user(),
        )

    # ----- (de)serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "dataset": self.dataset.to_dict(),
            "extractors": [e.to_dict() for e in self.extractors],
            "config_hash": self.config_hash,
            "git": dict(self.git),
            "versions": dict(self.versions),
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Manifest":
        ds_raw = dict(d.get("dataset", {}))
        chunking = ds_raw.pop("chunking", None)
        dataset = DatasetSpec(**ds_raw, chunking=chunking) if "hf_id" in ds_raw else DatasetSpec(
            hf_id="<unknown>"
        )
        extractors = [ExtractorEntry(**e) for e in d.get("extractors", [])]
        return cls(
            stage=str(d.get("stage", "")),
            dataset=dataset,
            extractors=extractors,
            config_hash=d.get("config_hash"),
            git=dict(d.get("git", {}) or {}),
            versions=dict(d.get("versions", {}) or {}),
            created_at=str(d.get("created_at", "")),
            created_by=str(d.get("created_by", "")),
            schema_version=int(d.get("schema_version", MANIFEST_SCHEMA_VERSION)),
        )

    def to_yaml(self) -> str:
        # Round-trip through JSON to flatten any non-yaml-serializable
        # types (e.g. ``pkg_resources.Version`` strings from some
        # transitive dependencies) into plain Python primitives.
        safe = json.loads(json.dumps(self.to_dict(), default=str))
        return yaml.safe_dump(safe, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> "Manifest":
        return cls.from_dict(yaml.safe_load(text))

    # ----- file-backed ------------------------------------------------------

    def write_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @classmethod
    def read_yaml(cls, path: str | Path) -> "Manifest":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    # ----- arrow embedding --------------------------------------------------

    def to_metadata_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")

    def attach_to_schema(self, schema: pa.Schema) -> pa.Schema:
        """Return a copy of ``schema`` with the manifest embedded.

        Existing metadata keys are preserved; the ``extraction_manifest``
        key is overwritten if already present.
        """
        existing = dict(schema.metadata or {})
        existing[MANIFEST_METADATA_KEY] = self.to_metadata_bytes()
        return schema.with_metadata(existing)

    @classmethod
    def from_parquet_metadata(cls, schema: pa.Schema) -> Optional["Manifest"]:
        raw = (schema.metadata or {}).get(MANIFEST_METADATA_KEY)
        if raw is None:
            return None
        return cls.from_dict(json.loads(raw.decode("utf-8")))

    # ----- equivalence ------------------------------------------------------

    def matches(
        self,
        other: "Manifest",
        ignore: Iterable[str] = DEFAULT_MATCH_IGNORE,
    ) -> bool:
        """Return True when ``self`` and ``other`` agree on the comparable fields.

        ``ignore`` lists top-level keys of :meth:`to_dict` to skip
        (defaults to environmental drift fields). Pass ``ignore=()`` for
        a strict comparison.
        """
        ignore_set = set(ignore)
        a = {k: v for k, v in self.to_dict().items() if k not in ignore_set}
        b = {k: v for k, v in other.to_dict().items() if k not in ignore_set}
        return a == b

    def diff(self, other: "Manifest") -> dict:
        """Return ``{field: (self_value, other_value)}`` for every disagreeing field.

        Useful for surfacing *why* a pulled artifact didn't match the
        expected manifest — much more debuggable than a boolean.
        """
        a, b = self.to_dict(), other.to_dict()
        keys = sorted(set(a) | set(b))
        return {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def hash_config(resolved_config: Mapping[str, Any]) -> str:
    """Deterministic SHA256 hash of a resolved Hydra config.

    The config is canonicalized as JSON with sorted keys before hashing
    so the hash is invariant under dict-key ordering.
    """
    blob = json.dumps(resolved_config, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def _runtime_versions() -> dict:
    """Snapshot of library versions that affect extraction reproducibility."""
    out: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "datasets", "huggingface_hub", "pyarrow"):
        try:
            mod = __import__(pkg)
            out[pkg] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            out[pkg] = "not-installed"
    out["platform"] = f"{platform.system()}-{platform.release()}-{platform.machine()}"
    out["python_implementation"] = platform.python_implementation()
    out["sys_executable"] = sys.executable
    return out


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 — getuser raises various OSError subclasses
        return "unknown"
