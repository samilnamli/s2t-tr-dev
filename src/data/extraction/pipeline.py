"""DataPipeline — three-stage orchestrator for raw → interim → processed.

For each stage the pipeline first checks whether the local artifact is
present and matches the expected manifest. If yes, it's used as-is.
If no, the pipeline tries to pull from HF Hub. If that misses, it
falls back to running the local extraction code, then optionally
pushes the freshly built artifact back to the Hub for next time.

This three-tier fallback (local → hub → build) is the central feature
of the pipeline: it makes ``prepare_data()`` cheap on collaborator
machines (just a hub pull) while keeping reproducibility intact (the
manifest gates every step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import List, Mapping, Optional

import yaml

from src.data.extraction.extractors import BaseFeatureExtractor
from src.data.extraction.hub import HubConfig, ManifestMismatchError, pull_artifact, push_artifact
from src.data.extraction.manifest import (
    DEFAULT_MATCH_IGNORE,
    DatasetSpec,
    ExtractorEntry,
    MANIFEST_FILENAME,
    Manifest,
)
from src.data.extraction.stages.chunk import ChunkingConfig, chunk_clips
from src.data.extraction.stages.interim import InterimWriteConfig, extract_interim
from src.data.extraction.stages.processed import build_processed
from src.data.extraction.stages.raw import RawDatasetConfig, load_raw_dataset, to_dataset_spec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class StorageConfig:
    """Where each stage's artifacts live on disk and on the Hub.

    ``data_root`` is the project-relative location under which
    ``interim/{dataset_key}/{extractor_name}/`` and
    ``processed/{dataset_key}/`` directories will be written. The
    ``dataset_key`` is the user-chosen short name (e.g. ``earnings22``)
    that doubles as the Hub repo's leaf name.
    """

    data_root: str = "data"
    dataset_key: str = "default"
    hub: Optional[HubConfig] = None


@dataclass
class ManifestPolicy:
    """How strict the pipeline is about manifest agreement."""

    mode: str = "strict"  # "strict" — error on mismatch; "warn" — log; "off" — ignore
    ignore_fields: List[str] = field(default_factory=lambda: list(DEFAULT_MATCH_IGNORE))


@dataclass
class PipelineConfig:
    """Hydra-friendly bundle of everything DataPipeline needs."""

    raw: RawDatasetConfig
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    write: InterimWriteConfig = field(default_factory=InterimWriteConfig)
    manifest_policy: ManifestPolicy = field(default_factory=ManifestPolicy)
    config_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """Three-tier orchestrator: local cache → HF Hub → live extraction.

    Construct with the Hydra config and a list of extractors (the
    extractors are passed separately so the heavy HF-model
    initialization only happens when they're actually needed —
    ``ensure_processed`` may complete without ever instantiating them).
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        extractor_specs: List["ExtractorSpecLike"],
        extractor_factory: Optional[callable] = None,
    ):
        self.cfg = cfg
        self.extractor_specs = list(extractor_specs)
        if not self.extractor_specs:
            raise ValueError("DataPipeline requires at least one extractor spec.")
        self._extractor_factory = extractor_factory  # () -> BaseFeatureExtractor instance per spec
        self._cached_extractors: dict[str, BaseFeatureExtractor] = {}

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    @property
    def data_root(self) -> Path:
        return Path(self.cfg.storage.data_root)

    def interim_dir(self, extractor_name: str) -> Path:
        return self.data_root / "interim" / self.cfg.storage.dataset_key / extractor_name

    def interim_parquet_path(self, extractor_name: str) -> Path:
        return self.interim_dir(extractor_name) / "features.parquet"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed" / self.cfg.storage.dataset_key

    @property
    def processed_parquet_path(self) -> Path:
        return self.processed_dir / "combined_features.parquet"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def ensure_processed(self) -> Path:
        """Guarantee the processed parquet exists locally and return its path."""
        path = self.processed_parquet_path
        expected = self._expected_processed_manifest()

        if self._artifact_ok(path, expected):
            logger.info("Processed parquet present and verified: %s", path)
            return path

        if self._try_pull(self.processed_dir, "processed", expected):
            return path

        # Build from interim — recurse upstream.
        interim_paths = self.ensure_interim()
        from src.data.extraction.stages.processed import build_processed as _build_processed

        # Use whatever extractor entries we have on hand (interim manifests
        # carry the discovered embedding_dim — read those for the processed
        # manifest so the dim is recorded).
        entries = self._extractor_entries_with_dims(interim_paths)
        _build_processed(
            interim_paths=interim_paths,
            output_path=path,
            dataset_spec=self._dataset_spec(),
            extractor_entries=entries,
            config_hash=self.cfg.config_hash,
        )
        self._try_push(self.processed_dir, "processed")
        return path

    def ensure_interim(self) -> Mapping[str, Path]:
        """Per-extractor: local-ok? else pull-from-Hub? else extract+push."""
        paths: dict[str, Path] = {}
        missing_extractors: list[ExtractorSpecLike] = []

        for spec in self.extractor_specs:
            path = self.interim_parquet_path(spec.name)
            expected = self._expected_interim_manifest(spec)
            if self._artifact_ok(path, expected):
                logger.info("Interim parquet present and verified: %s", path)
                paths[spec.name] = path
                continue
            if self._try_pull(self.interim_dir(spec.name), f"interim/{spec.name}", expected):
                paths[spec.name] = path
                continue
            missing_extractors.append(spec)

        if missing_extractors:
            self._extract_missing(missing_extractors, paths)
        return paths

    def ensure_raw(self):
        """Yield :class:`RawClip` rows for the configured HF dataset.

        Re-iterable: yields a fresh iterator each call. The HF
        ``datasets`` library handles caching under ``HF_HOME``.
        """
        return load_raw_dataset(self.cfg.raw)

    # ------------------------------------------------------------------
    # internal: extraction
    # ------------------------------------------------------------------

    def _extract_missing(
        self,
        missing: List["ExtractorSpecLike"],
        paths: dict[str, Path],
    ) -> None:
        """Load missing extractors and stream raw clips through them.

        We materialize chunked clips into memory once so each extractor
        sees the same iterator order — that keeps the per-extractor
        interim parquets aligned on ``clip_id`` for the join step.
        """
        logger.info(
            "Building interim for %d extractors: %s",
            len(missing),
            [s.name for s in missing],
        )

        # Collect the full chunked clip stream into memory; for the
        # smallest target dataset (Earnings-22 ~10–25k clips × ~10 s) this
        # is ~10–30 GB of float32 audio — acceptable for an extraction
        # box. For Common-Voice scale, swap to a per-extractor iteration
        # over the raw stream (each extractor reloads). The current
        # design keeps clip_ids deterministic by materializing once.
        clips = list(chunk_clips(self.ensure_raw(), self.cfg.chunking))
        if not clips:
            raise RuntimeError("Raw + chunk produced zero clips — check the dataset config.")
        logger.info("Materialized %d clips for extraction.", len(clips))

        for spec in missing:
            extractor = self._instantiate(spec)
            try:
                output = extract_interim(
                    extractor,
                    iter(clips),
                    self.interim_parquet_path(spec.name),
                    dataset_spec=self._dataset_spec(num_examples=len(clips)),
                    config_hash=self.cfg.config_hash,
                    write_cfg=self.cfg.write,
                )
                paths[spec.name] = output
                self._try_push(self.interim_dir(spec.name), f"interim/{spec.name}")
            finally:
                # Free GPU memory before loading the next model.
                self._cached_extractors.pop(spec.name, None)
                del extractor
                _free_cuda()

    def _instantiate(self, spec: "ExtractorSpecLike") -> BaseFeatureExtractor:
        if spec.name in self._cached_extractors:
            return self._cached_extractors[spec.name]
        if self._extractor_factory is None:
            raise RuntimeError(
                f"Need to instantiate extractor '{spec.name}' but no factory was "
                "provided. Pass `extractor_factory=...` when constructing DataPipeline."
            )
        ext = self._extractor_factory(spec)
        self._cached_extractors[spec.name] = ext
        return ext

    # ------------------------------------------------------------------
    # internal: manifests
    # ------------------------------------------------------------------

    def _dataset_spec(self, num_examples: Optional[int] = None) -> DatasetSpec:
        ds = to_dataset_spec(self.cfg.raw, num_examples=num_examples)
        # Carry the chunking config through so manifests for chunked
        # datasets describe the segmentation rule.
        return DatasetSpec(
            hf_id=ds.hf_id,
            revision=ds.revision,
            split=ds.split,
            config=ds.config,
            num_examples=num_examples,
            chunking=self.cfg.chunking.to_manifest(),
        )

    def _expected_interim_manifest(self, spec: "ExtractorSpecLike") -> Manifest:
        return Manifest.build(
            stage="interim",
            dataset=self._dataset_spec(),
            extractors=[spec.to_manifest_entry()],
            config_hash=self.cfg.config_hash,
        )

    def _expected_processed_manifest(self) -> Manifest:
        return Manifest.build(
            stage="processed",
            dataset=self._dataset_spec(),
            extractors=[s.to_manifest_entry() for s in self.extractor_specs],
            config_hash=self.cfg.config_hash,
        )

    def _extractor_entries_with_dims(
        self,
        interim_paths: Mapping[str, Path],
    ) -> List[ExtractorEntry]:
        """Read embedding_dim from each interim manifest so the processed manifest carries it."""
        entries: list[ExtractorEntry] = []
        for spec in self.extractor_specs:
            sidecar = interim_paths[spec.name].parent / MANIFEST_FILENAME
            entry = spec.to_manifest_entry()
            if sidecar.exists():
                m = Manifest.read_yaml(sidecar)
                if m.extractors:
                    found = m.extractors[0]
                    entry = ExtractorEntry(
                        name=entry.name,
                        hf_model_id=entry.hf_model_id,
                        hf_revision=entry.hf_revision,
                        embedding_dim=found.embedding_dim,
                        sample_rate=entry.sample_rate,
                    )
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # internal: cache + hub helpers
    # ------------------------------------------------------------------

    def _artifact_ok(self, parquet_path: Path, expected: Manifest) -> bool:
        if not parquet_path.exists():
            return False
        sidecar = parquet_path.parent / MANIFEST_FILENAME
        if not sidecar.exists():
            return False
        try:
            found = Manifest.read_yaml(sidecar)
        except (yaml.YAMLError, OSError) as e:
            logger.warning("Could not read manifest %s (%s); treating as miss.", sidecar, e)
            return False
        return self._compare(expected, found, label=str(sidecar))

    def _compare(self, expected: Manifest, found: Manifest, *, label: str) -> bool:
        if expected.matches(found, ignore=self.cfg.manifest_policy.ignore_fields):
            return True
        diff = expected.diff(found)
        msg = f"Manifest mismatch at {label}: disagreeing fields {sorted(diff.keys())}"
        if self.cfg.manifest_policy.mode == "strict":
            logger.warning("%s — treating as miss (strict mode).", msg)
            return False
        if self.cfg.manifest_policy.mode == "warn":
            logger.warning("%s — accepting anyway (warn mode).", msg)
            return True
        # off
        return True

    def _try_pull(self, local_dir: Path, remote_subdir: str, expected: Manifest) -> bool:
        hub = self.cfg.storage.hub
        if hub is None or not hub.pull:
            return False
        repo_id = hub.resolved_repo_id(self.cfg.storage.dataset_key)
        try:
            return pull_artifact(
                repo_id=repo_id,
                remote_subdir=remote_subdir,
                local_dir=local_dir,
                expected_manifest=expected,
                config=hub,
                ignore_for_match=self.cfg.manifest_policy.ignore_fields,
            )
        except ManifestMismatchError as e:
            if self.cfg.manifest_policy.mode == "strict":
                raise
            logger.warning("Manifest mismatch on pull (%s mode): %s", self.cfg.manifest_policy.mode, e)
            return self.cfg.manifest_policy.mode == "warn"

    def _try_push(self, local_dir: Path, remote_subdir: str) -> None:
        hub = self.cfg.storage.hub
        if hub is None or not hub.push:
            return
        repo_id = hub.resolved_repo_id(self.cfg.storage.dataset_key)
        try:
            push_artifact(
                local_dir=local_dir,
                repo_id=repo_id,
                remote_subdir=remote_subdir,
                config=hub,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Push to %s/%s failed (%s); continuing without push.", repo_id, remote_subdir, e)


# ---------------------------------------------------------------------------
# helpers + types
# ---------------------------------------------------------------------------


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ``ExtractorSpecLike`` is a minimal duck-typed protocol — anything with
# .name + .to_manifest_entry() works. In practice this is
# :class:`ExtractorSpec` from ``extractors.base``.
ExtractorSpecLike = "ExtractorSpec"  # type: ignore[assignment]
