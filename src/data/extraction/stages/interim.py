"""Interim stage — extract embeddings + greedy transcripts per (dataset × extractor).

One parquet per extractor, written under ``{root}/interim/{dataset_key}/{extractor_name}/features.parquet``.
Schema:

    clip_id            : str
    ground_truth       : str
    {extractor}_features      : list[list[float16]]  variable-length (T, D)
    {extractor}_transcription : str                  greedy decode
    {extractor}_wer           : float32              vs. ground_truth

A sidecar ``manifest.yaml`` is written alongside the parquet, and a
JSON-encoded copy is embedded in the parquet's Arrow schema metadata.

Memory: features are written incrementally with PyArrow's
``ParquetWriter``, so peak RAM stays ~one row group regardless of
dataset size. Default row-group size matches the legacy
``parquet_cache`` choice (256) so existing lazy-mode loaders work
without re-chunking.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.extraction.extractors.base import BaseFeatureExtractor, extractor_environment
from src.data.extraction.manifest import DatasetSpec, Manifest
from src.data.extraction.stages.raw import RawClip
from src.utils.git import git_state

logger = logging.getLogger(__name__)

DEFAULT_ROW_GROUP_SIZE = 256


@dataclass
class InterimWriteConfig:
    batch_size: int = 8
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE
    deterministic: bool = True


def feature_columns(extractor_name: str) -> dict[str, str]:
    """Logical → physical column name mapping for one extractor's interim parquet."""
    return {
        "features": f"{extractor_name}_features",
        "wer": f"{extractor_name}_wer",
        "transcription": f"{extractor_name}_transcription",
    }


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def extract_interim(
    extractor: BaseFeatureExtractor,
    clips: Iterable[RawClip],
    output_path: str | Path,
    *,
    dataset_spec: DatasetSpec,
    config_hash: Optional[str] = None,
    write_cfg: Optional[InterimWriteConfig] = None,
) -> Path:
    """Run ``extractor`` over ``clips`` and write the interim parquet.

    ``dataset_spec`` and ``config_hash`` flow into the manifest so the
    artifact records exactly which dataset version + Hydra config it
    came from.
    """
    write_cfg = write_cfg or InterimWriteConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = feature_columns(extractor.name)
    schema_no_meta = pa.schema(
        [
            ("clip_id", pa.string()),
            ("ground_truth", pa.string()),
            (cols["features"], pa.list_(pa.list_(pa.float16()))),
            (cols["transcription"], pa.string()),
            (cols["wer"], pa.float32()),
        ]
    )

    n_written = 0
    start_t = time.monotonic()
    writer: Optional[pq.ParquetWriter] = None

    env = extractor_environment(deterministic=write_cfg.deterministic) if write_cfg.deterministic else nullcontext()

    with env:
        for batch in _batched(clips, write_cfg.batch_size):
            audios = [c.audio for c in batch]
            srs = [c.sample_rate for c in batch]
            try:
                features = extractor.encode_batch(audios, srs)
                hyps = extractor.transcribe_batch(audios, srs)
            except Exception as e:  # noqa: BLE001 — log + skip dead clips
                logger.warning(
                    "Extractor %s failed on batch starting at clip %s (%d clips): %s. Skipping.",
                    extractor.name,
                    batch[0].clip_id if batch else "<empty>",
                    len(batch),
                    e,
                )
                continue

            wers = [_wer(c.ground_truth, h) for c, h in zip(batch, hyps)]

            arrays = {
                "clip_id": pa.array([c.clip_id for c in batch], type=pa.string()),
                "ground_truth": pa.array([c.ground_truth for c in batch], type=pa.string()),
                cols["features"]: _features_to_arrow(features),
                cols["transcription"]: pa.array(hyps, type=pa.string()),
                cols["wer"]: pa.array(np.asarray(wers, dtype=np.float32)),
            }
            arrow_batch = pa.RecordBatch.from_arrays(
                [arrays[name] for name in schema_no_meta.names],
                schema=schema_no_meta,
            )

            if writer is None:
                # Embed the manifest in the schema once we know the
                # extractor's embedding_dim (set on first ``encode``).
                manifest = Manifest.build(
                    stage="interim",
                    dataset=DatasetSpec(
                        hf_id=dataset_spec.hf_id,
                        revision=dataset_spec.revision,
                        split=dataset_spec.split,
                        config=dataset_spec.config,
                        num_examples=dataset_spec.num_examples,
                        chunking=dataset_spec.chunking,
                    ),
                    extractors=[extractor.to_manifest_entry()],
                    config_hash=config_hash,
                )
                schema = manifest.attach_to_schema(schema_no_meta)
                writer = pq.ParquetWriter(output_path, schema, compression="zstd")

            writer.write_batch(arrow_batch, row_group_size=write_cfg.row_group_size)
            n_written += len(batch)

    if writer is None:
        raise RuntimeError(
            f"No clips were extracted for {extractor.name} → {output_path}. "
            "The clip iterator was empty or every batch failed."
        )

    writer.close()

    # Refresh the manifest once we know the final num_examples and the
    # discovered embedding dim, then write the YAML sidecar.
    final_manifest = Manifest.build(
        stage="interim",
        dataset=DatasetSpec(
            hf_id=dataset_spec.hf_id,
            revision=dataset_spec.revision,
            split=dataset_spec.split,
            config=dataset_spec.config,
            num_examples=n_written,
            chunking=dataset_spec.chunking,
        ),
        extractors=[extractor.to_manifest_entry()],
        config_hash=config_hash,
    )
    final_manifest.write_yaml(output_path.parent / "manifest.yaml")

    elapsed = time.monotonic() - start_t
    logger.info(
        "Interim parquet written: %s (%d clips, %.1fs, git=%s)",
        output_path,
        n_written,
        elapsed,
        (git_state().get("commit") or "<no-git>")[:7],
    )
    return output_path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _batched(iterable: Iterable[RawClip], n: int) -> Iterator[List[RawClip]]:
    if n <= 0:
        raise ValueError(f"batch_size must be positive, got {n}")
    buf: List[RawClip] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _features_to_arrow(features: Sequence[np.ndarray]) -> pa.Array:
    """Pack a list of (T_i, D) float16 arrays into Arrow ``list<list<float16>>``."""
    py_lists = []
    for arr in features:
        if arr.ndim != 2:
            raise ValueError(f"Expected (T, D) features, got shape {arr.shape}")
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        py_lists.append([row.tolist() for row in arr])
    return pa.array(py_lists, type=pa.list_(pa.list_(pa.float16())))


def _wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via :mod:`jiwer`. Returns 1.0 on empty references."""
    from jiwer import wer

    ref = (reference or "").strip().lower()
    hyp = (hypothesis or "").strip().lower()
    if not ref:
        return 1.0 if hyp else 0.0
    try:
        return float(wer(ref, hyp))
    except Exception as e:  # noqa: BLE001
        logger.debug("jiwer.wer failed (%s); falling back to 1.0", e)
        return 1.0
