"""Raw stage — load audio + reference text from a HuggingFace dataset.

This stage does *not* materialize the audio to local disk: HF
``datasets`` already has a content-addressed cache under ``HF_HOME``
that we'd just be duplicating. Instead, the stage exposes a normalized
iterator interface that downstream stages (``chunk``, ``interim``)
consume.

The dataset revision is **always** snapshotted into the manifest, so
the artifact remains reproducible even though the raw bytes live in
the HF cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Iterator, Mapping, Optional

import numpy as np

from src.data.extraction.manifest import DatasetSpec

logger = logging.getLogger(__name__)


@dataclass
class RawClip:
    """One unit of raw audio + reference, before chunking / extraction."""

    clip_id: str
    audio: np.ndarray              # (T,) raw waveform — caller is responsible for SR
    sample_rate: int
    ground_truth: str
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RawDatasetConfig:
    """Hydra-friendly config for :func:`load_raw_dataset`.

    Mirrors the public surface of ``datasets.load_dataset`` for the
    fields we care about, plus a few project-specific knobs
    (``id_field``, ``text_field``, ``audio_field``, ``select_n``).
    """

    hf_id: str
    revision: Optional[str] = None
    split: str = "train"
    config: Optional[str] = None
    id_field: str = "id"
    text_field: str = "text"           # field carrying the ground-truth transcription
    audio_field: str = "audio"         # field carrying the {array, sampling_rate} dict
    select_n: Optional[int] = None     # ``None`` = use whole split; useful for smoke tests
    streaming: bool = False
    trust_remote_code: bool = False
    # Per-dataset post-load filter: a Python expression evaluated on the
    # raw row. Empty string disables filtering.
    filter_expr: str = ""


def to_dataset_spec(cfg: RawDatasetConfig, num_examples: Optional[int] = None) -> DatasetSpec:
    return DatasetSpec(
        hf_id=cfg.hf_id,
        revision=cfg.revision,
        split=cfg.split,
        config=cfg.config,
        num_examples=num_examples,
    )


def load_raw_dataset(cfg: RawDatasetConfig) -> Iterator[RawClip]:
    """Yield :class:`RawClip` rows for the configured HF dataset.

    Uses ``datasets.load_dataset`` under the hood. With ``streaming=True``
    the raw audio is fetched on demand — appropriate when you don't
    want a full local copy. Without streaming, HF caches the whole
    split into ``HF_HOME``.
    """
    from datasets import load_dataset

    logger.info(
        "Loading HF dataset %s%s (split=%s, revision=%s, streaming=%s, select_n=%s)",
        cfg.hf_id,
        f" config={cfg.config}" if cfg.config else "",
        cfg.split,
        cfg.revision or "<unpinned>",
        cfg.streaming,
        cfg.select_n,
    )

    if cfg.revision is None:
        logger.warning(
            "Loading %s without a pinned revision — manifest will record None and "
            "future loads may pick up an updated dataset commit.",
            cfg.hf_id,
        )

    ds = load_dataset(
        cfg.hf_id,
        cfg.config,
        split=cfg.split,
        revision=cfg.revision,
        streaming=cfg.streaming,
        trust_remote_code=cfg.trust_remote_code,
    )

    if cfg.filter_expr:
        ds = ds.filter(lambda row: bool(eval(cfg.filter_expr, {}, {"row": row})))  # noqa: S307

    if cfg.select_n is not None and not cfg.streaming:
        ds = ds.select(range(min(cfg.select_n, len(ds))))
    elif cfg.select_n is not None and cfg.streaming:
        ds = ds.take(cfg.select_n)

    for i, row in enumerate(ds):
        audio_field = row.get(cfg.audio_field)
        if audio_field is None:
            raise KeyError(
                f"Row {i} missing audio field '{cfg.audio_field}'. "
                f"Available keys: {list(row.keys())}"
            )
        array = np.asarray(audio_field["array"])
        sr = int(audio_field["sampling_rate"])
        clip_id = str(row.get(cfg.id_field, i))
        text = str(row.get(cfg.text_field, "")).strip()

        # Strip the (large) audio dict from extras before forwarding —
        # downstream stages only need the metadata, not a duplicate copy.
        extra = {k: v for k, v in row.items() if k not in {cfg.audio_field, cfg.text_field}}
        yield RawClip(
            clip_id=clip_id,
            audio=array,
            sample_rate=sr,
            ground_truth=text,
            extra=extra,
        )
