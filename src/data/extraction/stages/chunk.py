"""Chunking stage — segment long-form audio into model-friendly clips.

Some HF datasets (Common Voice, GigaSpeech) are pre-segmented into
~5–15 s clips and the chunker is a no-op for them. Others (Earnings-22,
TED-LIUM raw STM tracks) ship one ~30-min recording per row and need
explicit segmentation before extraction. The chunker handles both.

Strategies:
    none           – pass through unchanged (the row is already a clip).
    fixed_window   – deterministic non-overlapping (or overlapping) windows.
    transcript     – split on ground-truth utterance boundaries when the
                     dataset provides them in ``extra``. Falls back to
                     fixed_window when no segmentation hint is present.

The chunking config is captured in the manifest's
``dataset.chunking`` block so the same chunking can be replayed.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterator, Mapping, Optional

from src.data.extraction.extractors.audio_utils import fixed_window_chunks
from src.data.extraction.stages.raw import RawClip

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Hydra-friendly chunking config."""

    method: str = "none"             # "none" | "fixed_window" | "transcript"
    seconds: float = 10.0
    overlap: float = 0.0
    drop_last_below: float = 0.5
    min_seconds: float = 1.0         # discard clips shorter than this (post-chunking)
    max_seconds: float = 30.0        # hard cap; longer clips are window-split

    def to_manifest(self) -> Mapping[str, Any]:
        return {
            "method": self.method,
            "seconds": self.seconds,
            "overlap": self.overlap,
            "drop_last_below": self.drop_last_below,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
        }


def chunk_clips(clips: Iterator[RawClip], cfg: ChunkingConfig) -> Iterator[RawClip]:
    """Yield possibly-segmented clips per the configured strategy."""
    if cfg.method == "none":
        for clip in clips:
            yield from _enforce_bounds(clip, cfg)
        return

    if cfg.method == "fixed_window":
        for clip in clips:
            yield from _fixed_window_split(clip, cfg)
        return

    if cfg.method == "transcript":
        for clip in clips:
            segs = _try_transcript_split(clip, cfg)
            if segs is None:
                yield from _fixed_window_split(clip, cfg)
            else:
                yield from segs
        return

    raise ValueError(f"Unknown chunking method: {cfg.method!r}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fixed_window_split(clip: RawClip, cfg: ChunkingConfig) -> Iterator[RawClip]:
    """Window-split a clip into fixed-length sub-clips.

    Ground truth is *not* re-aligned per sub-clip (we don't have
    timestamps); each sub-clip inherits the whole clip's text. That
    means WER on a window-split dataset is computed against the full
    transcript — caller should be aware. For datasets where this is
    undesirable, prefer the ``transcript`` strategy.
    """
    duration = len(clip.audio) / clip.sample_rate
    if duration <= cfg.max_seconds and duration >= cfg.min_seconds:
        yield clip
        return

    for i, (start, end) in enumerate(
        fixed_window_chunks(
            clip.audio,
            clip.sample_rate,
            seconds=cfg.seconds,
            overlap=cfg.overlap,
            drop_last_below=cfg.drop_last_below,
        )
    ):
        sub = clip.audio[start:end]
        sub_dur = len(sub) / clip.sample_rate
        if sub_dur < cfg.min_seconds:
            continue
        yield RawClip(
            clip_id=f"{clip.clip_id}__w{i:04d}",
            audio=sub,
            sample_rate=clip.sample_rate,
            ground_truth=clip.ground_truth,
            extra={**clip.extra, "_chunk_method": "fixed_window", "_chunk_idx": i},
        )


def _try_transcript_split(clip: RawClip, cfg: ChunkingConfig) -> Optional[list[RawClip]]:
    """Try to segment using per-utterance metadata in ``clip.extra``.

    Returns ``None`` when the row doesn't carry usable segment metadata
    (caller falls back to fixed-window splitting). The expected shape
    is a list under ``extra["segments"]`` of ``{start, end, text}``
    items, where ``start`` / ``end`` are seconds.
    """
    segments = clip.extra.get("segments")
    if not segments or not isinstance(segments, (list, tuple)):
        return None

    out: list[RawClip] = []
    sr = clip.sample_rate
    for i, seg in enumerate(segments):
        try:
            start = int(round(float(seg["start"]) * sr))
            end = int(round(float(seg["end"]) * sr))
            text = str(seg.get("text", "")).strip()
        except (KeyError, TypeError, ValueError):
            return None
        if end <= start:
            continue
        sub = clip.audio[start:end]
        dur = (end - start) / sr
        if dur < cfg.min_seconds or dur > cfg.max_seconds * 2:  # sanity: reject outliers
            continue
        out.append(
            RawClip(
                clip_id=f"{clip.clip_id}__s{i:04d}",
                audio=sub,
                sample_rate=sr,
                ground_truth=text,
                extra={**clip.extra, "_chunk_method": "transcript", "_chunk_idx": i},
            )
        )
    return out


def _enforce_bounds(clip: RawClip, cfg: ChunkingConfig) -> Iterator[RawClip]:
    """Drop or window-split clips that violate min/max bounds, even with method=none."""
    duration = len(clip.audio) / clip.sample_rate
    if duration < cfg.min_seconds:
        return
    if duration > cfg.max_seconds:
        yield from _fixed_window_split(clip, cfg)
        return
    yield clip
