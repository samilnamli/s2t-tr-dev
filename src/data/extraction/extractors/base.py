"""Base classes for frozen ASR feature extractors.

A :class:`BaseFeatureExtractor` wraps one frozen HF ASR model and
exposes two operations:

    encode(audio)     -> (T, D) float16 — frame-level encoder hidden states
    transcribe(audio) ->     str        — greedy decode for WER computation

Both operate on a single 1-D float32 mono waveform at the model's target
sample rate. Batching is done at the stage level (``stages/interim.py``)
so each extractor can implement the per-clip path with minimal ceremony.

Extractors are deliberately stateless after construction (no per-clip
state on ``self``) so the same instance can be threaded through a
multi-clip loop without surprise. Models are frozen (``eval()`` +
``requires_grad_(False)``) on construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterator, Optional

import numpy as np
import torch

from src.data.extraction.extractors.audio_utils import normalize_audio
from src.data.extraction.manifest import ExtractorEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractorSpec:
    """Static description of a frozen ASR model used as a base expert.

    ``name`` is the logical column-name prefix (e.g. ``"wav2vec2_base"``)
    and shows up in the parquet schema and the manifest. ``hf_model_id``
    is the HuggingFace repo. ``hf_revision`` should be a pinned commit
    SHA; an unpinned ``main`` reference is allowed but logged as a
    reproducibility risk.
    """

    name: str
    hf_model_id: str
    hf_revision: Optional[str] = None
    target_sample_rate: int = 16000

    def to_manifest_entry(self, embedding_dim: Optional[int] = None) -> ExtractorEntry:
        return ExtractorEntry(
            name=self.name,
            hf_model_id=self.hf_model_id,
            hf_revision=self.hf_revision,
            embedding_dim=embedding_dim,
            sample_rate=self.target_sample_rate,
        )


class BaseFeatureExtractor(ABC):
    """Abstract frozen ASR extractor.

    Subclasses load HF model + processor in ``__init__`` and implement
    :meth:`_encode_batch` and :meth:`_transcribe_batch`. The public
    :meth:`encode` / :meth:`transcribe` methods handle audio
    normalization and dtype conversion uniformly.
    """

    spec: ExtractorSpec
    device: torch.device
    dtype: torch.dtype
    _embedding_dim: Optional[int]

    def __init__(
        self,
        spec: ExtractorSpec,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.spec = spec
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if self.device.type == "cpu" and dtype == torch.float16:
            # float16 on CPU is a footgun (slow + numerically unstable in
            # several transformers ops); auto-promote to float32.
            logger.info("CPU device with float16 requested; promoting to float32.")
            dtype = torch.float32
        self.dtype = dtype
        self._embedding_dim = None

        if spec.hf_revision is None:
            logger.warning(
                "Extractor '%s' uses unpinned revision for %s — "
                "reproducibility cannot be guaranteed across model-card edits.",
                spec.name,
                spec.hf_model_id,
            )

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _encode_batch(self, batch: list[np.ndarray]) -> list[np.ndarray]:
        """Return list of (T_i, D) float16 arrays — one per input clip."""

    @abstractmethod
    def _transcribe_batch(self, batch: list[np.ndarray]) -> list[str]:
        """Return list of greedy-decoded transcriptions."""

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def target_sample_rate(self) -> int:
        return self.spec.target_sample_rate

    @property
    def embedding_dim(self) -> Optional[int]:
        """Set after the first :meth:`encode` call."""
        return self._embedding_dim

    def prepare_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize a single waveform to float32 mono at the target SR."""
        return normalize_audio(audio, sr, target_sr=self.target_sample_rate)

    def encode(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return self.encode_batch([audio], [sr])[0]

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        return self.transcribe_batch([audio], [sr])[0]

    def encode_batch(self, audios: list[np.ndarray], srs: list[int]) -> list[np.ndarray]:
        prepped = [self.prepare_audio(a, sr) for a, sr in zip(audios, srs)]
        embs = self._encode_batch(prepped)
        if embs and self._embedding_dim is None:
            self._embedding_dim = int(embs[0].shape[-1])
        return embs

    def transcribe_batch(self, audios: list[np.ndarray], srs: list[int]) -> list[str]:
        prepped = [self.prepare_audio(a, sr) for a, sr in zip(audios, srs)]
        return self._transcribe_batch(prepped)

    def to_manifest_entry(self) -> ExtractorEntry:
        return self.spec.to_manifest_entry(self._embedding_dim)

    # ------------------------------------------------------------------
    # bookkeeping helpers for subclasses
    # ------------------------------------------------------------------

    def _freeze(self, module: torch.nn.Module) -> torch.nn.Module:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
        return module.to(device=self.device, dtype=self.dtype)


@contextmanager
def extractor_environment(deterministic: bool = True) -> Iterator[None]:
    """Set torch flags useful for deterministic extraction runs.

    Wrap extraction loops with this to (a) put torch in inference mode
    and (b) ask CUDA for deterministic algorithms. Determinism is
    best-effort — some HF kernels still have nondeterministic CUDA
    paths — but this gets us as close as possible without disabling
    CuDNN entirely.
    """
    prev_inference: Any = None
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    try:
        prev_inference = torch.is_inference_mode_enabled()
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        with torch.inference_mode():
            yield
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark
        del prev_inference
