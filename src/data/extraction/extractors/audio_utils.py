"""Audio normalization helpers used by extractors and stages.

Centralized so every callsite uses the same conventions: float32 mono
at the target sample rate, with a tunable peak-normalization step.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse multi-channel audio to mono by averaging."""
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"Expected (T,) or (T, C) audio, got shape {audio.shape}")
    # HF datasets returns (C, T) or (T, C) depending on backend — handle both
    # by inspecting which axis is "small" (channels) vs "large" (time).
    if audio.shape[0] < audio.shape[1]:
        return audio.mean(axis=0)
    return audio.mean(axis=1)


def to_float32(audio: np.ndarray) -> np.ndarray:
    """Cast common audio dtypes to float32 in [-1, 1]."""
    if audio.dtype == np.float32:
        return audio
    if audio.dtype == np.float64:
        return audio.astype(np.float32)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    return audio.astype(np.float32)


def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample mono float32 audio to ``target_sr`` if needed.

    Uses :func:`librosa.resample` (kaiser-best) when available, else
    :func:`scipy.signal.resample_poly`. Both are deterministic; results
    differ marginally between backends but are within typical model
    input tolerance.
    """
    if sr == target_sr:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly

        from math import gcd

        g = gcd(sr, target_sr)
        return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


def normalize_audio(
    audio: np.ndarray,
    sr: int,
    *,
    target_sr: int = 16000,
    peak: float = 0.99,
) -> np.ndarray:
    """One-shot pipeline: mono → float32 → resample → peak-normalize.

    ``peak`` of 0.0 disables peak normalization.
    """
    audio = to_mono(audio)
    audio = to_float32(audio)
    audio = resample_if_needed(audio, sr, target_sr)
    if peak > 0.0:
        amax = float(np.max(np.abs(audio))) if audio.size else 0.0
        if amax > 0.0:
            audio = audio * (peak / amax)
    return audio


def fixed_window_chunks(
    audio: np.ndarray,
    sr: int,
    *,
    seconds: float,
    overlap: float = 0.0,
    drop_last_below: float = 0.5,
) -> Iterator[Tuple[int, int]]:
    """Yield ``(start_sample, end_sample)`` indices for fixed-window chunking.

    ``drop_last_below`` discards the trailing chunk if it's shorter than
    that fraction of ``seconds`` — keeps the dataset clean of tiny
    fragments while not throwing away most of a final partial window.
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    win = int(round(seconds * sr))
    hop = max(1, int(round(win * (1.0 - overlap))))
    min_tail = int(round(win * drop_last_below))

    n = audio.shape[0]
    start = 0
    while start < n:
        end = start + win
        if end > n:
            if n - start < min_tail:
                break
            end = n
            yield start, end
            break
        yield start, end
        start += hop
