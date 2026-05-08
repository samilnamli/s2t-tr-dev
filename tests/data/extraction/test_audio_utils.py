"""Audio normalization + chunking helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.extraction.extractors.audio_utils import (
    fixed_window_chunks,
    normalize_audio,
    to_float32,
    to_mono,
)


def test_to_mono_handles_both_axis_orders() -> None:
    # (T, C)
    assert to_mono(np.ones((100, 2), dtype=np.float32)).shape == (100,)
    # (C, T)
    assert to_mono(np.ones((2, 100), dtype=np.float32)).shape == (100,)
    # (T,) passthrough
    assert to_mono(np.ones(100, dtype=np.float32)).shape == (100,)


def test_to_float32_int16() -> None:
    audio = np.array([-32768, 0, 32767], dtype=np.int16)
    out = to_float32(audio)
    assert out.dtype == np.float32
    assert -1.001 <= out.min() <= -0.99
    assert 0.99 <= out.max() <= 1.001


def test_normalize_audio_resamples_and_peaks() -> None:
    sr = 8000
    audio = np.sin(2 * np.pi * 440 * np.arange(8000) / sr).astype(np.float32) * 2.0
    out = normalize_audio(audio, sr=sr, target_sr=16000, peak=0.99)
    assert out.dtype == np.float32
    assert len(out) == pytest.approx(16000, rel=0.01)
    assert abs(out).max() <= 0.99 + 1e-6


def test_fixed_window_chunks_no_overlap_full() -> None:
    audio = np.arange(16000 * 25, dtype=np.float32)  # 25 s @ 16 kHz
    starts_ends = list(fixed_window_chunks(audio, 16000, seconds=10.0, overlap=0.0))
    assert starts_ends == [(0, 160000), (160000, 320000), (320000, 400000)]


def test_fixed_window_chunks_drops_tiny_tail() -> None:
    audio = np.zeros(16000 * 11, dtype=np.float32)  # 11 s
    starts_ends = list(
        fixed_window_chunks(audio, 16000, seconds=10.0, overlap=0.0, drop_last_below=0.5)
    )
    # 1 s tail < 5 s threshold ⇒ dropped, only the 10 s window remains.
    assert starts_ends == [(0, 160000)]


def test_fixed_window_chunks_overlap() -> None:
    audio = np.zeros(16000 * 30, dtype=np.float32)
    starts = [s for s, _ in fixed_window_chunks(audio, 16000, seconds=10.0, overlap=0.5)]
    # 50% overlap on 10 s windows ⇒ 5 s hop.
    assert starts[:3] == [0, 80000, 160000]
