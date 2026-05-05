"""Routing evaluation metrics.

Stateless helpers to compute mean WER, selection accuracy, and per-model
selection frequencies from selection indices and a WER matrix.
"""

from __future__ import annotations

import numpy as np


class SelectionMetrics:
    @staticmethod
    def wer_stats(selected_idx: np.ndarray, wer_matrix: np.ndarray) -> dict:
        N = wer_matrix.shape[0]
        selected_wer = wer_matrix[np.arange(N), selected_idx]
        sem = float(selected_wer.std(ddof=1) / np.sqrt(N)) if N > 1 else 0.0
        return {
            "wer_mean": float(selected_wer.mean()),
            "wer_sem": sem,
            "n": int(N),
        }

    @staticmethod
    def selection_accuracy(selected_idx: np.ndarray, wer_matrix: np.ndarray) -> dict:
        oracle_idx = wer_matrix.argmin(axis=-1)
        acc = float(np.mean(selected_idx == oracle_idx))
        return {"selection_accuracy": acc}

    @staticmethod
    def selection_frequencies(
        selected_idx: np.ndarray, model_names: list[str]
    ) -> dict[str, dict[str, float]]:
        K = len(model_names)
        counts = np.bincount(selected_idx, minlength=K)
        total = max(len(selected_idx), 1)
        return {
            name: {"count": int(c), "freq": float(c / total)}
            for name, c in zip(model_names, counts)
        }

    @classmethod
    def compute_all(
        cls,
        selected_idx: np.ndarray,
        wer_matrix: np.ndarray,
        model_names: list[str],
    ) -> dict:
        result: dict = {}
        result.update(cls.wer_stats(selected_idx, wer_matrix))
        result.update(cls.selection_accuracy(selected_idx, wer_matrix))
        result["selection_frequencies"] = cls.selection_frequencies(
            selected_idx, model_names
        )
        return result
