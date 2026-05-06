"""Training-free ASR routing baselines.

Hierarchy
---------
TrainingFreeBaseline (ABC)          – src.models.base
    SingleModelBaseline
    OracleBaseline
    RandomBaseline
    WeightedRandomBaseline          – fit() learns priors from dm.wer_train_matrix
    ROVERBaseline
    WeightedROVERBaseline           – fit() learns priors from dm.wer_train_matrix
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytorch_lightning as pl

from src.data.dataset import MODEL_NAMES
from src.models.base import TrainingFreeBaseline
from src.utils.rover import rover_combine


def _model_names(model_names: list[str] | None) -> list[str]:
    return list(model_names) if model_names is not None else list(MODEL_NAMES)


class SingleModelBaseline(TrainingFreeBaseline):
    """Always selects the same model."""

    def __init__(
        self, model_idx: int = 0, model_names: list[str] | None = None, seed: int = 42, **kwargs
    ):
        super().__init__(model_names=_model_names(model_names), seed=seed, **kwargs)
        self.model_idx = model_idx

    def select(self, batch: dict) -> np.ndarray:
        n = batch["wer_matrix"].shape[0]
        return np.full(n, self.model_idx, dtype=np.int64)


class OracleBaseline(TrainingFreeBaseline):
    """Selects the model with the lowest WER per sample (upper bound)."""

    def __init__(self, model_names: list[str] | None = None, seed: int = 42, **kwargs):
        super().__init__(model_names=_model_names(model_names), seed=seed, **kwargs)

    def select(self, batch: dict) -> np.ndarray:
        return batch["wer_matrix"].numpy().argmin(axis=-1).astype(np.int64)


class RandomBaseline(TrainingFreeBaseline):
    """Selects a uniformly random model per sample."""

    def __init__(self, model_names: list[str] | None = None, seed: int = 42, **kwargs):
        super().__init__(model_names=_model_names(model_names), seed=seed, **kwargs)
        self._rng = np.random.default_rng(self.seed)

    def select(self, batch: dict) -> np.ndarray:
        n = batch["wer_matrix"].shape[0]
        return self._rng.integers(0, self.K, size=n)


class WeightedRandomBaseline(TrainingFreeBaseline):
    """Samples from a prior over model quality derived from training-split WER."""

    def __init__(self, model_names: list[str] | None = None, seed: int = 42, **kwargs):
        super().__init__(model_names=_model_names(model_names), seed=seed, **kwargs)
        self._rng = np.random.default_rng(self.seed)
        self.weights: Optional[np.ndarray] = None

    def fit(self, datamodule: pl.LightningDataModule, **_) -> None:
        wer_train = datamodule.wer_train_matrix
        mean = wer_train.mean(axis=0)
        inv = np.maximum(1.0 - mean, 1e-6)
        self.weights = (inv / inv.sum()).astype(np.float32)

    def select(self, batch: dict) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit(datamodule) before select().")
        n = batch["wer_matrix"].shape[0]
        return self._rng.choice(self.K, size=n, p=self.weights)


class ROVERBaseline(TrainingFreeBaseline):
    """ROVER hypothesis combination with uniform system weights.

    Requires ``"transcription"`` key in the batch (dict[model_name, list[str]]).
    """

    def __init__(self, model_names: list[str] | None = None, seed: int = 42, **kwargs):
        super().__init__(model_names=_model_names(model_names), seed=seed, **kwargs)

    def fit(self, datamodule: pl.LightningDataModule, **_) -> None:
        pass

    def select(self, batch: dict) -> np.ndarray:
        weights = np.full(self.K, 1.0 / self.K, dtype=np.float32)
        return self._rover_select(batch, weights)

    def combine(self, batch: dict) -> list[str]:
        weights = np.full(self.K, 1.0 / self.K, dtype=np.float32)
        return rover_combine(batch["transcription"], self.model_names, weights)

    def _rover_select(self, batch: dict, weights: np.ndarray) -> np.ndarray:
        import jiwer

        rover_hyps = rover_combine(batch["transcription"], self.model_names, weights)
        n = len(rover_hyps)
        dist = np.zeros((n, self.K), dtype=np.float32)
        for k, name in enumerate(self.model_names):
            sys_hyps = batch["transcription"][name]
            for i, (rover_h, sys_h) in enumerate(zip(rover_hyps, sys_hyps)):
                r = rover_h if rover_h.strip() else "<empty>"
                h = sys_h if sys_h.strip() else ""
                dist[i, k] = float(jiwer.wer(r, h))
        return dist.argmin(axis=-1).astype(np.int64)

    def evaluate(self, datamodule: pl.LightningDataModule) -> dict:
        import jiwer

        from src.utils.metrics import SelectionMetrics

        datamodule.setup("test")
        all_idx, all_wer, all_gt, all_rover_hyps = [], [], [], []
        for batch in datamodule.test_dataloader():
            all_idx.append(self.select(batch))
            all_wer.append(batch["wer_matrix"].numpy())
            if batch.get("ground_truth") and batch.get("transcription"):
                all_gt.extend(batch["ground_truth"])
                all_rover_hyps.extend(self.combine(batch))

        selected_idx = np.concatenate(all_idx)
        wer_matrix = np.concatenate(all_wer, axis=0)
        metrics = SelectionMetrics.compute_all(selected_idx, wer_matrix, self.model_names)

        if all_gt and all_rover_hyps:
            metrics["rover_wer"] = float(jiwer.wer(all_gt, all_rover_hyps))

        return metrics


class WeightedROVERBaseline(ROVERBaseline):
    """ROVER with per-system weights derived from training-split WER."""

    def __init__(self, model_names: list[str] | None = None, seed: int = 42, **kwargs):
        super().__init__(model_names=_model_names(model_names), seed=seed, **kwargs)
        self.weights: Optional[np.ndarray] = None

    def fit(self, datamodule: pl.LightningDataModule, **_) -> None:
        wer_train = datamodule.wer_train_matrix
        mean = wer_train.mean(axis=0)
        inv = np.maximum(1.0 - mean, 1e-6)
        self.weights = (inv / inv.sum()).astype(np.float32)

    def select(self, batch: dict) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit(datamodule) before select().")
        return self._rover_select(batch, self.weights)

    def combine(self, batch: dict) -> list[str]:
        if self.weights is None:
            raise RuntimeError("Call fit(datamodule) before combine().")
        return rover_combine(batch["transcription"], self.model_names, self.weights)
