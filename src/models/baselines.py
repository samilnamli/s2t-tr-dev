"""Training-free ASR routing baselines.

All classes inherit TrainingFreeBaseline from src.models.base and implement
select(batch) -> np.ndarray(B,).

Hierarchy
---------
TrainingFreeBaseline (ABC)          – src.models.base
    SingleModelBaseline
    OracleBaseline
    RandomBaseline
    WeightedRandomBaseline          – fit() learns priors from dm.wer_train_matrix
    ROVERBaseline
    WeightedROVERBaseline           – fit() learns priors from dm.wer_train_matrix

SelectionMetrics  → src.utils.metrics (see there)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytorch_lightning as pl

from src.data.dataset import MODEL_NAMES
from src.models.base import TrainingFreeBaseline
from src.utils.rover import rover_combine


# ---------------------------------------------------------------------------
# Hydra structured configs
# ---------------------------------------------------------------------------

@dataclass
class BaseBaselineConfig:
    model_names: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    seed: int = 42


@dataclass
class SingleModelBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.models.baselines.SingleModelBaseline"
    name: str = "single_model"
    model_idx: int = 0


@dataclass
class OracleBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.models.baselines.OracleBaseline"
    name: str = "oracle"


@dataclass
class RandomBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.models.baselines.RandomBaseline"
    name: str = "random"


@dataclass
class WeightedRandomBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.models.baselines.WeightedRandomBaseline"
    name: str = "weighted_random"


@dataclass
class ROVERBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.models.baselines.ROVERBaseline"
    name: str = "rover"


@dataclass
class WeightedROVERBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.models.baselines.WeightedROVERBaseline"
    name: str = "weighted_rover"


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------

class SingleModelBaseline(TrainingFreeBaseline):
    """Always selects the same model."""

    def __init__(self, model_idx: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.model_idx = model_idx

    def select(self, batch: dict) -> np.ndarray:
        N = batch["wer_matrix"].shape[0]
        return np.full(N, self.model_idx, dtype=np.int64)


class OracleBaseline(TrainingFreeBaseline):
    """Selects the model with the lowest WER per sample (upper bound)."""

    def select(self, batch: dict) -> np.ndarray:
        return batch["wer_matrix"].numpy().argmin(axis=-1).astype(np.int64)


class RandomBaseline(TrainingFreeBaseline):
    """Selects a uniformly random model per sample."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(self.seed)

    def select(self, batch: dict) -> np.ndarray:
        N = batch["wer_matrix"].shape[0]
        return self._rng.integers(0, self.K, size=N)


class WeightedRandomBaseline(TrainingFreeBaseline):
    """Samples from a prior over model quality derived from training-split WER."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        N = batch["wer_matrix"].shape[0]
        return self._rng.choice(self.K, size=N, p=self.weights)


class ROVERBaseline(TrainingFreeBaseline):
    """ROVER hypothesis combination with uniform system weights.

    Requires ``"transcription"`` key in the batch (dict[model_name, list[str]]).
    The ``select`` return value is the index of the system closest to the
    ROVER output — used to unify the interface. The meaningful metric is
    the WER of the ROVER transcript itself, computed in evaluate().
    """

    def fit(self, datamodule: pl.LightningDataModule, **_) -> None:
        pass

    def select(self, batch: dict) -> np.ndarray:
        weights = np.full(self.K, 1.0 / self.K, dtype=np.float32)
        return self._rover_select(batch, weights)

    def combine(self, batch: dict) -> list[str]:
        weights = np.full(self.K, 1.0 / self.K, dtype=np.float32)
        return rover_combine(batch["transcription"], self.model_names, weights)

    def _rover_select(self, batch: dict, weights: np.ndarray) -> np.ndarray:
        """Proxy: index of the system whose hypothesis is closest to ROVER output."""
        import jiwer

        rover_hyps = rover_combine(batch["transcription"], self.model_names, weights)
        N = len(rover_hyps)
        dist = np.zeros((N, self.K), dtype=np.float32)
        for k, name in enumerate(self.model_names):
            sys_hyps = batch["transcription"][name]
            for i, (rover_h, sys_h) in enumerate(zip(rover_hyps, sys_hyps)):
                r = rover_h if rover_h.strip() else "<empty>"
                h = sys_h if sys_h.strip() else ""
                dist[i, k] = float(jiwer.wer(r, h))
        return dist.argmin(axis=-1).astype(np.int64)

    def evaluate(self, datamodule: pl.LightningDataModule) -> dict:
        """Compute standard routing metrics; also WER of the ROVER transcript."""
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
