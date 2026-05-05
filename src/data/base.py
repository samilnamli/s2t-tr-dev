"""Parquet-backed Lightning DataModule for ASR routing experiments.

ASRDataModule
    AMIDataModule   (src/data/ami.py)
    SyntheticDataModule  (future)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from src.data.dataset import ASRFeatureDataset, MODEL_NAMES, collate_fn


# ---------------------------------------------------------------------------
# Hydra structured config
# ---------------------------------------------------------------------------

@dataclass
class ASRDataModuleConfig:
    """Config for ASRDataModule.  _target_ is set by concrete subclass configs."""
    _target_: str = "src.data.base.ASRDataModule"
    parquet_path: str = "data/processed/synthetic/combined_features.parquet"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    batch_size: int = 64
    num_workers: int = 4
    max_seq_len: int = 2000
    eager_load: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ASRDataModule(pl.LightningDataModule):
    """Parquet-backed DataModule with deterministic random train/val/test split.

    Splits are derived by a seeded shuffle of the dataset indices so that
    the same seed always yields the same splits regardless of the parquet
    row ordering.

    Properties available after setup():
        model_dims       — dict[name, int] of encoder hidden sizes
        wer_train_matrix — (N_train, K) used by weighted baselines
        class_priors     — list[float] model-selection frequency on train split
    """

    def __init__(
        self,
        parquet_path: str = "data/processed/synthetic/combined_features.parquet",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        max_seq_len: int = 2000,
        eager_load: bool = False,
        seed: int = 42,
        **kwargs,  # absorb _target_ and any other Hydra-injected keys
    ):
        super().__init__()
        self.cfg = ASRDataModuleConfig(
            parquet_path=parquet_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            max_seq_len=max_seq_len,
            eager_load=eager_load,
            seed=seed,
        )
        self._dataset: Optional[ASRFeatureDataset] = None
        self._train: Optional[Subset] = None
        self._val: Optional[Subset] = None
        self._test: Optional[Subset] = None

    # ------------------------------------------------------------------
    # prepare_data / setup
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        if not Path(self.cfg.parquet_path).exists():
            raise FileNotFoundError(
                f"Parquet not found: {self.cfg.parquet_path}. "
                "Run the data preparation pipeline first."
            )

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        if self._dataset is None:
            self._dataset = ASRFeatureDataset(
                parquet_path=self.cfg.parquet_path,
                max_seq_len=self.cfg.max_seq_len,
                eager_load=self.cfg.eager_load,
            )
            N = len(self._dataset)
            rng = np.random.default_rng(self.cfg.seed)
            idx = rng.permutation(N).tolist()

            n_train = int(N * self.cfg.train_ratio)
            n_val = int(N * self.cfg.val_ratio)
            train_idx = idx[:n_train]
            val_idx = idx[n_train : n_train + n_val]
            test_idx = idx[n_train + n_val :]

            self._train = Subset(self._dataset, train_idx)
            self._val = Subset(self._dataset, val_idx)
            self._test = Subset(self._dataset, test_idx)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    # ------------------------------------------------------------------
    # Properties used by selectors / baselines
    # ------------------------------------------------------------------

    @property
    def model_dims(self) -> dict[str, int]:
        """Dict[model_name, hidden_dim] read from the dataset."""
        self._ensure_setup()
        ds = self._dataset
        if ds.eager_load:
            return {name: ds._model_dims[name] for name in MODEL_NAMES}
        # Lazy mode: read one sample to discover dims
        sample = ds[0]
        return {name: sample["hidden_states"][name].shape[-1] for name in MODEL_NAMES}

    @property
    def wer_train_matrix(self) -> np.ndarray:
        """(N_train, K) float32 — for fitting weighted baselines."""
        self._ensure_setup()
        idx = self._train.indices
        return self._dataset.wer_matrix[idx]

    @property
    def class_priors(self) -> list[float]:
        """Model selection frequency on train split (argmin WER).

        Used by class_balanced_loss in TrainableLightningSelector.
        """
        self._ensure_setup()
        wer = self.wer_train_matrix
        best = wer.argmin(axis=-1)
        K = len(MODEL_NAMES)
        counts = np.bincount(best, minlength=K).astype(float)
        total = counts.sum()
        return (counts / total).tolist() if total > 0 else [1.0 / K] * K

    def _ensure_setup(self) -> None:
        if self._dataset is None:
            self.setup()

    # ------------------------------------------------------------------
    # Convenience for BaselineEvaluator / metrics
    # ------------------------------------------------------------------

    @property
    def train_dataset(self):
        self._ensure_setup()
        return self._train

    @property
    def val_dataset(self):
        self._ensure_setup()
        return self._val

    @property
    def test_dataset(self):
        self._ensure_setup()
        return self._test
