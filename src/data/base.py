"""Parquet-backed Lightning DataModule for ASR routing experiments.

Subclasses implement :meth:`prepare_data` to ensure the parquet at
``self.parquet_path`` exists (build it if missing). The base class
loads the parquet, builds deterministic train/val/test splits, and
exposes selector-friendly properties.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import ASRFeatureDataset, make_collate_fn


class ASRDataModule(pl.LightningDataModule):
    """Parquet-backed DataModule with deterministic seeded splits.

    Properties available after ``setup()``:
        model_dims       — dict[name, int] of encoder hidden sizes
        wer_train_matrix — (N_train, K) used by weighted baselines
        class_priors     — list[float] argmin-WER frequency on train split
    """

    def __init__(
        self,
        parquet_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        max_seq_len: int = 2000,
        eager_load: bool = False,
        seed: int = 42,
        **_kwargs,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        # In eager mode the features already live in parent-process RAM and are
        # shared with workers via fork+COW, so worker IPC is pure overhead.
        # Force ``num_workers=0`` here to avoid pickling tensors back to the
        # main process.
        self.num_workers = 0 if eager_load else num_workers
        self.max_seq_len = max_seq_len
        self.eager_load = eager_load
        self.seed = seed

        self._dataset: Optional[ASRFeatureDataset] = None
        self._train: Optional[Subset] = None
        self._val: Optional[Subset] = None
        self._test: Optional[Subset] = None

    def prepare_data(self) -> None:
        """Default: require the parquet to exist. Subclasses override."""
        if not Path(self.parquet_path).exists():
            raise FileNotFoundError(
                f"Parquet not found: {self.parquet_path}. "
                "Subclass should override prepare_data() to build it."
            )

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        if self._dataset is not None:
            return
        self._dataset = ASRFeatureDataset(
            parquet_path=self.parquet_path,
            max_seq_len=self.max_seq_len,
            eager_load=self.eager_load,
        )
        self.reseed_split(self.seed)

    def reseed_split(self, split_seed: int) -> None:
        """Re-permute the train/val/test indices using a new seed.

        Cheap (one ``np.permutation`` over an int array) and idempotent.
        Crucially does **not** rebuild :class:`ASRFeatureDataset`, so the
        heavy eager-loaded feature buffers (~38 GB on AMI) stay in RAM and
        are shared across the per-seed runs of every method. This is what
        makes the per-seed split refactor compatible with ``eager_load=True``
        on Colab without re-decoding the parquet for every (method, seed).
        """
        if self._dataset is None:
            self.setup()
            return
        n = len(self._dataset)
        rng = np.random.default_rng(int(split_seed))
        idx = rng.permutation(n).tolist()

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        self._train = Subset(self._dataset, idx[:n_train])
        self._val = Subset(self._dataset, idx[n_train : n_train + n_val])
        self._test = Subset(self._dataset, idx[n_train + n_val :])
        self.seed = int(split_seed)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:  # noqa: ARG002
        """Cast float16 hidden states to float32 once per batch on-device.

        The eager-mode dataset returns float16 tensors so we don't pay a
        per-clip CPU upcast in ``__getitem__``. Lightning's
        ``transfer_batch_to_device`` has already moved the batch to the
        accelerator at this point, so the cast happens on GPU and is
        essentially free compared to the avoided CPU work.
        """
        if isinstance(batch, dict):
            hs = batch.get("hidden_states")
            if isinstance(hs, dict):
                batch["hidden_states"] = {
                    k: (v.float() if v.dtype == torch.float16 else v)
                    for k, v in hs.items()
                }
        return batch

    def _loader(self, subset: Subset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=make_collate_fn(self._dataset.model_names),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)

    @property
    def model_dims(self) -> dict[str, int]:
        self._ensure_setup()
        ds = self._dataset
        if ds.eager_load:
            return {name: ds._model_dims[name] for name in ds.model_names}
        sample = ds[0]
        return {name: sample["hidden_states"][name].shape[-1] for name in ds.model_names}

    @property
    def wer_train_matrix(self) -> np.ndarray:
        self._ensure_setup()
        return self._dataset.wer_matrix[self._train.indices]

    @property
    def class_priors(self) -> list[float]:
        self._ensure_setup()
        wer = self.wer_train_matrix
        best = wer.argmin(axis=-1)
        k = len(self._dataset.model_names)
        counts = np.bincount(best, minlength=k).astype(float)
        total = counts.sum()
        return (counts / total).tolist() if total > 0 else [1.0 / k] * k

    def _ensure_setup(self) -> None:
        if self._dataset is None:
            self.setup()

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
