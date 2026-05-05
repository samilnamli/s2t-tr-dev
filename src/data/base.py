# src/data/datamodule.py
from dataclasses import dataclass
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_dataset, IterableDataset


@dataclass
class DataModuleConfig:
    repo_id: str = "edinburghcstr/ami"
    subset: str = "ihm"
    text_column: str = "text"
    audio_column: str = "audio"
    
    batch_size: int = 16
    num_workers: int = 4
    streaming: bool = True          # key for fast/lazy loading
    
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"


class HFDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        """Called on every GPU. Keep it idempotent."""
        
        if stage == "fit":
            self.train_dataset = load_dataset(
                self.cfg.repo_id,
                self.cfg.subset,
                split=self.cfg.train_split,
                streaming=self.cfg.streaming,   # avoids downloading full dataset
            )
            self.val_dataset = load_dataset(
                self.cfg.repo_id,
                self.cfg.subset,
                split=self.cfg.val_split,
                streaming=self.cfg.streaming,
            )

        if stage == "test":
            self.test_dataset = load_dataset(
                self.cfg.repo_id,
                self.cfg.subset,
                split=self.cfg.test_split,
                streaming=self.cfg.streaming,
            )

    def _collate_fn(self, batch):
        # your tokenization / feature extraction here
        # called per-batch, not upfront
        texts = [item[self.cfg.text_column] for item in batch]
        # ... processor(texts, ...) etc.
        return texts

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
            # shuffle not supported with IterableDataset
            # handle shuffling via .shuffle() on the dataset instead
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn,
        )