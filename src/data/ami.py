"""AMI DataModule.

The parquet is expected at ``cfg.parquet_path`` (default:
``data/processed/edinburghcstr_ami/combined_features_with_transcripts.parquet``).

To generate it locally run::

    python -m src.data.get_processed   # downloads from Google Drive
    python -m src.data.preprocess      # builds the combined parquet

This module only loads the parquet — it does not produce it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.data.base import ASRDataModule, ASRDataModuleConfig


AMI_DEFAULT_PARQUET = (
    "data/processed/edinburghcstr_ami/combined_features_with_transcripts.parquet"
)


@dataclass
class AMIDataModuleConfig(ASRDataModuleConfig):
    _target_: str = "src.data.ami.AMIDataModule"
    parquet_path: str = AMI_DEFAULT_PARQUET
    # AMI-specific defaults (can be overridden via yaml/CLI)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    batch_size: int = 128
    num_workers: int = 4
    max_seq_len: int = 2000
    eager_load: bool = False   # set True on high-RAM GPU hosts


class AMIDataModule(ASRDataModule):
    """ASRDataModule specialised for the AMI dataset.

    prepare_data() raises a clear error if the parquet is missing, with
    instructions on how to produce it.

    TODO: optionally auto-download via src.data.get_processed when
    ``auto_download=True`` is added to AMIDataModuleConfig.
    """

    def __init__(
        self,
        parquet_path: str = AMI_DEFAULT_PARQUET,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 128,
        num_workers: int = 4,
        max_seq_len: int = 2000,
        eager_load: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            parquet_path=parquet_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            max_seq_len=max_seq_len,
            eager_load=eager_load,
            seed=seed,
            **kwargs,
        )

    def prepare_data(self) -> None:
        path = Path(self.cfg.parquet_path)
        if not path.exists():
            raise FileNotFoundError(
                f"AMI parquet not found at: {path}\n\n"
                "To produce it:\n"
                "  1. python -m src.data.get_processed   # download from Google Drive\n"
                "  2. python -m src.data.preprocess      # build combined parquet\n\n"
                "Or point cfg.parquet_path to an existing file."
            )
