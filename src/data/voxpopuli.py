"""VoxPopuli DataModule.

Mirror of :class:`AMIDataModule`. Parquet creation from raw audio is
GPU-bound and not implemented in this DataModule; ``prepare_data()``
falls back to a Google Drive download if ``auto_download=True``.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.data.base import ASRDataModule

VOXPOPULI_DEFAULT_PARQUET = (
    "data/processed/facebook_voxpopuli/combined_features_with_transcripts.parquet"
)
VOXPOPULI_DRIVE_FILE_ID = "1yf-G-DWhhZLlqeGXZ77GbuhTBmhqyyXA"


class VoxPopuliDataModule(ASRDataModule):
    """ASRDataModule specialised for VoxPopuli (en_accented test split)."""

    def __init__(
        self,
        parquet_path: str = VOXPOPULI_DEFAULT_PARQUET,
        auto_download: bool = True,
        drive_file_id: str = VOXPOPULI_DRIVE_FILE_ID,
        **kwargs,
    ):
        super().__init__(parquet_path=parquet_path, **kwargs)
        self.auto_download = auto_download
        self.drive_file_id = drive_file_id

    def prepare_data(self) -> None:
        path = Path(self.parquet_path)
        if path.exists():
            return
        if self.auto_download:
            from src.data.downloads import download_drive_file

            logger.info("VoxPopuli parquet missing — downloading from Google Drive ({})", path)
            path.parent.mkdir(parents=True, exist_ok=True)
            download_drive_file(self.drive_file_id, str(path))
            if path.exists():
                return
        raise FileNotFoundError(
            f"VoxPopuli parquet not found at {path} and auto_download did not succeed."
        )
