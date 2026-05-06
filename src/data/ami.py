"""AMI DataModule.

``prepare_data()`` ensures the AMI parquet exists. If missing, attempts
to download a pre-computed parquet from Google Drive (when
``auto_download=True``). Re-running ASR feature extraction from raw
audio is GPU-bound and lives outside this DataModule.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.data.base import ASRDataModule

AMI_DEFAULT_PARQUET = "data/processed/edinburghcstr_ami/combined_features_with_transcripts.parquet"
AMI_DRIVE_FILE_ID = "1FgN4FQ422HPDCwG-Wl4jrXQGl0tgX6ZP"


class AMIDataModule(ASRDataModule):
    """ASRDataModule specialised for the AMI dataset."""

    def __init__(
        self,
        parquet_path: str = AMI_DEFAULT_PARQUET,
        auto_download: bool = True,
        drive_file_id: str = AMI_DRIVE_FILE_ID,
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

            logger.info("AMI parquet missing — downloading from Google Drive ({})", path)
            path.parent.mkdir(parents=True, exist_ok=True)
            download_drive_file(self.drive_file_id, str(path))
            if path.exists():
                return
        raise FileNotFoundError(
            f"AMI parquet not found at {path} and auto_download did not succeed. "
            "Either set auto_download=true with a valid drive_file_id, or place "
            "the parquet at the configured path."
        )
