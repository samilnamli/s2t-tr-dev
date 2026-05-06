"""Helpers for downloading pre-built parquets from Google Drive."""

from __future__ import annotations

import os

from loguru import logger


def download_drive_file(file_id: str, output_path: str) -> None:
    """Download a Google Drive file to ``output_path`` via gdown."""
    import gdown
    import glob
    import shutil

    if os.path.exists(output_path):
        return

    part_files = glob.glob(f"{output_path}*.part")
    if part_files:
        logger.info(f"Renaming part file {part_files[0]} to {output_path}")
        shutil.move(part_files[0], output_path)
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    logger.info("gdown: id={} → {}", file_id, output_path)
    downloaded_path = gdown.download(id=file_id, output=output_path, quiet=False)
    if downloaded_path and downloaded_path != output_path and os.path.exists(downloaded_path):
        logger.info(f"gdown downloaded to {downloaded_path}, renaming to {output_path}")
        shutil.move(downloaded_path, output_path)

    if not os.path.exists(output_path):
        raise RuntimeError(f"gdown download did not produce {output_path}")
