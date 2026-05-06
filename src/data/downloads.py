"""Helpers for downloading pre-built parquets from Google Drive."""

from __future__ import annotations

import os

from loguru import logger


def download_drive_file(file_id: str, output_path: str) -> None:
    """Download a Google Drive file to ``output_path`` via gdown."""
    import gdown

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    logger.info("gdown: id={} → {}", file_id, output_path)
    gdown.download(id=file_id, output=output_path, quiet=False)
    if not os.path.exists(output_path):
        raise RuntimeError(f"gdown download did not produce {output_path}")
