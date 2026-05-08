"""Hydra CLI for the extraction pipeline.

Run with:

    python -m src.data.extraction stage=processed dataset=earnings22

Available top-level overrides (composed via Hydra defaults):

    dataset    – which configs/data/*.yaml to load
    extractor  – which configs/extractor/*.yaml entries to compose
    storage    – which configs/storage/*.yaml to load (Hub config)
    pipeline   – which configs/pipeline/*.yaml to load
    stage      – "raw" | "interim" | "processed" (default: "processed")
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="extract", version_base="1.3")
def main(cfg: DictConfig) -> int:
    from src.data.extraction.datamodule import build_pipeline

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    pipeline = build_pipeline(
        pipeline_cfg=cfg.pipeline,
        extractors=list(cfg.extractors.values()),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "float16"),
        full_resolved_config=cfg,
    )

    stage = str(cfg.get("stage", "processed")).lower()
    if stage == "raw":
        # Just exercise the loader — useful for verifying revision pinning.
        n = 0
        for _ in pipeline.ensure_raw():
            n += 1
            if n >= 5:
                break
        logger.info("Raw stage OK; first 5 clips iterated.")
    elif stage == "interim":
        paths = pipeline.ensure_interim()
        logger.info("Interim parquets ready: %s", {k: str(v) for k, v in paths.items()})
    elif stage == "processed":
        path = pipeline.ensure_processed()
        logger.info("Processed parquet ready: %s", path)
    else:
        raise ValueError(f"Unknown stage: {stage!r}. Choose raw|interim|processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
