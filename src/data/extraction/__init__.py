"""ASR feature extraction pipeline.

Package layout:
    extractors/  – frozen-model wrappers (BaseFeatureExtractor + concretes)
    stages/      – raw / chunk / interim / processed builders
    pipeline.py  – DataPipeline orchestrator
    manifest.py  – Manifest dataclass + YAML / Arrow metadata I/O
    hub.py       – HuggingFace Hub push / pull with manifest verification
    datamodule.py – HFASRDataModule (Lightning) — prepare_data() ⟶ pipeline
    cli.py       – `python -m src.data.extraction` CLI entry point
"""
