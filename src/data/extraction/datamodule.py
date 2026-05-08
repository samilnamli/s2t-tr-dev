"""HFASRDataModule — Lightning DataModule backed by the extraction pipeline.

``prepare_data()`` delegates to :meth:`DataPipeline.ensure_processed`,
so the rest of the training stack (``setup`` / dataloaders / model
dims) is exactly :class:`ASRDataModule` with no further plumbing.

Hydra wires this together:

    _target_: src.data.extraction.datamodule.HFASRDataModule
    pipeline:
      _target_: src.data.extraction.datamodule.build_pipeline
      ...
    batch_size: 64
    num_workers: 4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf

from src.data.base import ASRDataModule
from src.data.extraction.extractors import (
    BaseFeatureExtractor,
    CTCExtractor,
    ExtractorSpec,
    Seq2SeqExtractor,
)
from src.data.extraction.manifest import hash_config
from src.data.extraction.pipeline import DataPipeline, PipelineConfig

logger = logging.getLogger(__name__)


class HFASRDataModule(ASRDataModule):
    """ASR routing DataModule sourced from a HuggingFace dataset.

    The DataModule does not own the heavy extraction work — it
    delegates to a :class:`DataPipeline` whose :meth:`ensure_processed`
    is called once during :meth:`prepare_data`. After that the parent
    class handles dataset construction, splits, and dataloaders.
    """

    def __init__(
        self,
        pipeline: DataPipeline,
        *,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        max_seq_len: int = 2000,
        eager_load: bool = False,
        seed: int = 42,
        **_kwargs,
    ):
        # Compute the parquet path *before* calling super so the parent
        # has a path to point at; the file may not exist yet (built in
        # prepare_data), but the path is deterministic.
        parquet_path = str(pipeline.processed_parquet_path)
        super().__init__(
            parquet_path=parquet_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            max_seq_len=max_seq_len,
            eager_load=eager_load,
            seed=seed,
        )
        self.pipeline = pipeline

    def prepare_data(self) -> None:
        path = self.pipeline.ensure_processed()
        # In case ensure_processed() returned a different path than the
        # one we stored at init (it shouldn't, but defensively):
        self.parquet_path = str(path)


# ---------------------------------------------------------------------------
# Hydra factory helpers
# ---------------------------------------------------------------------------


def build_pipeline(
    pipeline_cfg: PipelineConfig | DictConfig,
    extractors: List[ExtractorSpec | DictConfig],
    *,
    device: str = "cuda",
    dtype: str = "float16",
    full_resolved_config: Optional[DictConfig | dict] = None,
) -> DataPipeline:
    """Hydra-friendly factory that turns config objects into a :class:`DataPipeline`.

    Accepts already-typed dataclass configs or raw OmegaConf dicts and
    coerces both into the dataclass form. Computes ``config_hash`` from
    the resolved Hydra config (when supplied) so the manifest carries
    a stable provenance hash.
    """
    cfg = _to_pipeline_config(pipeline_cfg)

    if full_resolved_config is not None and cfg.config_hash is None:
        if isinstance(full_resolved_config, DictConfig):
            full_resolved_config = OmegaConf.to_container(full_resolved_config, resolve=True)
        cfg.config_hash = hash_config(full_resolved_config)  # type: ignore[arg-type]

    extractor_specs: list[ExtractorSpec] = [
        _to_extractor_spec(e) for e in extractors
    ]

    factory = _make_extractor_factory(device=device, dtype=dtype)
    return DataPipeline(cfg=cfg, extractor_specs=extractor_specs, extractor_factory=factory)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_pipeline_config(obj: Any) -> PipelineConfig:
    if isinstance(obj, PipelineConfig):
        return obj
    if isinstance(obj, DictConfig):
        return _from_omegaconf(PipelineConfig, obj)
    if isinstance(obj, dict):
        return _from_omegaconf(PipelineConfig, OmegaConf.create(obj))
    raise TypeError(f"Cannot coerce {type(obj).__name__} into PipelineConfig")


def _to_extractor_spec(obj: Any) -> ExtractorSpec:
    if isinstance(obj, ExtractorSpec):
        return obj
    if isinstance(obj, DictConfig):
        return ExtractorSpec(**OmegaConf.to_container(obj, resolve=True))  # type: ignore[arg-type]
    if isinstance(obj, dict):
        return ExtractorSpec(**obj)
    raise TypeError(f"Cannot coerce {type(obj).__name__} into ExtractorSpec")


def _from_omegaconf(dc_type, cfg: DictConfig):
    """Lightweight :class:`DictConfig` → dataclass coercion.

    Hydra's :func:`structured_config` would normally do this, but the
    pipeline configs are nested dataclasses with optional Hub config,
    so a hand-rolled coercion keeps the YAML simple.
    """
    from dataclasses import fields, is_dataclass

    raw = OmegaConf.to_container(cfg, resolve=True)
    return _coerce(dc_type, raw)


def _coerce(dc_type, value):
    from dataclasses import fields, is_dataclass

    if value is None:
        return None
    if not is_dataclass(dc_type):
        return value
    if not isinstance(value, dict):
        return value
    kwargs = {}
    for f in fields(dc_type):
        if f.name in value:
            kwargs[f.name] = _coerce(f.type if isinstance(f.type, type) else _resolve_type(f.type), value[f.name])
    return dc_type(**kwargs)


def _resolve_type(t):
    """Best-effort string-type → class lookup so nested dataclasses coerce.

    Dataclass field types arrive as strings under ``from __future__ import
    annotations``. We don't need full PEP-563 resolution — we just look
    up the few config classes used by the pipeline.
    """
    if isinstance(t, type):
        return t
    if isinstance(t, str):
        from src.data.extraction.hub import HubConfig
        from src.data.extraction.pipeline import (
            ManifestPolicy,
            PipelineConfig,
            StorageConfig,
        )
        from src.data.extraction.stages.chunk import ChunkingConfig
        from src.data.extraction.stages.interim import InterimWriteConfig
        from src.data.extraction.stages.raw import RawDatasetConfig

        return {
            "RawDatasetConfig": RawDatasetConfig,
            "ChunkingConfig": ChunkingConfig,
            "StorageConfig": StorageConfig,
            "InterimWriteConfig": InterimWriteConfig,
            "ManifestPolicy": ManifestPolicy,
            "HubConfig": HubConfig,
            "PipelineConfig": PipelineConfig,
            "Optional[HubConfig]": HubConfig,
        }.get(t, object)
    return object


def _make_extractor_factory(*, device: str, dtype: str):
    """Return a callable ``spec -> BaseFeatureExtractor`` per the chosen dtype."""
    import torch

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]

    def factory(spec: ExtractorSpec) -> BaseFeatureExtractor:
        cls = _resolve_extractor_class(spec)
        return cls(spec, device=device, dtype=torch_dtype)

    return factory


_CTC_FAMILIES = {"wav2vec2", "hubert", "data2vec_audio", "data2vec-audio", "wavlm", "unispeech", "sew"}
_SEQ2SEQ_FAMILIES = {"whisper", "speech_to_text", "speech-to-text", "s2t"}


def _resolve_extractor_class(spec: ExtractorSpec):
    """Pick :class:`CTCExtractor` vs. :class:`Seq2SeqExtractor` from the model name.

    The HF model id is a strong hint (``facebook/wav2vec2-...`` →
    CTC, ``facebook/s2t-...`` → seq2seq). For unusual repos the user
    can subclass and pass an explicit factory to
    :class:`DataPipeline`.
    """
    needle = spec.hf_model_id.lower()
    if any(fam in needle for fam in _CTC_FAMILIES):
        return CTCExtractor
    if any(fam in needle for fam in _SEQ2SEQ_FAMILIES):
        return Seq2SeqExtractor
    raise ValueError(
        f"Cannot infer extractor class for hf_model_id={spec.hf_model_id!r}. "
        "Pass an explicit factory to DataPipeline if the model is from an unusual family."
    )
