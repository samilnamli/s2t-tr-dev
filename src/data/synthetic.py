"""Synthetic regime-switch dataset and DataModule.

Each clip contains ``frame_length`` frames split by a switch position
``rho``. The first half is drawn from regime ``r1`` and the second from
``r2 != r1``, sampled uniformly from a finite set of ``num_regimes``.
The per-expert WER scores depend on the *ordered* pair ``(r1, r2)``,
so the optimal expert is not recoverable from any pooled symmetric
summary; frame-level routers can read the ordering from the temporal
sequence and are expected to outperform pooled baselines.

Output schema matches :class:`src.data.dataset.ASRFeatureDataset`
(``hubert_features``, ``whisper_features``, ``w2v2_features`` plus
the corresponding ``_wer`` columns and ``ground_truth``).

Public API:
    build_synthetic_parquet(out_path, **gen_kwargs)  – pure function
    SyntheticDataModule(...)                         – Lightning DataModule
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.base import ASRDataModule

METADATA_KEY_REGIME_CENTERS = b"synthetic_regime_centers"
METADATA_KEY_WER_TABLE = b"synthetic_wer_table"
METADATA_KEY_GEN_PARAMS = b"synthetic_generation_params"

EXPERT_DIMS: Dict[str, int] = {"hubert": 1024, "whisper": 512, "wav2vec2": 1024}
EXPERT_ORDER = ["hubert", "whisper", "wav2vec2"]
FEATURE_COL = {
    "hubert": "hubert_features",
    "whisper": "whisper_features",
    "wav2vec2": "w2v2_features",
}
WER_COL = {"hubert": "hubert_wer", "whisper": "whisper_wer", "wav2vec2": "w2v2_wer"}
DTYPE_MAP = {"float16": np.float16, "float32": np.float32}

SYNTHETIC_DEFAULT_PARQUET = "data/processed/synthetic_regime_switch/combined_features.parquet"


def _build_wer_table(
    num_regimes: int,
    num_experts: int,
    best_wer: float,
    worst_wer: float,
    rng: np.random.Generator,
) -> np.ndarray:
    table = np.full((num_regimes, num_regimes, num_experts), worst_wer, dtype=np.float32)
    for r1 in range(num_regimes):
        for r2 in range(num_regimes):
            best_k = (r1 + 2 * r2) % num_experts
            table[r1, r2, best_k] = best_wer
    table = table + rng.normal(0.0, 0.02, size=table.shape).astype(np.float32)
    return np.clip(table, 0.01, 1.0)


def _generate_batch(
    rng: np.random.Generator,
    regime_centers: np.ndarray,
    expert_projections: Dict[str, np.ndarray],
    wer_table: np.ndarray,
    batch_size: int,
    frame_length: int,
    num_regimes: int,
    noise_std: float,
    wer_noise_std: float,
    feature_dtype: np.dtype,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    b, t = batch_size, frame_length
    r1 = rng.integers(0, num_regimes, size=b)
    r2 = rng.integers(0, num_regimes - 1, size=b)
    r2 = np.where(r2 >= r1, r2 + 1, r2)
    rho = rng.integers(t // 4, 3 * t // 4, size=b)

    is_first = np.arange(t)[None, :] < rho[:, None]
    centers_first = regime_centers[r1]
    centers_second = regime_centers[r2]
    centers_per_frame = np.where(
        is_first[..., None],
        centers_first[:, None, :],
        centers_second[:, None, :],
    )
    eta = rng.normal(0.0, noise_std, size=centers_per_frame.shape).astype(np.float32)
    latent = (centers_per_frame + eta).astype(np.float32)

    features: Dict[str, np.ndarray] = {}
    for name, proj in expert_projections.items():
        feat = np.einsum("btd,de->bte", latent, proj)
        features[name] = feat.astype(feature_dtype, copy=False)

    wer_clean = wer_table[r1, r2]
    wer_noise = rng.normal(0.0, wer_noise_std, size=wer_clean.shape).astype(np.float32)
    wer_scores = np.clip(wer_clean + wer_noise, 0.0, 1.0)

    metadata = {
        "regime_first": r1.astype(np.int32),
        "regime_second": r2.astype(np.int32),
        "switch_position": rho.astype(np.int32),
    }
    return features, wer_scores, metadata


def _nested_list_column(arr: np.ndarray) -> pa.Array:
    """Build a ``list<list<floatX>>`` PyArrow array from a (N, T, D) tensor."""
    n, t, d = arr.shape
    flat_values = pa.array(
        np.ascontiguousarray(arr).reshape(-1), type=pa.from_numpy_dtype(arr.dtype)
    )
    inner_offsets = pa.array(np.arange(0, n * t * d + 1, d, dtype=np.int32))
    inner_list = pa.ListArray.from_arrays(inner_offsets, flat_values)
    outer_offsets = pa.array(np.arange(0, n * t + 1, t, dtype=np.int32))
    return pa.ListArray.from_arrays(outer_offsets, inner_list)


def _record_batch(
    features: Dict[str, np.ndarray],
    wer_scores: np.ndarray,
    metadata: Dict[str, np.ndarray],
) -> pa.RecordBatch:
    b = wer_scores.shape[0]
    columns = {"ground_truth": pa.array([""] * b, type=pa.string())}
    for name in EXPERT_ORDER:
        columns[FEATURE_COL[name]] = _nested_list_column(features[name])
    for k, name in enumerate(EXPERT_ORDER):
        columns[WER_COL[name]] = pa.array(wer_scores[:, k].astype(np.float32), type=pa.float32())
    for k, v in metadata.items():
        columns[k] = pa.array(v, type=pa.int32())
    return pa.RecordBatch.from_pydict(columns)


def build_synthetic_parquet(
    out_path: str | Path,
    *,
    num_samples: int = 10000,
    frame_length: int = 128,
    num_regimes: int = 4,
    regime_dim: int = 32,
    noise_std: float = 0.5,
    best_wer: float = 0.15,
    worst_wer: float = 0.50,
    wer_noise_std: float = 0.02,
    feature_dtype: str = "float16",
    write_batch_size: int = 500,
    seed: int = 42,
) -> Path:
    """Generate a synthetic regime-switch parquet matching the real schema."""
    if feature_dtype not in DTYPE_MAP:
        raise ValueError(f"feature_dtype must be one of {list(DTYPE_MAP)}, got {feature_dtype!r}")
    np_feature_dtype = DTYPE_MAP[feature_dtype]

    rng = np.random.default_rng(seed)
    regime_centers = rng.standard_normal((num_regimes, regime_dim)).astype(np.float32)
    expert_projections = {
        name: (rng.standard_normal((regime_dim, dim)) / np.sqrt(regime_dim)).astype(np.float32)
        for name, dim in EXPERT_DIMS.items()
    }
    wer_table = _build_wer_table(num_regimes, len(EXPERT_ORDER), best_wer, worst_wer, rng)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Generating {} synthetic clips → {} (T={}, R={}, dtype={})",
        num_samples,
        out,
        frame_length,
        num_regimes,
        feature_dtype,
    )

    schema_metadata = {
        METADATA_KEY_REGIME_CENTERS: json.dumps(regime_centers.tolist()).encode("utf-8"),
        METADATA_KEY_WER_TABLE: json.dumps(wer_table.tolist()).encode("utf-8"),
        METADATA_KEY_GEN_PARAMS: json.dumps(
            {
                "num_samples": num_samples,
                "frame_length": frame_length,
                "num_regimes": num_regimes,
                "regime_dim": regime_dim,
                "noise_std": noise_std,
                "best_wer": best_wer,
                "worst_wer": worst_wer,
                "wer_noise_std": wer_noise_std,
                "feature_dtype": feature_dtype,
                "expert_dims": EXPERT_DIMS,
                "expert_order": EXPERT_ORDER,
                "seed": seed,
            }
        ).encode("utf-8"),
    }

    writer: pq.ParquetWriter | None = None
    n_written = 0
    try:
        while n_written < num_samples:
            this_batch = min(write_batch_size, num_samples - n_written)
            features, wer_scores, metadata = _generate_batch(
                rng=rng,
                regime_centers=regime_centers,
                expert_projections=expert_projections,
                wer_table=wer_table,
                batch_size=this_batch,
                frame_length=frame_length,
                num_regimes=num_regimes,
                noise_std=noise_std,
                wer_noise_std=wer_noise_std,
                feature_dtype=np_feature_dtype,
            )
            batch = _record_batch(features, wer_scores, metadata)
            if writer is None:
                schema_with_meta = batch.schema.with_metadata(schema_metadata)
                writer = pq.ParquetWriter(out.as_posix(), schema_with_meta, compression="snappy")
            writer.write_batch(batch)
            n_written += this_batch
    finally:
        if writer is not None:
            writer.close()

    logger.info("Synthetic parquet written: {} ({} clips)", out, n_written)
    return out


class SyntheticDataModule(ASRDataModule):
    """ASRDataModule that auto-generates the synthetic parquet on first use."""

    def __init__(
        self,
        parquet_path: str = SYNTHETIC_DEFAULT_PARQUET,
        num_samples: int = 10000,
        frame_length: int = 128,
        num_regimes: int = 4,
        regime_dim: int = 32,
        noise_std: float = 0.5,
        best_wer: float = 0.15,
        worst_wer: float = 0.50,
        wer_noise_std: float = 0.02,
        feature_dtype: str = "float16",
        write_batch_size: int = 500,
        gen_seed: int = 42,
        **kwargs,
    ):
        super().__init__(parquet_path=parquet_path, **kwargs)
        self.num_samples = num_samples
        self.frame_length = frame_length
        self.num_regimes = num_regimes
        self.regime_dim = regime_dim
        self.noise_std = noise_std
        self.best_wer = best_wer
        self.worst_wer = worst_wer
        self.wer_noise_std = wer_noise_std
        self.feature_dtype = feature_dtype
        self.write_batch_size = write_batch_size
        self.gen_seed = gen_seed

    def prepare_data(self) -> None:
        if Path(self.parquet_path).exists():
            return
        build_synthetic_parquet(
            self.parquet_path,
            num_samples=self.num_samples,
            frame_length=self.frame_length,
            num_regimes=self.num_regimes,
            regime_dim=self.regime_dim,
            noise_std=self.noise_std,
            best_wer=self.best_wer,
            worst_wer=self.worst_wer,
            wer_noise_std=self.wer_noise_std,
            feature_dtype=self.feature_dtype,
            write_batch_size=self.write_batch_size,
            seed=self.gen_seed,
        )
