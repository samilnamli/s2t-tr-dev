"""Legacy 3-slot schema must still load when ``asr_models`` metadata is absent.

Exercises the fallback path in :func:`src.data.dataset._resolve_model_spec`
and the dynamic-K consumers in :class:`ASRDataModule`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.base import ASRDataModule
from src.data.dataset import (
    ASRFeatureDataset,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    TRANSCRIPTION_COLUMNS,
    WER_COLUMNS,
    _resolve_model_spec,
    make_collate_fn,
)


def _write_legacy_parquet(path: Path, n_rows: int = 8) -> None:
    """Write a tiny legacy-schema parquet (no ``asr_models`` metadata)."""
    rng = np.random.default_rng(0)
    feature_arrays = {}
    dims = {"hubert": 16, "whisper": 12, "wav2vec2": 24}
    for name in MODEL_NAMES:
        col = FEATURE_COLUMNS[name]
        rows = []
        for _ in range(n_rows):
            T = int(rng.integers(5, 12))
            rows.append(rng.standard_normal((T, dims[name])).astype(np.float32).tolist())
        feature_arrays[col] = rows

    arrays = {"ground_truth": ["hello"] * n_rows}
    arrays.update(feature_arrays)
    arrays.update({WER_COLUMNS[n]: np.full(n_rows, 0.1, dtype=np.float32) for n in MODEL_NAMES})
    arrays.update({TRANSCRIPTION_COLUMNS[n]: ["hello world"] * n_rows for n in MODEL_NAMES})

    table = pa.Table.from_pydict(arrays)
    # NOTE: deliberately no metadata attached — this is the legacy path.
    pq.write_table(table, path)


def test_resolve_model_spec_falls_back_to_legacy(tmp_path: Path) -> None:
    p = tmp_path / "legacy.parquet"
    _write_legacy_parquet(p)
    spec = _resolve_model_spec(str(p))
    names = [m["name"] for m in spec]
    assert names == MODEL_NAMES
    assert spec[0]["feature_col"] == FEATURE_COLUMNS["hubert"]


def test_dataset_loads_legacy_parquet(tmp_path: Path) -> None:
    p = tmp_path / "legacy.parquet"
    _write_legacy_parquet(p, n_rows=6)
    ds = ASRFeatureDataset(str(p), eager_load=False, max_seq_len=100)
    assert ds.model_names == MODEL_NAMES
    sample = ds[0]
    assert set(sample["hidden_states"].keys()) == set(MODEL_NAMES)


def test_dataset_loads_legacy_parquet_eager(tmp_path: Path) -> None:
    p = tmp_path / "legacy.parquet"
    _write_legacy_parquet(p, n_rows=6)
    ds = ASRFeatureDataset(str(p), eager_load=True, max_seq_len=100)
    assert ds.model_names == MODEL_NAMES
    sample = ds[0]
    assert set(sample["hidden_states"].keys()) == set(MODEL_NAMES)


def test_collate_fn_dynamic_k_legacy(tmp_path: Path) -> None:
    p = tmp_path / "legacy.parquet"
    _write_legacy_parquet(p, n_rows=4)
    ds = ASRFeatureDataset(str(p), eager_load=False, max_seq_len=100)
    batch = [ds[i] for i in range(4)]
    collate = make_collate_fn(ds.model_names)
    out = collate(batch)
    for name in MODEL_NAMES:
        assert name in out["hidden_states"]
        assert out["hidden_states"][name].shape[0] == 4
    assert out["wer_matrix"].shape == (4, 3)
    assert out["targets"].shape == (4,)


def test_datamodule_smoke(tmp_path: Path) -> None:
    p = tmp_path / "legacy.parquet"
    _write_legacy_parquet(p, n_rows=20)
    dm = ASRDataModule(parquet_path=str(p), batch_size=4, num_workers=0, max_seq_len=100, seed=7)
    dm.prepare_data()
    dm.setup()
    assert set(dm.model_dims.keys()) == set(MODEL_NAMES)
    batch = next(iter(dm.train_dataloader()))
    assert batch["wer_matrix"].shape[1] == 3
