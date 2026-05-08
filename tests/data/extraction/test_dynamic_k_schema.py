"""New ``asr_models``-tagged parquets are read with the dynamic K-model spec.

Verifies the round-trip:
    processed-stage writer → embedded ``asr_models`` metadata
    ASRFeatureDataset reader → discovers model_names from metadata
    ASRDataModule          → exposes correct model_dims / dataloaders
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.base import ASRDataModule
from src.data.dataset import ASR_MODELS_METADATA_KEY, ASRFeatureDataset


def _write_dynamic_k_parquet(
    path: Path,
    *,
    model_names: list[str],
    n_rows: int = 8,
) -> None:
    """Write a tiny dynamic-K parquet with ``asr_models`` metadata embedded."""
    rng = np.random.default_rng(0)
    arrays: dict = {"ground_truth": ["hello"] * n_rows}
    spec = []
    for i, name in enumerate(model_names):
        feat_col = f"{name}_features"
        wer_col = f"{name}_wer"
        tr_col = f"{name}_transcription"
        D = 8 + 2 * i
        rows = []
        for _ in range(n_rows):
            T = int(rng.integers(4, 9))
            rows.append(rng.standard_normal((T, D)).astype(np.float32).tolist())
        arrays[feat_col] = rows
        arrays[wer_col] = np.full(n_rows, 0.2 - 0.05 * i, dtype=np.float32)
        arrays[tr_col] = ["hello world"] * n_rows
        spec.append(
            {
                "name": name,
                "feature_col": feat_col,
                "wer_col": wer_col,
                "transcription_col": tr_col,
                "embedding_dim": D,
            }
        )

    table = pa.Table.from_pydict(arrays)
    metadata = {ASR_MODELS_METADATA_KEY: json.dumps(spec).encode("utf-8")}
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def test_two_model_dynamic_k(tmp_path: Path) -> None:
    p = tmp_path / "dyn2.parquet"
    _write_dynamic_k_parquet(p, model_names=["alpha", "beta"], n_rows=10)
    ds = ASRFeatureDataset(str(p), eager_load=False, max_seq_len=100)
    assert ds.model_names == ["alpha", "beta"]
    s = ds[0]
    assert set(s["hidden_states"].keys()) == {"alpha", "beta"}
    assert s["wer_matrix"].shape == (2,)


def test_four_model_dynamic_k_eager(tmp_path: Path) -> None:
    names = ["m1", "m2", "m3", "m4"]
    p = tmp_path / "dyn4.parquet"
    _write_dynamic_k_parquet(p, model_names=names, n_rows=12)
    ds = ASRFeatureDataset(str(p), eager_load=True, max_seq_len=100)
    assert ds.model_names == names
    assert set(ds._model_dims.keys()) == set(names)


def test_datamodule_dynamic_k(tmp_path: Path) -> None:
    names = ["wav2vec2_base", "data2vec_audio_base", "s2t_small"]
    p = tmp_path / "dyn3.parquet"
    _write_dynamic_k_parquet(p, model_names=names, n_rows=20)
    dm = ASRDataModule(parquet_path=str(p), batch_size=4, num_workers=0, max_seq_len=100, seed=1)
    dm.prepare_data()
    dm.setup()
    assert list(dm.model_dims.keys()) == names
    batch = next(iter(dm.train_dataloader()))
    assert set(batch["hidden_states"].keys()) == set(names)
    assert batch["wer_matrix"].shape[1] == len(names)
