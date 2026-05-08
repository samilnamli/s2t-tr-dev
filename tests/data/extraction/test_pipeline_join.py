"""Stage-level integration: build interim parquets directly with PyArrow,
join via :func:`build_processed`, and verify the result round-trips through
:class:`ASRFeatureDataset`.

Skips the live HF model loads (those need GPUs and network); the goal
here is to validate the join, the manifest plumbing, and the schema
contract — not extractor correctness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dataset import ASRFeatureDataset
from src.data.extraction.manifest import (
    DatasetSpec,
    ExtractorEntry,
    Manifest,
)
from src.data.extraction.stages.interim import feature_columns
from src.data.extraction.stages.processed import build_processed


def _write_fake_interim(path: Path, *, name: str, dim: int, clip_ids: list[str], hyp: str) -> None:
    cols = feature_columns(name)
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    feats = []
    for _ in clip_ids:
        T = int(rng.integers(3, 8))
        feats.append(rng.standard_normal((T, dim)).astype(np.float16).tolist())
    table = pa.table(
        {
            "clip_id": clip_ids,
            "ground_truth": ["hello world"] * len(clip_ids),
            cols["features"]: pa.array(feats, type=pa.list_(pa.list_(pa.float16()))),
            cols["transcription"]: [hyp] * len(clip_ids),
            cols["wer"]: np.full(len(clip_ids), 0.5, dtype=np.float32),
        }
    )
    # Attach a manifest so the reader has a (mock) provenance trail.
    manifest = Manifest.build(
        stage="interim",
        dataset=DatasetSpec(hf_id="fake/ds", revision="r1", split="train", num_examples=len(clip_ids)),
        extractors=[ExtractorEntry(name=name, hf_model_id=f"fake/{name}", embedding_dim=dim)],
        config_hash="sha256:fake",
    )
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"extraction_manifest": manifest.to_metadata_bytes()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    manifest.write_yaml(path.parent / "manifest.yaml")


def test_processed_join_round_trip(tmp_path: Path) -> None:
    # Create three fake interim parquets sharing clip_ids.
    clip_ids = [f"c{i:03d}" for i in range(20)]
    interim_dir = tmp_path / "interim"
    paths = {}
    for name, dim in [("wav2vec2_base", 8), ("data2vec_audio_base", 6), ("s2t_small", 10)]:
        p = interim_dir / name / "features.parquet"
        _write_fake_interim(p, name=name, dim=dim, clip_ids=clip_ids, hyp="hello world")
        paths[name] = p

    entries = [
        ExtractorEntry(name="wav2vec2_base", hf_model_id="fake/wv", embedding_dim=8),
        ExtractorEntry(name="data2vec_audio_base", hf_model_id="fake/d2v", embedding_dim=6),
        ExtractorEntry(name="s2t_small", hf_model_id="fake/s2t", embedding_dim=10),
    ]
    out = tmp_path / "processed" / "combined_features.parquet"
    build_processed(
        interim_paths=paths,
        output_path=out,
        dataset_spec=DatasetSpec(hf_id="fake/ds", revision="r1", split="train", num_examples=20),
        extractor_entries=entries,
        config_hash="sha256:fake",
    )

    # Read with ASRFeatureDataset → must auto-discover the dynamic K schema.
    ds = ASRFeatureDataset(str(out), eager_load=False, max_seq_len=100)
    assert ds.model_names == ["wav2vec2_base", "data2vec_audio_base", "s2t_small"]
    sample = ds[0]
    assert sample["hidden_states"]["wav2vec2_base"].shape[-1] == 8
    assert sample["hidden_states"]["data2vec_audio_base"].shape[-1] == 6
    assert sample["hidden_states"]["s2t_small"].shape[-1] == 10
    assert sample["wer_matrix"].shape == (3,)


def test_processed_rejects_missing_clip(tmp_path: Path) -> None:
    """If one extractor's interim is missing clips, build_processed errors loudly."""
    interim_dir = tmp_path / "interim"
    full_ids = [f"c{i:03d}" for i in range(10)]
    short_ids = full_ids[:5]
    paths = {
        "a": interim_dir / "a" / "features.parquet",
        "b": interim_dir / "b" / "features.parquet",
    }
    _write_fake_interim(paths["a"], name="a", dim=4, clip_ids=full_ids, hyp="x")
    _write_fake_interim(paths["b"], name="b", dim=4, clip_ids=short_ids, hyp="x")
    entries = [
        ExtractorEntry(name="a", hf_model_id="fake/a", embedding_dim=4),
        ExtractorEntry(name="b", hf_model_id="fake/b", embedding_dim=4),
    ]
    out = tmp_path / "processed" / "combined.parquet"
    import pytest

    with pytest.raises(ValueError, match="missing clip_ids"):
        build_processed(
            interim_paths=paths,
            output_path=out,
            dataset_spec=DatasetSpec(hf_id="fake/ds", num_examples=10),
            extractor_entries=entries,
        )
