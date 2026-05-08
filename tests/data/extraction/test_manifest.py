"""Manifest round-trip + matching/mismatch detection."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from src.data.extraction.manifest import (
    DEFAULT_MATCH_IGNORE,
    DatasetSpec,
    ExtractorEntry,
    MANIFEST_METADATA_KEY,
    Manifest,
    hash_config,
)


def _example_manifest(stage: str = "interim") -> Manifest:
    return Manifest.build(
        stage=stage,
        dataset=DatasetSpec(
            hf_id="revdotcom/earnings22",
            revision="abc123",
            split="train",
            num_examples=42,
            chunking={"method": "fixed_window", "seconds": 10.0},
        ),
        extractors=[
            ExtractorEntry(
                name="wav2vec2_base",
                hf_model_id="facebook/wav2vec2-base-960h",
                hf_revision="def456",
                embedding_dim=768,
            )
        ],
        config_hash="sha256:deadbeef",
    )


def test_yaml_round_trip(tmp_path: Path) -> None:
    m = _example_manifest()
    p = m.write_yaml(tmp_path / "manifest.yaml")
    loaded = Manifest.read_yaml(p)
    # ``matches`` ignores environmental drift fields; we want full equality
    # here, so include those fields too — except for ``versions`` whose
    # mapping types may differ trivially after YAML round-trip.
    assert m.matches(loaded, ignore=())


def test_arrow_metadata_embedding() -> None:
    m = _example_manifest()
    schema = pa.schema([("clip_id", pa.string())])
    schema = m.attach_to_schema(schema)
    assert MANIFEST_METADATA_KEY in (schema.metadata or {})
    recovered = Manifest.from_parquet_metadata(schema)
    assert recovered is not None
    assert m.matches(recovered)


def test_matches_ignores_environmental_drift() -> None:
    a = _example_manifest()
    b = _example_manifest()
    # ``created_at``, ``created_by``, ``git``, ``versions`` will differ
    # in real life — by default ``matches`` ignores them.
    object.__setattr__(b, "created_at", "1999-01-01T00:00:00+00:00")
    object.__setattr__(b, "created_by", "someone-else")
    assert a.matches(b)
    assert not a.matches(b, ignore=())


def test_diff_surfaces_disagreements() -> None:
    a = _example_manifest()
    b = _example_manifest()
    b_dict = b.to_dict()
    b_dict["dataset"]["revision"] = "different"
    b2 = Manifest.from_dict(b_dict)
    diff = a.diff(b2)
    assert "dataset" in diff
    self_val, other_val = diff["dataset"]
    assert self_val["revision"] == "abc123"
    assert other_val["revision"] == "different"


def test_hash_config_is_stable_under_key_order() -> None:
    a = {"data": {"hf_id": "x", "split": "train"}, "seed": 42}
    b = {"seed": 42, "data": {"split": "train", "hf_id": "x"}}
    assert hash_config(a) == hash_config(b)


def test_default_match_ignore_includes_drift_fields() -> None:
    # Sanity check on the constants the rest of the codebase depends on.
    assert "created_at" in DEFAULT_MATCH_IGNORE
    assert "git" in DEFAULT_MATCH_IGNORE
