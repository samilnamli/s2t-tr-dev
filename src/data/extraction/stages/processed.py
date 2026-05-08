"""Processed stage — combine K interim parquets into the trainer-ready parquet.

Reads each extractor's interim parquet (one per base model), joins on
``clip_id``, and writes a single parquet with the schema expected by
:class:`src.data.dataset.ASRFeatureDataset`. The combined parquet
embeds an ``asr_models`` metadata key so the dataset auto-discovers
the K-model spec; it also carries a full :class:`Manifest` under the
``extraction_manifest`` key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dataset import ASR_MODELS_METADATA_KEY
from src.data.extraction.manifest import (
    DatasetSpec,
    ExtractorEntry,
    Manifest,
    MANIFEST_METADATA_KEY,
)
from src.data.extraction.stages.interim import feature_columns

logger = logging.getLogger(__name__)


def build_processed(
    interim_paths: Mapping[str, Path],
    output_path: str | Path,
    *,
    dataset_spec: DatasetSpec,
    extractor_entries: Sequence[ExtractorEntry],
    config_hash: Optional[str] = None,
    row_group_size: int = 256,
) -> Path:
    """Join per-extractor interim parquets into the combined processed parquet.

    ``interim_paths`` is ``{extractor_name: parquet_path}``, one entry
    per base model. ``extractor_entries`` is the matching ordered list
    of manifest entries — the order determines column ordering in the
    output and the per-model index used by the WER matrix.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not extractor_entries:
        raise ValueError("extractor_entries must be non-empty.")
    if set(e.name for e in extractor_entries) != set(interim_paths.keys()):
        raise ValueError(
            f"Mismatch between interim_paths keys {sorted(interim_paths)} "
            f"and extractor_entries names {[e.name for e in extractor_entries]}"
        )

    # ---- read all interim tables (one per extractor) -------------------
    tables: dict[str, pa.Table] = {}
    for entry in extractor_entries:
        path = interim_paths[entry.name]
        tables[entry.name] = pq.read_table(path)

    # ---- align on clip_id ---------------------------------------------
    canonical = tables[extractor_entries[0].name]
    canonical_ids = canonical.column("clip_id").to_pylist()
    canonical_set = set(canonical_ids)

    aligned: dict[str, pa.Table] = {extractor_entries[0].name: canonical}
    for entry in extractor_entries[1:]:
        t = tables[entry.name]
        ids = t.column("clip_id").to_pylist()
        if ids == canonical_ids:
            aligned[entry.name] = t
            continue

        if not canonical_set.issubset(set(ids)):
            missing = canonical_set - set(ids)
            raise ValueError(
                f"Extractor '{entry.name}' is missing clip_ids present in "
                f"'{extractor_entries[0].name}' ({len(missing)} missing — "
                f"e.g. {sorted(missing)[:5]}). Re-run interim for this extractor."
            )
        # Reorder t to match canonical_ids.
        idx_map = {cid: i for i, cid in enumerate(ids)}
        order = [idx_map[cid] for cid in canonical_ids]
        aligned[entry.name] = t.take(pa.array(order, type=pa.int64()))

    # ---- assemble combined table --------------------------------------
    out_columns: list[tuple[str, pa.Array]] = [
        ("clip_id", canonical.column("clip_id")),
        ("ground_truth", canonical.column("ground_truth")),
    ]
    asr_models_meta: list[dict] = []
    for entry in extractor_entries:
        t = aligned[entry.name]
        cols = feature_columns(entry.name)
        out_columns.extend(
            [
                (cols["features"], t.column(cols["features"])),
                (cols["wer"], t.column(cols["wer"])),
                (cols["transcription"], t.column(cols["transcription"])),
            ]
        )
        asr_models_meta.append(
            {
                "name": entry.name,
                "feature_col": cols["features"],
                "wer_col": cols["wer"],
                "transcription_col": cols["transcription"],
                "embedding_dim": entry.embedding_dim,
            }
        )

    schema_no_meta = pa.schema([(name, arr.type) for name, arr in out_columns])
    arrays = [arr for _, arr in out_columns]

    # ---- attach metadata: asr_models spec + manifest -------------------
    manifest = Manifest.build(
        stage="processed",
        dataset=DatasetSpec(
            hf_id=dataset_spec.hf_id,
            revision=dataset_spec.revision,
            split=dataset_spec.split,
            config=dataset_spec.config,
            num_examples=canonical.num_rows,
            chunking=dataset_spec.chunking,
        ),
        extractors=list(extractor_entries),
        config_hash=config_hash,
    )
    metadata = {
        ASR_MODELS_METADATA_KEY: json.dumps(asr_models_meta).encode("utf-8"),
        MANIFEST_METADATA_KEY: manifest.to_metadata_bytes(),
    }
    schema = schema_no_meta.with_metadata(metadata)
    table = pa.Table.from_arrays(arrays, schema=schema)

    pq.write_table(table, output_path, compression="zstd", row_group_size=row_group_size)
    manifest.write_yaml(output_path.parent / "manifest.yaml")

    logger.info(
        "Processed parquet written: %s (%d clips, K=%d models)",
        output_path,
        canonical.num_rows,
        len(extractor_entries),
    )
    return output_path
