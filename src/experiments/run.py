"""Generic Hydra-driven experiment runner.

Replaces the old ``main_results.py`` (which only handled the AMI/VoxPopuli
main-results pipeline) with a dataset- and task-agnostic runner. The
behavior of a single invocation is fully determined by the chosen
experiment YAML in ``configs/experiments/<name>.yaml``:

    1. Optionally compute training-free baselines via
       :mod:`src.training.eval_baselines` (skipped when
       ``skip_training_free=true``, read from the experiment YAML
       first then the top-level config).
    2. Optionally compute ROVER and weighted ROVER via
       :mod:`src.training.rover` (skipped when ``skip_rover=true``,
       e.g. for the synthetic experiment which has no transcripts).
    3. For every entry in ``cfg.experiments.methods``, train the
       declared variant via :mod:`src.training.train`. Multi-seed
       methods (``seed: [42, 2, 123]``) are auto-expanded into one
       independent run per seed.
    4. Aggregate per-method ``test_results.json`` files into a single
       ``main_results.json`` under ``--output-dir``.
    5. Render the manuscript-grade table via :mod:`src.reporting.tables`
       (best-effort: stub falls through silently if reporting is not
       wired for that experiment yet).

Usage:
    uv run python -m src.experiments.run experiments=ablation_loss
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.experiments._common import (
    expand_seeds,
    hydra_overrides,
    python_module,
    run,
)
from src.utils.logging import setup_unified_logging


def _config_name_from_argv() -> str:
    """Return the value of the ``experiments=...`` CLI override, or ``"default"``."""
    for arg in sys.argv:
        if arg.startswith("experiments="):
            return arg.split("=", 1)[1]
    return "default"


def _log_results_table(aggregated: Dict[str, Any]) -> None:
    """Print a compact stdout table of every method's test WER.

    This is a best-effort summary aimed at a human reading the terminal;
    the manuscript-grade tables are produced by
    :mod:`src.reporting.tables` from the same JSON.
    """
    rows: list[tuple[str, float, int]] = []

    if aggregated.get("training_free"):
        for k, v in aggregated["training_free"].items():
            if isinstance(v, dict):
                continue
            if isinstance(v, float) and "wer" in k.lower():
                rows.append((f"baseline:{k}", float(v), 1))

    if aggregated.get("rover"):
        for k, v in aggregated["rover"].items():
            if isinstance(v, dict):
                continue
            if isinstance(v, float) and "wer" in k.lower():
                rows.append((f"rover:{k}", float(v), 1))

    for name, data in (aggregated.get("methods") or {}).items():
        wer = data.get("test", {}).get("selected_wer")
        if wer is None:
            wer = data.get("test", {}).get("wer")
        if wer is not None:
            rows.append((f"method:{name}", float(wer), 1))

    if not rows:
        logger.info("No results to print.")
        return

    rows.sort(key=lambda r: r[1])
    width = max(len(r[0]) for r in rows)
    logger.info("=" * (width + 18))
    logger.info("FINAL TEST WER (terminal summary)")
    logger.info("=" * (width + 18))
    for name, wer, _ in rows:
        logger.info("  {:<{w}}  {:.4f}", name, wer, w=width)
    logger.info("=" * (width + 18))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Orchestrate a full experiment pipeline declared in ``cfg.experiments``."""
    setup_unified_logging(level="INFO")

    pipeline = cfg.experiments
    parquet_path: str = pipeline.parquet_path
    shared = OmegaConf.to_container(pipeline.get("shared", {}), resolve=True)
    raw_methods = OmegaConf.to_container(pipeline.get("methods", []), resolve=True)
    metadata = OmegaConf.to_container(pipeline.get("experiment_metadata", {}), resolve=True)

    # Skip flags MUST be read from the experiment block first, falling back
    # to the top-level default in configs/config.yaml. Reading only
    # cfg.skip_* loses any override declared inside the experiment YAML
    # (e.g. synthetic_v2 sets skip_rover=true because it has no transcripts).
    skip_training_free = bool(
        pipeline.get("skip_training_free", cfg.get("skip_training_free", False))
    )
    skip_rover = bool(pipeline.get("skip_rover", cfg.get("skip_rover", False)))

    config_name = _config_name_from_argv()
    methods = expand_seeds(raw_methods)

    out_root = Path(cfg.get("output_dir", "reports") or "reports") / "main_results" / config_name
    out_root.mkdir(parents=True, exist_ok=True)

    if metadata:
        logger.info("=== experiment_metadata ===")
        for k, v in metadata.items():
            logger.info("  {}: {}", k, v)
    else:
        logger.warning(
            "Experiment '{}' has no `experiment_metadata` block; please backfill "
            "per configs/README.md.",
            config_name,
        )

    split_args = {k: shared[k] for k in ("train_ratio", "val_ratio", "seed") if k in shared}

    baselines_json = out_root / "baselines.json"
    if skip_training_free:
        logger.info("skip_training_free=true — skipping eval_baselines step.")
    else:
        run(
            python_module(
                "src.training.eval_baselines",
                "--parquet-path",
                parquet_path,
                "--split",
                "test",
                "--save-json",
                str(baselines_json),
                *(f for k, v in split_args.items() for f in (f"--{k.replace('_', '-')}", str(v))),
            ),
            description="training-free baselines",
        )

    rover_json = out_root / "rover.json"
    if skip_rover:
        logger.info("skip_rover=true — skipping ROVER step (synthetic data has no transcripts).")
    else:
        run(
            python_module(
                "src.training.rover",
                "--parquet-path",
                parquet_path,
                "--split",
                "test",
                "--save-json",
                str(rover_json),
                *(f for k, v in split_args.items() for f in (f"--{k.replace('_', '-')}", str(v))),
            ),
            description="ROVER baselines",
        )

    aggregated: Dict[str, Any] = {
        "config_name": config_name,
        "experiment_metadata": metadata,
        "parquet_path": parquet_path,
        "shared": shared,
        "training_free": (
            json.loads(baselines_json.read_text()) if baselines_json.exists() else None
        ),
        "rover": (json.loads(rover_json.read_text()) if rover_json.exists() else None),
        "methods": {},
    }

    log_dir = Path(cfg.get("log_dir", "logs"))
    for method in methods:
        name = method["name"]
        logger.info("=" * 60)
        logger.info("Method: {}", name)
        logger.info("=" * 60)
        run_dir = log_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        merged = {**shared, **{k: v for k, v in method.items() if k != "name"}}

        train_overrides = {
            "parquet_path": parquet_path,
            "experiment_name": name,
            "log_dir": str(log_dir),
            "wandb_group": config_name,
            **merged,
        }
        run(
            python_module("src.training.train", *hydra_overrides(train_overrides)),
            description=f"train method={name}",
        )

        test_json = log_dir / name / "test_results.json"
        if not test_json.exists():
            logger.warning("No test_results.json for {}; skipping aggregation.", name)
            continue
        aggregated["methods"][name] = {
            "config_overrides": {k: v for k, v in method.items() if k != "name"},
            "test": json.loads(test_json.read_text()),
        }
        with open(out_root / "main_results.json", "w") as f:
            json.dump(aggregated, f, indent=2)

    _log_results_table(aggregated)

    try:
        from src.reporting.tables import render_main_results_table

        render_main_results_table(
            json_path=out_root / "main_results.json",
            output_dir=out_root,
        )
    except Exception as exc:  # pragma: no cover - best-effort manuscript rendering
        logger.warning("Skipping reporting.tables render: {}", exc)

    logger.info("Pipeline complete. Aggregated results: {}", out_root / "main_results.json")


if __name__ == "__main__":
    main()
