"""CLI: ``python -m src.experiments.reproduce <mlflow_run_id> [--force]``.

Restores the git state of a logged MLflow parent run and re-executes
it via the project's run.py entry point.
"""

from __future__ import annotations

import argparse

from src.experiments.base import BaseExperiment
from src.utils.logging import setup_unified_logging
from src.utils.mlflow_setup import setup_mlflow


def main() -> None:
    setup_unified_logging(level="INFO")

    parser = argparse.ArgumentParser(description="Reproduce a logged MLflow run.")
    parser.add_argument("run_id", help="MLflow parent run id to reproduce.")
    parser.add_argument(
        "--force", action="store_true", help="Allow checkout even with dirty working tree."
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Restore git state and config, but don't invoke run.py.",
    )
    parser.add_argument(
        "--tracking-uri", default=None, help="MLflow tracking URI (defaults to env / ./mlruns)."
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf

    setup_mlflow(
        OmegaConf.create(
            {
                "tracking_uri": args.tracking_uri,
                "experiment_name": "reproduce",  # not actually written, just required
            }
        )
    )

    BaseExperiment.reproduce_from_mlflow(
        args.run_id,
        force=args.force,
        run=not args.no_run,
    )


if __name__ == "__main__":
    main()
