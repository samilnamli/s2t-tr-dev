"""Base experiment orchestration.

MLflow layout:
    experiment           = manuscript experiment family
                           (e.g. ``main_results_ami``, ``ablation_loss``)
    parent run           = one variant of that family (``v1``, ``v2`` ...)
    nested child runs    = each routing method evaluated within the variant

Reproducibility artefacts logged on every parent run:
    repro/git_state.json       — commit/branch/dirty/remote
    repro/git_dirty.patch      — staged + unstaged diff (only if dirty)
    repro/resolved_config.yaml — fully composed Hydra config

To re-run a logged experiment exactly as it was, use
:meth:`BaseExperiment.reproduce_from_mlflow`.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import traceback
from typing import Any, Dict, List, Optional

from hydra.utils import instantiate
from loguru import logger
import mlflow
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pytorch_lightning as pl
import yaml

from src.utils.git import git_diff_patch, git_state, working_tree_clean


class BaseExperiment:
    """Generic experiment runner driven entirely by Hydra YAML.

    The full experiment config (including ``data``, ``trainer`` and the
    list of ``child_runs``) is provided by ``BaseExperiment.run(cfg)``.
    Subclassing is unnecessary unless an experiment needs custom
    orchestration; the YAML is the single source of truth.
    """

    def __init__(
        self,
        parent_run_name: str = "v1",
        seed: int = 42,
        **_kwargs,
    ):
        self.parent_run_name = parent_run_name
        self.seed = seed
        self.results: Dict[str, Dict[str, Any]] = {}
        self.parent_run_id: Optional[str] = None
        pl.seed_everything(seed, workers=True)

    def run(self, cfg: DictConfig) -> Dict[str, Dict[str, Any]]:
        with mlflow.start_run(run_name=self.parent_run_name) as parent_run:
            self.parent_run_id = parent_run.info.run_id
            self._log_reproducibility(cfg)

            datamodule = self._setup_datamodule(cfg.data)
            child_runs = list(cfg.child_runs)
            mlflow.log_params({"seed": self.seed, "n_children": len(child_runs)})

            for child_cfg in child_runs:
                self._run_single_child(child_cfg, datamodule, cfg.trainer)

            self._log_comparison_table()

        return self.results

    def _log_reproducibility(self, cfg: DictConfig) -> None:
        state = git_state()
        params = {f"git_{k}": str(v) for k, v in state.items() if v is not None}
        if params:
            mlflow.log_params(params)
        mlflow.log_dict(state, "repro/git_state.json")

        patch = git_diff_patch()
        if patch:
            mlflow.log_text(patch, "repro/git_dirty.patch")

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(cfg_dict, "repro/resolved_config.yaml")

    def _setup_datamodule(self, data_cfg: DictConfig) -> pl.LightningDataModule:
        datamodule = instantiate(data_cfg)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule

    def _run_single_child(
        self,
        child_cfg: Any,
        datamodule: pl.LightningDataModule,
        trainer_cfg: DictConfig,
    ) -> None:
        child_name = OmegaConf.select(child_cfg, "name", default="unnamed")
        run_name = f"{self.parent_run_name}__{child_name}"
        logger.info("[{}] running child: {}", self.parent_run_name, child_name)

        with mlflow.start_run(run_name=run_name, nested=True) as child_run:
            try:
                selector = instantiate(child_cfg)
                selector.fit(
                    datamodule,
                    trainer_cfg=trainer_cfg,
                    mlflow_run_id=child_run.info.run_id,
                )
                metrics = selector.evaluate(datamodule)
                loggable = {
                    f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))
                }
                mlflow.log_metrics(loggable)
                self.results[child_name] = metrics
                mlflow.set_tag("status", "success")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e)[:500])
                self.results[child_name] = {"error": str(e)}

    def _log_comparison_table(self) -> None:
        rows = []
        for name, metrics in self.results.items():
            if "error" in metrics:
                continue
            rows.append(
                {
                    "name": name,
                    "wer_mean": metrics.get("wer_mean"),
                    "wer_sem": metrics.get("wer_sem"),
                    "selection_accuracy": metrics.get("selection_accuracy"),
                    "n": metrics.get("n"),
                }
            )
        if not rows:
            return
        df = pd.DataFrame(rows).sort_values("wer_mean")
        mlflow.log_table(data=df, artifact_file="results/test_wer_comparison.json")
        logger.info("Test WER comparison:\n{}", df.to_string(index=False))

    @classmethod
    def reproduce_from_mlflow(
        cls,
        run_id: str,
        *,
        force: bool = False,
        run: bool = True,
        cwd: Optional[Path] = None,
    ) -> Path:
        """Restore the repo state of a logged run and (optionally) re-run it.

        Steps:
            1. Download the parent run's ``repro/`` artefacts.
            2. Refuse to act if the working tree is dirty (override with
               ``force=True``).
            3. ``git checkout`` the logged commit; ``git apply`` the
               patch if any.
            4. Invoke ``python run.py --config-dir <tmp> --config-name
               resolved_config`` so the new run uses the exact same
               composed config.

        Returns:
            Path to the downloaded ``resolved_config.yaml``.
        """
        client = mlflow.tracking.MlflowClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            artifact_dir = Path(client.download_artifacts(run_id, "repro", str(tmp)))

            state_path = artifact_dir / "git_state.json"
            patch_path = artifact_dir / "git_dirty.patch"
            cfg_path = artifact_dir / "resolved_config.yaml"

            with state_path.open() as f:
                state = yaml.safe_load(f)
            commit = state.get("commit")
            if not commit:
                raise RuntimeError(f"run {run_id} has no logged git commit")

            if not force and not working_tree_clean():
                raise RuntimeError(
                    "Working tree is dirty; refusing to checkout. "
                    "Pass force=True to override (your changes may be lost)."
                )

            logger.info("checkout {} (from mlflow run {})", commit, run_id)
            subprocess.run(["git", "checkout", commit], check=True)
            if patch_path.exists():
                logger.info("applying logged dirty patch")
                subprocess.run(
                    ["git", "apply", "--whitespace=fix", str(patch_path)],
                    check=True,
                )

            persistent_cfg = (cwd or Path.cwd()) / ".repro_resolved_config.yaml"
            persistent_cfg.write_text(cfg_path.read_text())

            if run:
                logger.info("re-running with config: {}", persistent_cfg)
                env = {**os.environ}
                subprocess.run(
                    [
                        "python",
                        "run.py",
                        "--config-dir",
                        str(persistent_cfg.parent),
                        "--config-name",
                        persistent_cfg.stem,
                    ],
                    check=True,
                    env=env,
                    cwd=str(cwd) if cwd else None,
                )

            return persistent_cfg

    def child_run_configs(self) -> List[Any]:
        """Compatibility shim for the old ABC contract."""
        return []
