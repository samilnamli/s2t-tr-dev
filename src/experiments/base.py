"""Base experiment orchestration.

MLflow layout:
    experiment           = manuscript experiment family
                           (e.g. ``main_results_ami``, ``ablation_loss``)
    parent run           = one variant of that family (``v1``, ``v2`` ...)
      method run (child) = each routing method evaluated within the variant;
                           holds across-seed aggregate metrics
        seed run (gchild)= one fit/eval at a specific seed

Reproducibility artefacts logged on every parent run:
    repro/git_state.json       — commit/branch/dirty/remote
    repro/git_dirty.patch      — staged + unstaged diff (only if dirty)
    repro/resolved_config.yaml — fully composed Hydra config

To re-run a logged experiment exactly as it was, use
:meth:`BaseExperiment.reproduce_from_mlflow`.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import subprocess
import tempfile
import traceback
from typing import Any, Dict, Iterable, List, Optional

from hydra.utils import instantiate
from loguru import logger
import mlflow
from omegaconf import DictConfig, OmegaConf, open_dict
import pandas as pd
import pytorch_lightning as pl
import yaml

from src.utils.git import git_diff_patch, git_state, working_tree_clean

# Across-seed SEM is exposed under this suffix in the aggregated metrics dict.
SEED_SEM_SUFFIX = "__seed_sem"


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
        seeds: Optional[Iterable[int]] = None,
        **_kwargs,
    ):
        self.parent_run_name = parent_run_name
        self.seed = seed  # data-split seed; constant across method runs
        self.seeds: List[int] = [int(s) for s in seeds] if seeds is not None else [int(seed)]
        self.results: Dict[str, Dict[str, Any]] = {}
        self.parent_run_id: Optional[str] = None
        pl.seed_everything(seed, workers=True)

    def run(self, cfg: DictConfig) -> Dict[str, Dict[str, Any]]:
        with mlflow.start_run(run_name=self.parent_run_name) as parent_run:
            self.parent_run_id = parent_run.info.run_id
            self._log_reproducibility(cfg)

            datamodule = self._setup_datamodule(cfg.data)
            child_runs = list(cfg.child_runs)
            mlflow.log_params(
                {
                    "seed": self.seed,
                    "seeds": str(self.seeds),
                    "n_seeds": len(self.seeds),
                    "n_children": len(child_runs),
                }
            )

            for child_cfg in child_runs:
                self._run_method(child_cfg, datamodule, cfg.trainer)

            self._log_comparison_table()

        return self.results

    def _log_reproducibility(self, cfg: DictConfig) -> None:
        state = git_state()
        params = {f"git_{k}": str(v) for k, v in state.items() if v is not None}
        if params:
            mlflow.log_params(params)
            mlflow.set_tag("mlflow.source.git.commit", params["git_commit"])
            mlflow.set_tag("mlflow.source.git.branch", params["git_branch"])
            mlflow.set_tag("mlflow.source.git.repo_url", params["git_remote"])

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

    def _run_method(
        self,
        child_cfg: Any,
        datamodule: pl.LightningDataModule,
        trainer_cfg: DictConfig,
    ) -> None:
        """Run one routing method across all configured seeds.

        Layout produced (always 3 levels deep, even with one seed):

            method_run = ``{parent}__{method_name}``           ← aggregate
                seed_run = ``{method_run}__seed{N}``           ← actual fit
        """
        method_name = OmegaConf.select(child_cfg, "name", default="unnamed")
        method_run_name = f"{self.parent_run_name}__{method_name}"
        logger.info(
            "[{}] running method: {} ({} seed{})",
            self.parent_run_name,
            method_name,
            len(self.seeds),
            "s" if len(self.seeds) != 1 else "",
        )

        with mlflow.start_run(run_name=method_run_name, nested=True):
            mlflow.log_params(
                {
                    "method_name": method_name,
                    "seeds": str(self.seeds),
                    "n_seeds": len(self.seeds),
                }
            )
            per_seed: List[Dict[str, Any]] = []
            for seed in self.seeds:
                per_seed.append(
                    self._run_single_seed(
                        child_cfg=child_cfg,
                        datamodule=datamodule,
                        trainer_cfg=trainer_cfg,
                        seed=seed,
                        method_run_name=method_run_name,
                    )
                )

            succeeded = [m for m in per_seed if "error" not in m]
            if not succeeded:
                self.results[method_name] = {"error": "all seeds failed"}
                mlflow.set_tag("status", "failed")
                return

            agg = self._aggregate_seed_metrics(per_seed)
            mlflow.log_metrics(
                {f"final/{k}": v for k, v in agg.items() if isinstance(v, (int, float))}
            )
            self.results[method_name] = agg
            mlflow.set_tag(
                "status",
                "success" if len(succeeded) == len(per_seed) else "partial_success",
            )

    def _run_single_seed(
        self,
        *,
        child_cfg: Any,
        datamodule: pl.LightningDataModule,
        trainer_cfg: DictConfig,
        seed: int,
        method_run_name: str,
    ) -> Dict[str, Any]:
        """Fit + evaluate one (method, seed) pair inside its own MLflow run."""
        run_name = f"{method_run_name}__seed{seed}"
        with mlflow.start_run(run_name=run_name, nested=True) as seed_run:
            try:
                pl.seed_everything(seed, workers=True)
                seed_cfg = copy.deepcopy(child_cfg)
                with open_dict(seed_cfg):
                    seed_cfg.seed = seed
                mlflow.log_param("seed", seed)
                selector = instantiate(seed_cfg)
                selector.fit(
                    datamodule,
                    trainer_cfg=trainer_cfg,
                    mlflow_run_id=seed_run.info.run_id,
                )
                metrics = selector.evaluate(datamodule)
                mlflow.log_metrics(
                    {f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
                mlflow.set_tag("status", "success")
                return metrics
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e)[:500])
                return {"error": str(e)}

    @staticmethod
    def _aggregate_seed_metrics(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mean across seeds for every numeric metric, plus across-seed SEM.

        For each numeric key ``k``: the aggregate stores ``k`` = mean across
        seeds and (if n_seeds > 1) ``k + SEED_SEM_SUFFIX`` = standard error
        across seeds. Specifically ``wer_sem`` is overridden with the
        across-seed SEM of ``wer_mean`` when n_seeds > 1, so the manuscript
        comparison table reads the manuscript-grade error bar from the same
        key as in single-seed mode.
        """
        succeeded = [m for m in per_seed if "error" not in m]
        if not succeeded:
            return {"error": "all seeds failed"}

        keys = sorted(
            {
                k
                for m in succeeded
                for k, v in m.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        )
        n = len(succeeded)
        out: Dict[str, Any] = {"n_seeds": n}
        for k in keys:
            values = [
                m[k]
                for m in succeeded
                if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            out[k] = mean
            if len(values) > 1:
                var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                out[f"{k}{SEED_SEM_SUFFIX}"] = math.sqrt(var / len(values))

        # Manuscript convention: in multi-seed mode, ``wer_sem`` reports
        # across-seed SEM rather than within-seed sampling SEM.
        if n > 1 and f"wer_mean{SEED_SEM_SUFFIX}" in out:
            out["wer_sem"] = out[f"wer_mean{SEED_SEM_SUFFIX}"]

        return out

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
