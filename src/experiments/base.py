"""Base experiment orchestration.

Parent run  = one row in the manuscript comparison table (e.g., "AMI main results").
Child run   = one routing method evaluated under the same conditions.

MLflow layout
-------------
  parent run  (ParentRunConfig.parent_run_name)
    ├─ child run  selector_a   (nested=True)
    ├─ child run  selector_b
    └─ ...
  artifact: results/test_wer_comparison.json  (on parent)
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import mlflow
import pandas as pd
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data.base import ASRDataModuleConfig
from src.models.base import ChildRunConfig, TrainerConfig


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class ParentRunConfig:
    """Top-level config for one experiment (= one manuscript comparison row)."""
    parent_run_name: str = "experiment"
    seed: int = 42
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: ASRDataModuleConfig = field(default_factory=ASRDataModuleConfig)
    child_runs: List[Any] = field(default_factory=lambda: [ChildRunConfig()])


# ---------------------------------------------------------------------------
# BaseExperiment
# ---------------------------------------------------------------------------

class BaseExperiment(ABC):
    """Orchestrates a parent MLflow run containing multiple child selector runs.

    Subclasses implement :meth:`child_run_configs` to return the list of
    selector configs to compare.  The base class handles DataModule setup,
    MLflow nesting, error recovery, and the final comparison table.
    """

    def __init__(
        self,
        parent_run_name: str = "experiment",
        seed: int = 42,
        trainer=None,
        data=None,
        child_runs=None,
        **kwargs,  # absorb _target_ and other Hydra-injected keys
    ):
        # Hydra passes fields as kwargs; wrap in a namespace for dot-access
        self.cfg = SimpleNamespace(
            parent_run_name=parent_run_name,
            seed=seed,
            trainer=trainer if trainer is not None else TrainerConfig(),
            data=data if data is not None else ASRDataModuleConfig(),
            child_runs=child_runs if child_runs is not None else [],
        )
        self.results: Dict[str, Dict[str, Any]] = {}
        pl.seed_everything(seed, workers=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Dict[str, Any]]:
        with mlflow.start_run(run_name=self.cfg.parent_run_name) as parent_run:
            self.parent_run_id = parent_run.info.run_id
            mlflow.log_params({
                "seed": self.cfg.seed,
                "n_children": len(self.child_run_configs()),
            })
            try:
                cfg_dict = OmegaConf.to_container(
                    OmegaConf.structured(self.cfg), resolve=True
                )
            except Exception:
                cfg_dict = {}
            mlflow.log_dict(cfg_dict, "parent_config.yaml")

            self._setup_datamodule()

            for child_cfg in self.child_run_configs():
                self._run_single_child(child_cfg)

            self._log_comparison_table()

        return self.results

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def child_run_configs(self) -> List[Any]:
        """Return ordered list of child selector configs."""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _setup_datamodule(self) -> None:
        self.datamodule = instantiate(self.cfg.data)
        self.datamodule.prepare_data()
        self.datamodule.setup()

    def _run_single_child(self, child_cfg: Any) -> None:
        child_name = getattr(child_cfg, "name", "unnamed")
        child_run_name = f"{self.cfg.parent_run_name}__{child_name}"
        print(f"[{self.cfg.parent_run_name}] Running child: {child_name}")

        with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
            try:
                selector = instantiate(child_cfg)
                selector.fit(
                    self.datamodule,
                    trainer_cfg=self.cfg.trainer,
                    mlflow_run_id=child_run.info.run_id,
                )
                metrics = selector.evaluate(self.datamodule)

                loggable = {
                    f"final/{k}": v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                mlflow.log_metrics(loggable)
                self.results[child_name] = metrics
                mlflow.set_tag("status", "success")

            except Exception as e:
                traceback.print_exc()
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e)[:500])
                self.results[child_name] = {"error": str(e)}

    def _log_comparison_table(self) -> None:
        """Build a WER comparison table and log it to the parent MLflow run."""
        rows = []
        for name, metrics in self.results.items():
            if "error" in metrics:
                continue
            rows.append({
                "name": name,
                "wer_mean": metrics.get("wer_mean"),
                "wer_sem": metrics.get("wer_sem"),
                "selection_accuracy": metrics.get("selection_accuracy"),
                "n": metrics.get("n"),
            })

        if not rows:
            return

        df = pd.DataFrame(rows).sort_values("wer_mean")

        # Parent run is still active when this is called — log directly.
        mlflow.log_table(
            data=df,
            artifact_file="results/test_wer_comparison.json",
        )

        # Also print to console for quick inspection
        print("\n=== Test WER Comparison ===")
        print(df.to_string(index=False))
        print()
