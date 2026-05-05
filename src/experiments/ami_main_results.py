"""AMI main-results experiment.

Implements the comparison from configs/experiments/main_results_ami.yaml:
  8 training-free baselines  +  2 trainable routers  =  10 child runs.

Expected test WER (AMI ihm, from yaml hypothesis):
  single best model     ~0.40
  random / w-random     ~0.40
  ROVER / w-ROVER       ~0.35–0.37
  MLP-pool baseline     ~0.35–0.37
  Proposed (hier. tf.)  ~0.31–0.33
  Oracle                ~0.24
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from src.data.ami import AMIDataModuleConfig
from src.data.dataset import MODEL_NAMES
from src.experiments.base import BaseExperiment, ParentRunConfig, TrainerConfig
from src.models.baselines import (
    OracleBaselineConfig,
    RandomBaselineConfig,
    ROVERBaselineConfig,
    SingleModelBaselineConfig,
    WeightedRandomBaselineConfig,
    WeightedROVERBaselineConfig,
)
from src.models.mlp import MLPPoolSelectorConfig
from src.models.transformer import HierarchicalTransformerConfig


def _default_child_runs() -> List[Any]:
    return [
        # ── Training-free ──────────────────────────────────────────
        SingleModelBaselineConfig(name="single_hubert",   model_idx=0,
                                  model_names=MODEL_NAMES),
        SingleModelBaselineConfig(name="single_whisper",  model_idx=1,
                                  model_names=MODEL_NAMES),
        SingleModelBaselineConfig(name="single_wav2vec2", model_idx=2,
                                  model_names=MODEL_NAMES),
        RandomBaselineConfig(name="random",               model_names=MODEL_NAMES),
        WeightedRandomBaselineConfig(name="weighted_random",
                                     model_names=MODEL_NAMES),
        OracleBaselineConfig(name="oracle",               model_names=MODEL_NAMES),
        ROVERBaselineConfig(name="rover",                 model_names=MODEL_NAMES),
        WeightedROVERBaselineConfig(name="weighted_rover",model_names=MODEL_NAMES),
        # ── Trainable ──────────────────────────────────────────────
        MLPPoolSelectorConfig(
            name="mlp_pool_hard_ce",
            primary_weight=0.0,
            aux_ce_weight=1.0,
            soft_ce_weight=0.0,
            class_balanced_loss=True,
        ),
        HierarchicalTransformerConfig(
            name="hierarchical_transformer",
            primary_weight=1.0,
            aux_ce_weight=0.3,
            soft_ce_weight=0.5,
            soft_ce_temperature=1.5,
            class_balanced_loss=True,
        ),
    ]


@dataclass
class AMIMainResultsConfig(ParentRunConfig):
    _target_: str = "src.experiments.ami_main_results.AMIMainResultsExperiment"
    parent_run_name: str = "ami_main_results"
    seed: int = 42
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(
        max_epochs=50,
        precision="bf16-mixed",
    ))
    data: AMIDataModuleConfig = field(default_factory=lambda: AMIDataModuleConfig(
        batch_size=128,
        num_workers=4,
        max_seq_len=2000,
        eager_load=False,
    ))
    child_runs: List[Any] = field(default_factory=_default_child_runs)


class AMIMainResultsExperiment(BaseExperiment):
    """Runs all 10 child selectors on AMI and logs a WER comparison table."""

    def __init__(
        self,
        parent_run_name: str = "ami_main_results",
        seed: int = 42,
        trainer=None,
        data=None,
        child_runs=None,
        **kwargs,
    ):
        # Use the default child runs if not provided
        if child_runs is None:
            child_runs = _default_child_runs()
        super().__init__(
            parent_run_name=parent_run_name,
            seed=seed,
            trainer=trainer,
            data=data,
            child_runs=child_runs,
            **kwargs,
        )

    def child_run_configs(self) -> List[Any]:
        return self.cfg.child_runs
