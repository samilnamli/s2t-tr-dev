"""Register all structured configs with the Hydra ConfigStore."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.data.ami import AMIDataModuleConfig
from src.data.base import ASRDataModuleConfig
from src.experiments.ami_main_results import AMIMainResultsConfig
from src.experiments.base import ParentRunConfig, TrainerConfig
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


@dataclass
class MLflowConfig:
    tracking_uri: str = "mlruns"
    experiment_name: str = "ASR_Model_Selector"


@dataclass
class RootConfig:
    """Top-level Hydra config composed from the `experiment` config group."""
    experiment: ParentRunConfig = MISSING
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


cs = ConfigStore.instance()

# Data
cs.store(group="data", name="base",      node=ASRDataModuleConfig)
cs.store(group="data", name="ami",       node=AMIDataModuleConfig)

# Models
cs.store(group="model", name="mlp_pool",                 node=MLPPoolSelectorConfig)
cs.store(group="model", name="hierarchical_transformer", node=HierarchicalTransformerConfig)
cs.store(group="model", name="single_model",             node=SingleModelBaselineConfig)
cs.store(group="model", name="oracle",                   node=OracleBaselineConfig)
cs.store(group="model", name="random",                   node=RandomBaselineConfig)
cs.store(group="model", name="weighted_random",          node=WeightedRandomBaselineConfig)
cs.store(group="model", name="rover",                    node=ROVERBaselineConfig)
cs.store(group="model", name="weighted_rover",           node=WeightedROVERBaselineConfig)

# Trainer
cs.store(group="trainer", name="default", node=TrainerConfig)

# Experiments
cs.store(group="experiment", name="ami_main_results", node=AMIMainResultsConfig)

# Root config
cs.store(name="config", node=RootConfig)
