"""Experiment entry point.

Examples
--------
    # AMI main results (all child runs):
    python run.py experiment=main_results_ami

    # Quick smoke test:
    python run.py experiment=smoke

    # CLI overrides flow through Hydra:
    python run.py experiment=main_results_ami trainer.max_epochs=2 data.batch_size=8

    # Override MLflow tracking URI (DagsHub by default if env is set):
    python run.py mlflow.tracking_uri=http://localhost:5000
"""

from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

from src.utils.logging import setup_unified_logging
from src.utils.mlflow_setup import setup_mlflow


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging_kwargs = {"level": cfg.logging.level}
    if "fmt" in cfg.logging:
        logging_kwargs["fmt"] = cfg.logging.fmt
    setup_unified_logging(**logging_kwargs)

    precision = cfg.get("float32_matmul_precision", "high")
    torch.set_float32_matmul_precision(precision)

    mlflow_cfg = OmegaConf.to_container(cfg.mlflow, resolve=True)
    if not mlflow_cfg.get("experiment_name"):
        mlflow_cfg["experiment_name"] = HydraConfig.get().runtime.choices.experiment
    setup_mlflow(OmegaConf.create(mlflow_cfg))

    from hydra.utils import instantiate
    experiment = instantiate(cfg.experiment, _recursive_=False)
    experiment.run(cfg.experiment)


if __name__ == "__main__":
    main()
