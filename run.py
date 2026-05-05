"""Experiment entry point.

Usage
-----
# AMI main results (all 10 child runs):
    python run.py experiment=ami_main_results

# Quick smoke test on AMI with 2 epochs, lazy mode, small batch:
    python run.py experiment=ami_main_results \\
        experiment.trainer.max_epochs=2 \\
        experiment.data.batch_size=8 \\
        experiment.data.eager_load=false

# Override MLflow tracking URI:
    python run.py mlflow.tracking_uri=http://localhost:5000 \\
                  mlflow.experiment_name=my_experiment
"""

import hydra
import mlflow
from hydra.utils import instantiate

from src.experiments import RootConfig  # also registers all ConfigStore nodes


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: RootConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    experiment = instantiate(cfg.experiment, _recursive_=False)
    experiment.run()


if __name__ == "__main__":
    main()
