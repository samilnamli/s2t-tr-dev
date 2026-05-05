from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import re
import tempfile

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.base import ChildRunConfig
from src.data.base import DataModuleConfig


@dataclass
class TrainerConfig:
    """pl.Trainer için config. Tüm child run'lar paylaşır — fair comparison."""
    _target_: str = "pytorch_lightning.Trainer"
    max_epochs: int = 100
    accelerator: str = "auto"
    devices: str = "auto"
    deterministic: bool = True
    log_every_n_steps: int = 50


# Parent run = experiment'in tek bir denemesi. Manuscript'te anlam ifade eden
# en küçük bütün. MLflow experiment'i environment'tan çekiliyor.
@dataclass
class ParentRunConfig:
    parent_run_name: str = "synthetic_v1"
    seed: int = 42

    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    child_runs: List[Any] = field(default_factory=lambda: [
        ChildRunConfig(name="model_a"),
        ChildRunConfig(name="model_b"),
    ])

    data: DataModuleConfig = field(default_factory=DataModuleConfig)


class BaseExperiment(ABC):
    def __init__(self, cfg: ParentRunConfig):
        self.cfg = cfg
        self.results: Dict[str, Dict[str, Any]] = {}
        pl.seed_everything(cfg.seed, workers=True)

    def run(self) -> Dict[str, Dict[str, Any]]:
        with mlflow.start_run(run_name=self.cfg.parent_run_name) as parent_run:
            self.parent_run_id = parent_run.info.run_id
            mlflow.log_params({
                "seed": self.cfg.seed,
                "n_children": len(self.cfg.child_runs),
            })
            mlflow.log_dict(
                OmegaConf.to_container(OmegaConf.structured(self.cfg), resolve=True),
                "parent_config.yaml",
            )

            self.setup()
            self.run_child_runs()
            self.aggregate()

        return self.results

    def setup(self) -> None:
        self.datamodule = instantiate(self.cfg.data)
        self.datamodule.prepare_data()
        self.datamodule.setup()

    def run_child_runs(self) -> None:
        for child_cfg in self.cfg.child_runs:
            child_short_name = getattr(child_cfg, "name", "unnamed")
            child_run_name = f"{self.cfg.parent_run_name}_{child_short_name}"
            print(f"[{self.cfg.parent_run_name}] Running child: {child_run_name}")

            with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                # Trainer'ın yazacağı geçici dizin — fit bitince temizlenir.
                # MLflow zaten artifact'leri kendine kopyalıyor.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        self._run_single_child(child_cfg, child_short_name, child_run, tmp_dir)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        mlflow.set_tag("status", "failed")
                        mlflow.set_tag("error", str(e)[:250])
                        self.results[child_short_name] = {"error": str(e)}

    def _run_single_child(
        self,
        child_cfg: Any,
        child_short_name: str,
        child_run: mlflow.ActiveRun,
        tmp_dir: str,
    ) -> None:
        model: pl.LightningModule = instantiate(child_cfg)

        # log_model=True → best checkpoint MLflow'a otomatik yüklenir.
        # run_id inject ediyoruz ki logger yeni run açmasın.
        logger = MLFlowLogger(
            run_id=child_run.info.run_id,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model=True,
        )

        # # Filename template: best_epoch'u path'ten parse edebilelim diye.
        # checkpoint_cb = pl.callbacks.ModelCheckpoint(
        #     filename="epoch={epoch:03d}-val_loss={val/loss:.4f}",
        #     auto_insert_metric_name=False,
        #     monitor="val/loss",
        #     mode="min",
        #     save_last=False,
        # )
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',  
            mode='min',          
            save_top_k=1,        # Sadece en iyi 1 modeli sakla
        )
        lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        trainer: pl.Trainer = instantiate(
            self.cfg.trainer,
            default_root_dir=tmp_dir,
            logger=logger,
            callbacks=[checkpoint_cb, lr_cb],
        )

        trainer.fit(model, datamodule=self.datamodule)

        # best_epoch = self._extract_best_epoch(checkpoint_cb.best_model_path)
        # best_val_score = (
        #     checkpoint_cb.best_model_score.item()
        #     if checkpoint_cb.best_model_score is not None
        #     else None
        # )

        test_results = trainer.test(
            model,
            datamodule=self.datamodule,
            ckpt_path="best",
        )
        test_metrics = test_results[0] if test_results else {}

        summary_metrics = {
            # "best_epoch": best_epoch,
            # "best_val_score": best_val_score,
            **{f"final/{k}": v for k, v in test_metrics.items()},
        }
        mlflow.log_metrics(
            {k: v for k, v in summary_metrics.items() if v is not None}
        )

        self.results[child_short_name] = {
            # "best_epoch": best_epoch,
            # "best_val_score": best_val_score,
            "test_metrics": test_metrics,
            "mlflow_run_id": child_run.info.run_id,
        }

    @staticmethod
    def _extract_best_epoch(ckpt_path: str) -> Optional[int]:
        if not ckpt_path:
            return None
        match = re.search(r"epoch=(\d+)", ckpt_path)
        return int(match.group(1)) if match else None

    @abstractmethod
    def aggregate(self) -> None:
        ...