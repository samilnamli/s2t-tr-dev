"""Base selector abstractions.

Hierarchy
---------
BaseSelector (ABC)
    TrainableLightningSelector   – pl.LightningModule; fit() runs trainer
    TrainingFreeBaseline         – no training; fit() derives priors from data
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
import hydra


# ---------------------------------------------------------------------------
# Hydra structured configs
# ---------------------------------------------------------------------------

@dataclass
class ChildRunConfig:
    """Minimal config for a single child run. Concrete selectors extend this."""
    name: str = "unnamed"
    _target_: str = "src.models.base.TrainableLightningSelector"


@dataclass
class TrainerConfig:
    """pl.Trainer config shared by all trainable child runs."""
    _target_: str = "pytorch_lightning.Trainer"
    max_epochs: int = 50
    accelerator: str = "auto"
    devices: str = "auto"
    deterministic: bool = True
    log_every_n_steps: int = 50
    precision: str = "32"


# ---------------------------------------------------------------------------
# BaseSelector
# ---------------------------------------------------------------------------

class BaseSelector(ABC):
    """Common interface for trainable and training-free ASR routers.

    All subclasses expose:
      - ``fit(datamodule, ...)``         — train or derive priors
      - ``predict_proba(batch)``          — (B, K) soft probabilities
      - ``select(batch)``                 — (B,) integer model indices
      - ``evaluate(datamodule)``          — compute test-split metrics
    """

    def fit(
        self,
        datamodule: pl.LightningDataModule,
        trainer_cfg: Optional[TrainerConfig] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        """Default: no-op. Override in trainable / prior-learning subclasses."""

    @abstractmethod
    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Return (B, K) probability tensor."""

    def select(self, batch: dict) -> np.ndarray:
        """Return (B,) integer indices of the selected model per sample."""
        with torch.no_grad():
            probs = self.predict_proba(batch)
        return probs.argmax(dim=-1).cpu().numpy()

    def evaluate(self, datamodule: pl.LightningDataModule) -> dict:
        """Run select() over the test dataloader and return SelectionMetrics."""
        from src.utils.metrics import SelectionMetrics

        datamodule.setup("test")
        all_idx, all_wer = [], []
        for batch in datamodule.test_dataloader():
            all_idx.append(self.select(batch))
            all_wer.append(batch["wer_matrix"].numpy())

        selected_idx = np.concatenate(all_idx)
        wer_matrix = np.concatenate(all_wer, axis=0)
        model_names = list(datamodule.model_dims.keys())
        return SelectionMetrics.compute_all(selected_idx, wer_matrix, model_names)


# ---------------------------------------------------------------------------
# TrainableLightningSelector
# ---------------------------------------------------------------------------

class TrainableLightningSelector(BaseSelector, pl.LightningModule):
    """Abstract trainable selector.

    Subclasses implement ``forward(hidden_states, attention_masks) -> (B, K)``
    returning *probabilities* (post-softmax).  Everything else — composite
    loss, optimizer, logging, fit/evaluate orchestration — lives here.

    Loss
    ----
    total = primary_weight  * weighted_wer
          + aux_ce_weight   * hard_cross_entropy(log p, argmin WER)
          + soft_ce_weight  * soft_cross_entropy(p, softmax(-wer / T))

    Optimizer
    ---------
    AdamW + linear warmup → cosine annealing.
    """

    def __init__(
        self,
        primary_weight: float = 1.0,
        aux_ce_weight: float = 0.3,
        soft_ce_weight: float = 0.5,
        soft_ce_temperature: float = 1.5,
        label_smoothing: float = 0.1,
        class_balanced_loss: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 200,
        *args: Any,
        **kwargs: Any,
    ):
        pl.LightningModule.__init__(self)
        self.primary_weight = primary_weight
        self.aux_ce_weight = aux_ce_weight
        self.soft_ce_weight = soft_ce_weight
        self.soft_ce_temperature = soft_ce_temperature
        self.label_smoothing = label_smoothing
        self.class_balanced_loss = class_balanced_loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        # class_weights registered later via _init_class_weights()
        self.register_buffer("class_weights", None)

    def _init_class_weights(self, class_priors: list[float]) -> None:
        """Call after datamodule.setup() when class_balanced_loss=True."""
        weights = torch.tensor(
            [1.0 / p if p > 0 else 0.0 for p in class_priors],
            dtype=torch.float32,
        )
        weights = weights / weights.sum() * len(class_priors)
        self.class_weights = weights

    @abstractmethod
    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return (B, K) probability tensor."""

    # ------------------------------------------------------------------
    # Composite loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self, probs: torch.Tensor, wer_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        weighted_wer = (probs * wer_matrix).sum(dim=-1)
        primary_loss = weighted_wer.mean()

        best_idx = wer_matrix.argmin(dim=-1)

        hard_ce = F.cross_entropy(
            torch.log(probs + 1e-8),
            best_idx,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        soft_target = F.softmax(-wer_matrix / self.soft_ce_temperature, dim=-1)
        soft_ce = -(soft_target * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        total_loss = (
            self.primary_weight * primary_loss
            + self.aux_ce_weight * hard_ce
            + self.soft_ce_weight * soft_ce
        )

        selected = probs.argmax(dim=-1)
        oracle_wer = wer_matrix.min(dim=-1).values
        selected_wer = wer_matrix.gather(1, selected.unsqueeze(1)).squeeze(1)
        sel_acc = (selected == best_idx).float().mean()

        metrics = {
            "primary_loss": primary_loss,
            "hard_ce": hard_ce,
            "soft_ce": soft_ce,
            "total_loss": total_loss,
            "selected_wer": selected_wer.mean(),
            "oracle_wer": oracle_wer.mean(),
            "wer_gap": (selected_wer - oracle_wer).mean(),
            "selection_accuracy": sel_acc,
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int = 0) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_matrix"])
        prog = {"total_loss", "selected_wer", "selection_accuracy"}
        for k, v in metrics.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=(k in prog))
        return loss

    def validation_step(self, batch: dict, batch_idx: int = 0) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_matrix"])
        prog = {"total_loss", "selected_wer", "selection_accuracy"}
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=(k in prog))
        return loss

    def test_step(self, batch: dict, batch_idx: int = 0) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_matrix"])
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_epoch=True)
        return loss

    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Unpack batch and delegate to forward()."""
        with torch.no_grad():
            return self(batch["hidden_states"], batch["attention_masks"])

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, total_steps - self.warmup_steps
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ------------------------------------------------------------------
    # BaseSelector.fit / evaluate overrides
    # ------------------------------------------------------------------

    def fit(
        self,
        datamodule: pl.LightningDataModule,
        trainer_cfg: Optional[TrainerConfig] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        if self.class_balanced_loss and hasattr(datamodule, "class_priors"):
            self._init_class_weights(datamodule.class_priors)

        logger = None
        if mlflow_run_id is not None:
            logger = MLFlowLogger(
                run_id=mlflow_run_id,
                tracking_uri=mlflow.get_tracking_uri(),
                log_model=True,
            )

        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=1,
        )
        lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="step")

        with tempfile.TemporaryDirectory() as tmp_dir:
            if trainer_cfg is not None:
                trainer: pl.Trainer = hydra.utils.instantiate(
                    trainer_cfg,
                    default_root_dir=tmp_dir,
                    logger=logger,
                    callbacks=[checkpoint_cb, lr_cb],
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=50,
                    accelerator="auto",
                    default_root_dir=tmp_dir,
                    logger=logger,
                    callbacks=[checkpoint_cb, lr_cb],
                )
            trainer.fit(self, datamodule=datamodule)
            test_results = trainer.test(self, datamodule=datamodule, ckpt_path="best")

        self._last_test_results = test_results[0] if test_results else {}

    def evaluate(self, datamodule: pl.LightningDataModule) -> dict:
        """Return last test results from fit(); also compute selection metrics."""
        from src.utils.metrics import SelectionMetrics

        lightning_metrics = getattr(self, "_last_test_results", {})

        # Also compute routing metrics
        datamodule.setup("test")
        all_idx, all_wer = [], []
        self.eval()
        with torch.no_grad():
            for batch in datamodule.test_dataloader():
                probs = self(batch["hidden_states"], batch["attention_masks"])
                all_idx.append(probs.argmax(dim=-1).cpu().numpy())
                all_wer.append(batch["wer_matrix"].numpy())

        selected_idx = np.concatenate(all_idx)
        wer_matrix = np.concatenate(all_wer, axis=0)
        model_names = list(datamodule.model_dims.keys())
        routing_metrics = SelectionMetrics.compute_all(selected_idx, wer_matrix, model_names)
        return {**lightning_metrics, **routing_metrics}


# ---------------------------------------------------------------------------
# TrainingFreeBaseline base
# ---------------------------------------------------------------------------

class TrainingFreeBaseline(BaseSelector):
    """Base for training-free routing baselines.

    Subclasses implement ``select(batch) -> np.ndarray`` returning (B,) indices.
    ``predict_proba`` wraps select() as one-hot for interface uniformity.
    """

    def __init__(self, model_names: list[str], seed: int = 42, **kwargs: Any):
        self.model_names = list(model_names)
        self.K = len(model_names)
        self.seed = seed

    def predict_proba(self, batch: dict) -> torch.Tensor:
        idx = self.select(batch)
        B = len(idx)
        one_hot = torch.zeros(B, self.K)
        one_hot[torch.arange(B), torch.from_numpy(idx)] = 1.0
        return one_hot

    @abstractmethod
    def select(self, batch: dict) -> np.ndarray:
        """Return (B,) integer array of selected model indices."""

    def select_all(self, batches: list[dict]) -> np.ndarray:
        return np.concatenate([self.select(b) for b in batches])

    def evaluate(self, datamodule: pl.LightningDataModule) -> dict:
        from src.utils.metrics import SelectionMetrics

        datamodule.setup("test")
        all_idx, all_wer = [], []
        for batch in datamodule.test_dataloader():
            all_idx.append(self.select(batch))
            all_wer.append(batch["wer_matrix"].numpy())

        selected_idx = np.concatenate(all_idx)
        wer_matrix = np.concatenate(all_wer, axis=0)
        return SelectionMetrics.compute_all(selected_idx, wer_matrix, self.model_names)
