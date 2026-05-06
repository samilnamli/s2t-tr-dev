"""Lightning progress-bar selection.

We use a custom progress bar to ensure logs are cleanly printed 
via loguru instead of relying on tqdm, which can spam lines in 
various environments.
"""

from __future__ import annotations

from typing import Any, Union
import pytorch_lightning as pl
import torch
from loguru import logger


class RobustProgressBar(pl.callbacks.ProgressBar):
    """A completely text-based progress bar that never uses \\r or interactive display.
    
    Guaranteed to not spam the DOM or terminal with newlines or broken UI elements.
    It simply logs text at the specified refresh_rate.
    """
    def __init__(self, refresh_rate: int = 10):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.is_disabled = False

    def disable(self) -> None:
        self.is_disabled = True

    def enable(self) -> None:
        self.is_disabled = False

    def _format_value(self, v: Union[float, int, str, torch.Tensor]) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        if hasattr(v, "item"):
            try:
                # Handle single-element tensors
                val = v.item()
                if isinstance(val, float):
                    return f"{val:.4f}"
                return str(val)
            except ValueError:
                # Handle multi-element tensors or other cases
                pass
        return str(v)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.is_disabled or self.refresh_rate <= 0:
            return

        total_steps = trainer.estimated_stepping_batches
        if total_steps == float('inf'):
            total_steps = trainer.num_training_batches

        if (batch_idx + 1) % self.refresh_rate == 0 or (batch_idx + 1) == total_steps:
            metrics = self.get_metrics(trainer, pl_module)
            metrics.pop("v_num", None)
            
            metrics_str = " | ".join(f"{k}: {self._format_value(v)}" for k, v in metrics.items())
            logger.info(f"Epoch {trainer.current_epoch} | Step {batch_idx + 1}/{total_steps} | {metrics_str}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        if self.is_disabled:
            return
            
        metrics = self.get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        
        metrics_str = " | ".join(f"{k}: {self._format_value(v)}" for k, v in metrics.items())
        logger.info(f"Epoch {trainer.current_epoch} Validation | {metrics_str}")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_epoch_end(trainer, pl_module)
        if self.is_disabled:
            return
            
        metrics = self.get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        
        metrics_str = " | ".join(f"{k}: {self._format_value(v)}" for k, v in metrics.items())
        logger.info(f"Test | {metrics_str}")


def make_progress_bar(refresh_rate: int = 50) -> pl.callbacks.Callback:
    """Return a robust text-based progress bar for all environments."""
    return RobustProgressBar(refresh_rate=refresh_rate)
