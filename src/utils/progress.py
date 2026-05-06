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

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        # Wait for validation to print train and val metrics together
        if trainer.enable_validation and trainer.val_dataloaders is not None:
            return
            
        if self.is_disabled or trainer.sanity_checking:
            return
            
        metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
        metrics.pop("v_num", None)
        metrics_str = " | ".join(f"{k}: {self._format_value(v)}" for k, v in metrics.items())
        if metrics_str:
            logger.info(f"Epoch {trainer.current_epoch} | {metrics_str}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        if self.is_disabled or trainer.sanity_checking:
            return
            
        metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
        metrics.pop("v_num", None)
        
        metrics_str = " | ".join(f"{k}: {self._format_value(v)}" for k, v in metrics.items())
        if metrics_str:
            logger.info(f"Epoch {trainer.current_epoch} | {metrics_str}")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_epoch_end(trainer, pl_module)
        if self.is_disabled:
            return
            
        metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
        metrics.pop("v_num", None)
        
        metrics_str = " | ".join(f"{k}: {self._format_value(v)}" for k, v in metrics.items())
        if metrics_str:
            logger.info(f"Test | {metrics_str}")


def make_progress_bar(refresh_rate: int = 50) -> pl.callbacks.Callback:
    """Return a robust text-based progress bar for all environments."""
    return RobustProgressBar(refresh_rate=refresh_rate)
