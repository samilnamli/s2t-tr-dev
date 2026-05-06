"""Lightning progress-bar selection that works in both notebooks and TTYs.

Colab's stderr proxy mishandles ``\\r`` once any non-CR-terminated line
sneaks between two ``tqdm`` refreshes — and the unified loguru sink in
:mod:`src.utils.logging` regularly emits such lines from Lightning's
stdlib loggers. The combination produces one new line per training
step instead of an in-place progress bar.

Strategy:
    * In a Jupyter / Colab front-end → return a custom :class:`RobustColabProgressBar`
      which avoids all interactive display elements and simply logs standard text
      via loguru, guaranteeing it cannot be broken by the Colab DOM.
    * In a TTY → return :class:`TQDMProgressBar` from ``tqdm.auto``.
"""

from __future__ import annotations

from typing import Any
from loguru import logger
import pytorch_lightning as pl


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    # ZMQInteractiveShell = jupyter / colab; TerminalInteractiveShell = ipython tty.
    name = ip.__class__.__name__
    if name == "ZMQInteractiveShell":
        return True
    if "google.colab" in str(ip.__class__):
        return True
    return False


class RobustColabProgressBar(pl.callbacks.ProgressBar):
    """A completely text-based progress bar that never uses \\r or interactive display.
    
    Guaranteed to not spam the Colab DOM with newlines or broken UI elements.
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
            loss_val = "N/A"
            if isinstance(outputs, dict) and "loss" in outputs:
                loss_val = f"{outputs['loss'].item():.4f}"
            elif hasattr(outputs, "item"):
                loss_val = f"{outputs.item():.4f}"
            elif isinstance(outputs, (float, int)):
                loss_val = f"{outputs:.4f}"

            logger.info(f"Epoch {trainer.current_epoch} | Step {batch_idx + 1}/{total_steps} | Train Loss: {loss_val}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        if self.is_disabled:
            return
            
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val/total_loss")
        val_wer = metrics.get("val/selected_wer")
        
        loss_str = f"{val_loss.item():.4f}" if hasattr(val_loss, "item") else (f"{val_loss:.4f}" if isinstance(val_loss, (float, int)) else "N/A")
        wer_str = f"{val_wer.item():.4f}" if hasattr(val_wer, "item") else (f"{val_wer:.4f}" if isinstance(val_wer, (float, int)) else "N/A")
        
        logger.info(f"Epoch {trainer.current_epoch} Validation | Loss: {loss_str} | WER: {wer_str}")


def make_progress_bar(refresh_rate: int = 50) -> pl.callbacks.Callback:
    """Return a Lightning progress-bar callback appropriate for the env."""
    if _is_notebook():
        return RobustColabProgressBar(refresh_rate=refresh_rate)
    return pl.callbacks.TQDMProgressBar(refresh_rate=refresh_rate)
