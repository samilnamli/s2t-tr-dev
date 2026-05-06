import torch
import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import BoringModel
from loguru import logger
from typing import Any

class RobustColabProgressBar(pl.callbacks.ProgressBar):
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

            logger.info(f"Epoch {trainer.current_epoch} | Step {batch_idx + 1}/{total_steps} | Loss: {loss_val}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        if self.is_disabled:
            return
            
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val/total_loss")
        val_wer = metrics.get("val/selected_wer")
        
        loss_str = f"{val_loss.item():.4f}" if val_loss is not None else "N/A"
        wer_str = f"{val_wer.item():.4f}" if val_wer is not None else "N/A"
        
        logger.info(f"Epoch {trainer.current_epoch} Validation | Loss: {loss_str} | WER: {wer_str}")

class MyModel(BoringModel):
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log("train/total_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        self.log("val/total_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/selected_wer", loss * 0.5, prog_bar=False, on_step=False, on_epoch=True)
        return loss

if __name__ == "__main__":
    model = MyModel()
    bar = RobustColabProgressBar(refresh_rate=2)
    trainer = pl.Trainer(max_epochs=2, limit_train_batches=4, limit_val_batches=2, callbacks=[bar], enable_model_summary=False, logger=False)
    trainer.fit(model)
