import torch
import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import BoringModel
import sys
import os

# Create a dummy test to simulate progress bar
class MyModel(BoringModel):
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log("train/total_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        self.log("val/total_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/selected_wer", loss * 0.5, prog_bar=True, on_step=False, on_epoch=True)
        return loss

if __name__ == "__main__":
    from src.utils.progress import make_progress_bar
    model = MyModel()
    # Force _is_notebook to return True for testing if we want, or False
    bar = make_progress_bar(refresh_rate=10)
    trainer = pl.Trainer(max_epochs=2, limit_train_batches=40, limit_val_batches=10, callbacks=[bar], enable_model_summary=False, logger=False)
    trainer.fit(model)
