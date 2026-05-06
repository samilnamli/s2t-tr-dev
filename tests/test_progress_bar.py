import torch
import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import BoringModel
from src.utils.progress import make_progress_bar

class MyModel(BoringModel):
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        loss = loss["loss"] if isinstance(loss, dict) else loss
        self.log("train/total_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        loss = loss["x"] if isinstance(loss, dict) else loss
        self.log("val/total_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/selected_wer", loss * 0.5, prog_bar=True, on_step=False, on_epoch=True)
        return loss

def test_robust_progress_bar():
    model = MyModel()
    bar = make_progress_bar(refresh_rate=10)
    
    # Run a short training loop to verify progress bar prints properly
    trainer = pl.Trainer(
        max_epochs=2, 
        limit_train_batches=40, 
        limit_val_batches=10, 
        callbacks=[bar], 
        enable_model_summary=False, 
        logger=False
    )
    trainer.fit(model)

if __name__ == "__main__":
    test_robust_progress_bar()
