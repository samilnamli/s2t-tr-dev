import torch
import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import BoringModel

class MyModel(BoringModel):
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        return loss

class TestCB(pl.callbacks.ProgressBar):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print(f"batch_idx: {batch_idx}, outputs: {outputs}")

if __name__ == "__main__":
    model = MyModel()
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=2, limit_val_batches=0, callbacks=[TestCB()], enable_model_summary=False, logger=False)
    trainer.fit(model)
