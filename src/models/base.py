from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torch
import hydra 
from dataclasses import dataclass

# burada child run diye geçen şey, aslında individual modelin trainingi.
@dataclass
class ChildRunConfig:
    name: str = "synthetic_v1"


class BaseASRSelectorLightningModule(pl.LightningModule, ABC):
    """
    Abstract base module handling common PyTorch Lightning logic:
    optimizer instantiation, learning rate scheduling, metric logging,
    and the shared classification loss computation.
    """

    def __init__(self, optimizer_cfg=None, scheduler_cfg=None, *args, **kwargs):
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.save_hyperparameters(ignore=["optimizer_cfg", "scheduler_cfg"])

    @abstractmethod
    def forward(self, hidden_states: dict[str, torch.Tensor], attention_masks: dict[str, torch.Tensor]):
        """Forward pass must be implemented by subclasses."""
        pass

    def compute_loss(self, batch, probs):
        """Shared loss and metric computation for all ASR selectors."""
        # Assumes the batch dictionary contains a 'targets' key mapping to the ideal ASR model index
        targets = batch["targets"]
        
        # Using NLL Loss since models output probabilities (Softmax).
        # Log is added for numerical stability to conform to NLL requirements.
        loss = F.nll_loss(torch.log(probs + 1e-8), targets)
        
        preds = torch.argmax(probs, dim=-1)
        accuracy = (preds == targets).float().mean()
        
        metrics = {
            "total_loss": loss,
            "accuracy": accuracy
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        probs = self.forward(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self.compute_loss(batch, probs)
        for k, v in metrics.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=(k in {"total_loss", "accuracy"}))
        return loss

    def validation_step(self, batch, batch_idx):
        probs = self.forward(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self.compute_loss(batch, probs)
        for k, v in metrics.items():
            self.log(f"val_{k}", v, on_epoch=True, prog_bar=(k in {"total_loss", "accuracy"}))
        return loss

    def test_step(self, batch, batch_idx):
        probs = self.forward(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self.compute_loss(batch, probs)
        for k, v in metrics.items():
            self.log(f"test_{k}", v, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_cfg is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
            return {"optimizer": optimizer}
            
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
        
        if self.scheduler_cfg is None:
            return {"optimizer": optimizer}
            
        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

