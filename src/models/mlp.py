
import torch
import torch.nn as nn
from src.experiments.base import BaseASRSelectorLightningModule

class MLPPoolSelector(nn.Module):
    # ... [Keep your exact MLPPoolSelector nn.Module implementation here] ...
    def __init__(self, model_dims: dict[str, int], model_names: list[str], d_hidden: int = 1024, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.model_names = model_names
        self.n_models = len(model_names)
        total_in = sum(model_dims[name] for name in model_names)

        layers: list[nn.Module] = []
        prev = total_in
        for _ in range(n_layers):
            layers += [nn.Linear(prev, d_hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = d_hidden
        layers.append(nn.Linear(prev, self.n_models))
        self.classifier = nn.Sequential(*layers)

    def forward(self, hidden_states: dict[str, torch.Tensor], attention_masks: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = []
        for name in self.model_names:
            x = hidden_states[name]
            mask = attention_masks[name].unsqueeze(-1).to(x.dtype)
            summed = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled.append(summed / denom)
        feat = torch.cat(pooled, dim=-1)
        logits = self.classifier(feat)
        return F.softmax(logits, dim=-1)


class MLPPoolSelectorLightning(BaseASRSelectorLightningModule):
    """Lightning Module wrapper for the MLPPoolSelector architecture."""

    def __init__(self, model_dims: dict[str, int], model_names: list[str], d_hidden: int = 1024, n_layers: int = 2, dropout: float = 0.1, optimizer_cfg=None, scheduler_cfg=None):
        super().__init__(optimizer_cfg, scheduler_cfg)
        self.model = MLPPoolSelector(
            model_dims=model_dims,
            model_names=model_names,
            d_hidden=d_hidden,
            n_layers=n_layers,
            dropout=dropout
        )
        self.save_hyperparameters()

    def forward(self, hidden_states: dict[str, torch.Tensor], attention_masks: dict[str, torch.Tensor]):
        return self.model(hidden_states, attention_masks)

