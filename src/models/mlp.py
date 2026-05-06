"""MLP-pool ASR model selector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import MODEL_NAMES
from src.models.base import TrainableLightningSelector


class MLPPoolSelector(nn.Module):
    """Mean-pool each expert's frames then classify."""

    def __init__(
        self,
        model_dims: dict[str, int],
        model_names: list[str],
        d_hidden: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_names = model_names
        self.n_models = len(model_names)
        total_in = sum(model_dims[n] for n in model_names)

        layers: list[nn.Module] = []
        prev = total_in
        for _ in range(n_layers):
            layers += [nn.Linear(prev, d_hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = d_hidden
        layers.append(nn.Linear(prev, self.n_models))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
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


class MLPPoolSelectorLightning(TrainableLightningSelector):
    """Lightning wrapper for MLPPoolSelector."""

    def __init__(
        self,
        model_names: list[str] | None = None,
        d_hidden: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.1,
        primary_weight: float = 0.0,
        aux_ce_weight: float = 1.0,
        soft_ce_weight: float = 0.0,
        soft_ce_temperature: float = 1.5,
        label_smoothing: float = 0.1,
        class_balanced_loss: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 200,
        early_stopping_patience: int | None = None,
        model_dims: dict[str, int] | None = None,
        **kwargs,
    ):
        super().__init__(
            primary_weight=primary_weight,
            aux_ce_weight=aux_ce_weight,
            soft_ce_weight=soft_ce_weight,
            soft_ce_temperature=soft_ce_temperature,
            label_smoothing=label_smoothing,
            class_balanced_loss=class_balanced_loss,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            early_stopping_patience=early_stopping_patience,
        )
        self.save_hyperparameters()
        self.model_names = list(model_names) if model_names is not None else list(MODEL_NAMES)
        self._d_hidden = d_hidden
        self._n_layers = n_layers
        self._dropout = dropout
        self._model_dims = model_dims
        if model_dims is not None:
            self._build(model_dims)

    def _build(self, model_dims: dict[str, int]) -> None:
        self._model_dims = model_dims
        self.model = MLPPoolSelector(
            model_dims=model_dims,
            model_names=self.model_names,
            d_hidden=self._d_hidden,
            n_layers=self._n_layers,
            dropout=self._dropout,
        )

    def fit(self, datamodule, trainer_cfg=None, mlflow_run_id=None) -> None:
        if self._model_dims is None:
            self._build(datamodule.model_dims)
        super().fit(datamodule, trainer_cfg=trainer_cfg, mlflow_run_id=mlflow_run_id)

    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.model(hidden_states, attention_masks)
