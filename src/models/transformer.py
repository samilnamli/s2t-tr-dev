"""Hierarchical transformer ASR model selector (proposed architecture)."""

from __future__ import annotations

import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix for PyTorch TransformerEncoder masking bug on Apple Silicon (MPS)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.data.dataset import MODEL_NAMES
from src.models.base import TrainableLightningSelector


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PerModelProjection(nn.Module):
    def __init__(self, model_dims: dict[str, int], d_model: int):
        super().__init__()
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, d_model) for name, dim in model_dims.items()}
        )

    def forward(self, model_name: str, x: torch.Tensor) -> torch.Tensor:
        return self.projections[model_name](x)


class CrossAttentionBridge(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cls_token: torch.Tensor,
        other_sequences: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=cls_token,
            key=other_sequences,
            value=other_sequences,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(cls_token + self.dropout(attn_out))


class ASRModelSelector(nn.Module):
    def __init__(
        self,
        model_dims: dict[str, int],
        model_names: list[str],
        d_model: int = 256,
        n_heads: int = 4,
        stage1_layers: int = 2,
        stage2_layers: int = 1,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_cross_attention_bridge: bool = True,
        share_stage1_weights: bool = True,
        max_seq_len: int = 2500,
    ):
        super().__init__()
        self.model_names = model_names
        self.n_models = len(model_names)
        self.use_cross_attention_bridge = use_cross_attention_bridge
        self.share_stage1_weights = share_stage1_weights

        self.projection = PerModelProjection(model_dims, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )
        self.model_embeddings = nn.Embedding(self.n_models, d_model)
        self.cls_tokens = nn.Parameter(torch.randn(self.n_models, 1, d_model) * 0.02)

        # PyTorch NestedTensor optimization is broken on Apple Silicon (MPS).
        # We disable it locally to avoid crashes, but keep it enabled on CUDA servers
        # for maximum performance.
        enable_nested_tensor = not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

        if share_stage1_weights:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.stage1_encoder = nn.TransformerEncoder(
                enc_layer, num_layers=stage1_layers, enable_nested_tensor=enable_nested_tensor
            )
        else:
            self.stage1_encoders = nn.ModuleDict()
            for name in model_names:
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.stage1_encoders[name] = nn.TransformerEncoder(
                    enc_layer, num_layers=stage1_layers, enable_nested_tensor=enable_nested_tensor
                )

        if use_cross_attention_bridge:
            self.cross_attention_bridges = nn.ModuleDict(
                {name: CrossAttentionBridge(d_model, n_heads, dropout) for name in model_names}
            )

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.stage2_encoder = nn.TransformerEncoder(
            fusion_layer, num_layers=stage2_layers, enable_nested_tensor=enable_nested_tensor
        )

        self.global_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.n_models),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _run_stage1(
        self,
        model_idx: int,
        model_name: str,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = hidden_states.size(0)
        x = self.projection(model_name, hidden_states)
        x = self.pos_encoder(x)
        model_emb = self.model_embeddings(torch.tensor(model_idx, device=x.device))
        x = x + model_emb.unsqueeze(0).unsqueeze(0)

        cls = self.cls_tokens[model_idx].unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        extended_mask = torch.cat([cls_mask, attention_mask], dim=1)
        padding_mask = ~extended_mask

        if self.share_stage1_weights:
            x = self.stage1_encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = self.stage1_encoders[model_name](x, src_key_padding_mask=padding_mask)

        return x[:, 0:1, :], x, padding_mask

    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B = next(iter(hidden_states.values())).size(0)
        cls_outputs: dict[str, torch.Tensor] = {}
        full_sequences: dict[str, torch.Tensor] = {}
        full_padding_masks: dict[str, torch.Tensor] = {}

        for idx, name in enumerate(self.model_names):
            cls_out, full_seq, pad_mask = self._run_stage1(
                idx, name, hidden_states[name], attention_masks[name]
            )
            cls_outputs[name] = cls_out
            full_sequences[name] = full_seq
            full_padding_masks[name] = pad_mask

        if self.use_cross_attention_bridge:
            updated_cls: dict[str, torch.Tensor] = {}
            for name in self.model_names:
                others = [n for n in self.model_names if n != name]
                other_seqs = torch.cat([full_sequences[n][:, 1:, :] for n in others], dim=1)
                other_masks = torch.cat([full_padding_masks[n][:, 1:] for n in others], dim=1)
                updated_cls[name] = self.cross_attention_bridges[name](
                    cls_token=cls_outputs[name],
                    other_sequences=other_seqs,
                    key_padding_mask=other_masks,
                )
            cls_outputs = updated_cls

        global_cls = self.global_cls.expand(B, -1, -1)
        model_summaries = torch.cat([cls_outputs[name] for name in self.model_names], dim=1)
        fusion_input = torch.cat([global_cls, model_summaries], dim=1)
        fusion_output = self.stage2_encoder(fusion_input)

        logits = self.classifier(fusion_output[:, 0, :])
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------------


class ASRModelSelectorLightning(TrainableLightningSelector):
    """Lightning wrapper for the hierarchical transformer architecture."""

    def __init__(
        self,
        model_names: list[str] | None = None,
        d_model: int = 256,
        n_heads: int = 4,
        stage1_layers: int = 2,
        stage2_layers: int = 1,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_cross_attention_bridge: bool = True,
        share_stage1_weights: bool = True,
        max_seq_len: int = 2500,
        primary_weight: float = 1.0,
        aux_ce_weight: float = 0.3,
        soft_ce_weight: float = 0.5,
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
        self._arch_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            stage1_layers=stage1_layers,
            stage2_layers=stage2_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_cross_attention_bridge=use_cross_attention_bridge,
            share_stage1_weights=share_stage1_weights,
            max_seq_len=max_seq_len,
        )
        self._model_dims = model_dims
        if model_dims is not None:
            self._build(model_dims)

    def _build(self, model_dims: dict[str, int]) -> None:
        self._model_dims = model_dims
        self.model = ASRModelSelector(
            model_dims=model_dims,
            model_names=self.model_names,
            **self._arch_kwargs,
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
