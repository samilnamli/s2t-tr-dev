"""PyTorch Lightning training script for the ASR Model Selector.

Architectures (see ``--arch``):
    hierarchical_transformer (default): the proposed hierarchical
        transformer router from :mod:`src.models.selector`.
    mlp_pool: a mean-pool MLP baseline from :mod:`src.models.mlp_pool`
        used in the synthetic experiment and as a real-world baseline.

Loss:
    Primary: Weighted WER = sum(p_k * wer_k) per sample, averaged over batch.
        Differentiable through softmax; pushes probability toward the model
        with the lowest WER for each clip.
    Auxiliary: Cross-entropy on the hard best-model label (with label smoothing).
        Aids early convergence by providing a stronger gradient signal.
        Supports class-balancing to prevent shadowing of minority experts (e.g., HuBERT).
"""

import json
import os
import sys
import time
from typing import Optional

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.data.dataset import MODEL_NAMES, ASRFeatureDataset, collate_fn
from src.models.mlp_pool import MLPPoolSelector
from src.models.selector import ASRModelSelector
from src.utils.checkpoint import legacy_torch_load
from src.utils.git import git_state
from src.utils.logging import setup_unified_logging

ARCH_HIERARCHICAL = "hierarchical_transformer"
ARCH_MLP_POOL = "mlp_pool"
SUPPORTED_ARCHS = (ARCH_HIERARCHICAL, ARCH_MLP_POOL)

MODEL_DIMS = {
    "hubert":   1024,  # facebook/hubert-large-ls960-ft
    "whisper":  512,   # openai/whisper-base
    "wav2vec2": 1024,   # facebook/wav2vec2-base-960h
}

# Matmul precision will be set in train() based on cfg


class EpochSummary(Callback):
    """One concise log line per epoch — designed for non-tty subprocess output."""

    def __init__(self):
        super().__init__()
        self._epoch_start: Optional[float] = None
        self._val_start: Optional[float] = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start is None:
            return
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        elapsed = time.time() - self._epoch_start
        logger.info(
            "Epoch {}/{} — {:.1f}s — "
            "train_loss={:.4f} train_wer={:.4f} train_acc={:.4f}  "
            "val_loss={:.4f} val_wer={:.4f} val_acc={:.4f}",
            trainer.current_epoch + 1, trainer.max_epochs, elapsed,
            m.get("train/total_loss_epoch", float("nan")),
            m.get("train/selected_wer_epoch", float("nan")),
            m.get("train/selection_accuracy_epoch", float("nan")),
            m.get("val/total_loss", float("nan")),
            m.get("val/selected_wer", float("nan")),
            m.get("val/selection_accuracy", float("nan")),
        )


class SaveSpecificEpochsCallback(Callback):
    """Saves checkpoints for specific epochs explicitly requested by the user."""
    def __init__(self, epochs, dirpath):
        super().__init__()
        self.epochs = set(epochs)
        self.dirpath = dirpath

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        if current_epoch in self.epochs:
            os.makedirs(self.dirpath, exist_ok=True)
            ckpt_path = os.path.join(self.dirpath, f"epoch-{current_epoch:02d}.ckpt")
            trainer.save_checkpoint(ckpt_path)
            logger.info("Saved specific epoch checkpoint: {}", ckpt_path)


class ASRSelectorModule(pl.LightningModule):
    """Lightning wrapper for the ASR Model Selector."""

    def __init__(
        self,
        arch: str = ARCH_HIERARCHICAL,
        d_model: int = 256,
        n_heads: int = 4,
        stage1_layers: int = 2,
        stage2_layers: int = 1,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_cross_attention_bridge: bool = True,
        share_stage1_weights: bool = True,
        max_seq_len: int = 2000,
        mlp_hidden: int = 1024,
        mlp_layers: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 200,
        total_steps: int = 5000,
        primary_weight: float = 1.0,
        aux_ce_weight: float = 0.3,
        soft_ce_weight: float = 0.0,
        soft_ce_temperature: float = 0.1,
        label_smoothing: float = 0.1,
        class_balanced_loss: bool = False,
        class_priors: Optional[list[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if arch not in SUPPORTED_ARCHS:
            raise ValueError(
                f"Unknown arch={arch!r}; expected one of {SUPPORTED_ARCHS}."
            )
        self.arch = arch

        if arch == ARCH_HIERARCHICAL:
            self.model = ASRModelSelector(
                model_dims=MODEL_DIMS,
                model_names=MODEL_NAMES,
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
        else:
            self.model = MLPPoolSelector(
                model_dims=MODEL_DIMS,
                model_names=MODEL_NAMES,
                d_hidden=mlp_hidden,
                n_layers=mlp_layers,
                dropout=dropout,
            )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.primary_weight = primary_weight
        self.aux_ce_weight = aux_ce_weight
        self.soft_ce_weight = soft_ce_weight
        self.soft_ce_temperature = soft_ce_temperature
        self.label_smoothing = label_smoothing
        self.class_balanced_loss = class_balanced_loss

        if self.class_balanced_loss:
            if class_priors is None:
                raise ValueError(
                    "class_balanced_loss=True requires `class_priors` to be passed "
                    "explicitly (computed dynamically from the training split). "
                    "Hard-coded AMI priors used to live here as a fallback; that "
                    "code path was removed because it silently produced wrong "
                    "weights on any dataset other than AMI."
                )
            weights = torch.tensor(
                [1.0 / p if p > 0 else 0.0 for p in class_priors], dtype=torch.float32
            )
            weights = weights / weights.sum() * len(MODEL_NAMES)
            self.register_buffer("class_weights", weights)
        else:
            self.class_weights = None

    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.model(hidden_states, attention_masks)

    def _compute_loss(
        self,
        probs: torch.Tensor,
        wer_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute primary + auxiliary loss and evaluation metrics."""
        weighted_wer = (probs * wer_scores).sum(dim=-1)
        primary_loss = weighted_wer.mean()

        best_model_idx = wer_scores.argmin(dim=-1)
        
        # Hard CE with optional class balancing to prevent mode collapse
        hard_ce = F.cross_entropy(
            torch.log(probs + 1e-8),
            best_model_idx,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # Soft CE: target distribution is softmax(-wer / T)
        soft_target = F.softmax(-wer_scores / self.soft_ce_temperature, dim=-1)
        soft_ce = -(soft_target * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        total_loss = (
            self.primary_weight * primary_loss
            + self.aux_ce_weight * hard_ce
            + self.soft_ce_weight * soft_ce
        )

        selected_model = probs.argmax(dim=-1)
        oracle_wer = wer_scores.min(dim=-1).values
        selected_wer = wer_scores.gather(1, selected_model.unsqueeze(1)).squeeze(1)
        selection_accuracy = (selected_model == best_model_idx).float().mean()

        selection_freq = {
            MODEL_NAMES[i]: (selected_model == i).float().mean()
            for i in range(len(MODEL_NAMES))
        }

        metrics = {
            "primary_loss": primary_loss,
            "hard_ce": hard_ce,
            "soft_ce": soft_ce,
            "total_loss": total_loss,
            "selected_wer": selected_wer.mean(),
            "oracle_wer": oracle_wer.mean(),
            "wer_gap": (selected_wer - oracle_wer).mean(),
            "selection_accuracy": selection_accuracy,
            **{f"freq_{k}": v for k, v in selection_freq.items()},
        }
        return total_loss, metrics

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_scores"])

        prog_bar_keys = {"total_loss", "selected_wer", "selection_accuracy"}
        for k, v in metrics.items():
            self.log(
                f"train/{k}", v,
                on_step=True, on_epoch=True,
                prog_bar=(k in prog_bar_keys),
            )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_scores"])

        prog_bar_keys = {"total_loss", "selected_wer", "selection_accuracy"}
        for k, v in metrics.items():
            self.log(
                f"val/{k}", v,
                on_epoch=True,
                prog_bar=(k in prog_bar_keys),
            )
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_scores"])
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict:
        """AdamW with linear warmup + cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    """Train the ASR Model Selector."""
    setup_unified_logging(level="INFO")

    allow_tf32 = cfg.get("allow_tf32", True)
    if allow_tf32:
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.allow_tf32 = True
    else:
        logger.info("TF32 optimizations disabled for reproducibility.")
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pl.seed_everything(cfg.seed)

    full_dataset = ASRFeatureDataset(
        parquet_path=cfg.parquet_path,
        max_seq_len=cfg.max_seq_len,
        auto_rechunk=cfg.get("auto_rechunk", True),
        target_row_group_size=cfg.get("target_row_group_size", 256),
        cache_dir=cfg.get("parquet_cache_dir", None),
        cache_size=cfg.get("row_group_cache_size", None),
        eager_load=cfg.get("eager_load", False),
    )

    n_total = len(full_dataset)
    n_train = int(n_total * cfg.train_ratio)
    n_val = int(n_total * cfg.val_ratio)
    n_test = n_total - n_train - n_val
    from src.data.analyze_priors import analyze_dataset_priors

    logger.info("Dataset splits: train={}, val={}, test={}", n_train, n_val, n_test)

    analyze_dataset_priors(cfg.parquet_path, verbose=True)

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.max_epochs

    class_priors = None
    if cfg.class_balanced_loss:
        train_wers = full_dataset.wer_matrix[train_ds.indices]
        best_model_indices = np.argmin(train_wers, axis=1)
        counts = np.bincount(best_model_indices, minlength=len(MODEL_NAMES))
        class_priors = (counts / len(best_model_indices)).tolist()
        logger.info(
            "Computed train set class priors for balanced loss: {}",
            dict(zip(MODEL_NAMES, class_priors)),
        )

    lightning_model = ASRSelectorModule(
        arch=cfg.arch,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        stage1_layers=cfg.stage1_layers,
        stage2_layers=cfg.stage2_layers,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        use_cross_attention_bridge=not cfg.no_cross_attention,
        share_stage1_weights=not cfg.separate_stage1,
        max_seq_len=cfg.max_seq_len,
        mlp_hidden=cfg.mlp_hidden,
        mlp_layers=cfg.mlp_layers,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_steps,
        primary_weight=cfg.primary_weight,
        aux_ce_weight=cfg.aux_ce_weight,
        soft_ce_weight=cfg.soft_ce_weight,
        soft_ce_temperature=cfg.soft_ce_temperature,
        label_smoothing=cfg.label_smoothing,
        class_balanced_loss=cfg.class_balanced_loss,
        class_priors=class_priors,
    )

    param_counts = lightning_model.model.count_parameters()
    logger.info("=== Model Parameter Counts ===")
    for k, v in param_counts.items():
        logger.info("  {:>25}: {:>10}", k, v)

    is_tty = sys.stdout.isatty()
    test_average_epochs = cfg.get("test_average_epochs", 1)
    save_top_k = 3
    if isinstance(test_average_epochs, int) and test_average_epochs > 0:
        save_top_k = max(3, test_average_epochs)

    ckpt_dir = os.path.join(cfg.log_dir, cfg.experiment_name, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}",
        monitor="val/selected_wer",
        mode="min",
        save_top_k=save_top_k,
        save_last=False,
    )
    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/selected_wer",
            patience=cfg.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        EpochSummary(),
    ]
    
    if is_tty:
        callbacks.append(TQDMProgressBar(refresh_rate=cfg.progress_bar_refresh))

    if isinstance(test_average_epochs, (list, tuple)):
        callbacks.append(SaveSpecificEpochsCallback(test_average_epochs, ckpt_dir))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["git"] = git_state()
    wandb_logger = WandbLogger(
        project="s2t-tr-dev",
        name=cfg.experiment_name,
        group=cfg.get("wandb_group"),
        save_dir=os.path.join(cfg.log_dir, cfg.experiment_name),
        log_model="best",
        save_code=True,
        config=cfg_dict,
    )

    trainer_kwargs = dict(
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=cfg.gradient_clip_val,
        accelerator="auto",
        devices=1,
        precision=cfg.precision,
        log_every_n_steps=10,
        deterministic=cfg.deterministic,
        enable_progress_bar=is_tty,
    )
    if cfg.limit_batches is not None:
        trainer_kwargs.update(
            limit_train_batches=cfg.limit_batches,
            limit_val_batches=cfg.limit_batches,
            limit_test_batches=cfg.limit_batches,
        )

    trainer = pl.Trainer(**trainer_kwargs)

    logger.info("Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)

    logger.info("Running test evaluation on the best checkpoint(s)...")
    ckpts_to_test: list[Optional[str]] = []

    if isinstance(test_average_epochs, int) and test_average_epochs > 1:
        if checkpoint_callback.best_k_models:
            sorted_ckpts = sorted(checkpoint_callback.best_k_models.items(), key=lambda x: x[1])
            ckpts_to_test = [p for p, _ in sorted_ckpts[:test_average_epochs]]
            logger.info("Averaging top {} checkpoints: {}", len(ckpts_to_test), ckpts_to_test)
        else:
            ckpts_to_test = [checkpoint_callback.best_model_path]
    elif isinstance(test_average_epochs, (list, tuple)):
        for ep in test_average_epochs:
            p = os.path.join(ckpt_dir, f"epoch-{ep:02d}.ckpt")
            if os.path.exists(p):
                ckpts_to_test.append(p)
            else:
                logger.warning("Requested specific epoch checkpoint not found: {}", p)
        logger.info("Averaging specific epoch checkpoints: {}", ckpts_to_test)
    else:
        if checkpoint_callback.best_model_path:
            ckpts_to_test = [checkpoint_callback.best_model_path]
        else:
            ckpts_to_test = [None]
        logger.info("Testing on single best checkpoint: {}", ckpts_to_test)

    if not ckpts_to_test:
        logger.warning("No checkpoints found to test. Testing on current weights.")
        ckpts_to_test = [None]

    # Per-checkpoint metrics. We disable the W&B logger on the trainer for
    # the per-checkpoint passes so the W&B run summary contains only the
    # final averaged result (one row per model = clean dashboards).
    test_trainer_kwargs = dict(
        accelerator="auto", devices=1,
        precision=cfg.precision, logger=False,
        enable_progress_bar=is_tty,
    )
    if cfg.limit_batches is not None:
        test_trainer_kwargs["limit_test_batches"] = cfg.limit_batches

    trainer_for_test = pl.Trainer(**test_trainer_kwargs)
    all_metrics = []
    with legacy_torch_load():
        for ckpt in ckpts_to_test:
            logger.info("Evaluating checkpoint: {}", ckpt)
            test_metrics_list = trainer_for_test.test(
                lightning_model, test_loader, ckpt_path=ckpt
            )
            if test_metrics_list:
                all_metrics.append(test_metrics_list[0])

    if all_metrics:
        avg_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0].keys()
        }
        test_results = avg_metrics
        logger.info("Averaged Test Metrics: {}", test_results)
        # Log only the final averaged result to W&B under final_test/
        # so the run summary reflects a single, unambiguous test row.
        final_payload = {
            f"final_test/{k.replace('test/', '')}": float(v)
            for k, v in test_results.items()
        }
        try:
            wandb_logger.experiment.log(final_payload)
            wandb_logger.experiment.summary.update(final_payload)
        except Exception as exc:  # pragma: no cover - W&B may be offline
            logger.warning("Failed to log final_test metrics to W&B: {}", exc)
    else:
        test_results = {}

    results_dir = os.path.join(cfg.log_dir, cfg.experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    flat_test = {
        k.replace("test/", ""): float(v) for k, v in test_results.items()
    }
    test_json_path = os.path.join(results_dir, "test_results.json")
    with open(test_json_path, "w") as f:
        json.dump(
            {"split": "test", **flat_test, "averaged_checkpoints": ckpts_to_test},
            f, indent=2,
        )
    logger.info("Wrote test results to {}", test_json_path)

    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    config_snapshot["git"] = git_state()
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)
    logger.info("Training complete. Logs saved to {}", results_dir)


if __name__ == "__main__":
    train()