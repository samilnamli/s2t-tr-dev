# tests/test_experiment_smoke.py
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

import mlflow

from src.experiments.base import BaseExperiment

DAGSHUB_TRACKING_URI = "https://dagshub.com/huseyin-karaca/s2t-tr-dev.mlflow"
DAGSHUB_USERNAME = "huseyin-karaca"


# --- Dummy model ---
class DummyModel(pl.LightningModule):
    def __init__(self, input_dim: int = 10, name: str = "dummy"):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(input_dim, 1)

    def _step(self, batch, stage: str):
        x, y = batch
        loss = nn.functional.mse_loss(self.net(x).squeeze(-1), y)
        self.log(f"{stage}/loss", loss, prog_bar=False)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")
    def configure_optimizers(self):       return torch.optim.SGD(self.parameters(), lr=1e-2)


# --- Dummy datamodule ---
class DummyDataModule(pl.LightningDataModule):
    def __init__(self, input_dim: int = 10, n: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.n = n

    def setup(self, stage=None):
        x = torch.randn(self.n, self.input_dim)
        y = x.sum(dim=1)
        self.ds = TensorDataset(x, y)

    def train_dataloader(self): return DataLoader(self.ds, batch_size=8)
    def val_dataloader(self):   return DataLoader(self.ds, batch_size=8)
    def test_dataloader(self):  return DataLoader(self.ds, batch_size=8)


# --- Concrete experiment for testing ---
class _TestableExperiment(BaseExperiment):
    def aggregate(self) -> None:
        self.aggregate_called = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """MLflow'u lokal SQLite + filesystem'e yönlendir, DagsHub'a hit etme."""
    tracking_uri = f"sqlite:///{tmp_path}/mlruns.db"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = "test_experiment"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=str(artifact_root))
    mlflow.set_experiment(exp_name)

    yield tracking_uri


@pytest.fixture
def dagshub_mlflow(monkeypatch):
    """MLflow'u gerçek DagsHub tracking server'a yönlendir.

    DAGSHUB_USER_TOKEN env var'ı set edilmemişse test skip edilir.
    """
    token = os.environ.get("DAGSHUB_USER_TOKEN")
    if not token:
        pytest.skip("DAGSHUB_USER_TOKEN not set — skipping DagsHub integration test")

    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", DAGSHUB_USERNAME)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", token)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", DAGSHUB_TRACKING_URI)
    mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)

    exp_name = "ci_smoke"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    yield DAGSHUB_TRACKING_URI


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def _make_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "parent_run_name": "smoke_v1",
        "seed": 0,
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "max_epochs": 2,
            "accelerator": "cpu",
            "devices": 1,
            "deterministic": True,
            "log_every_n_steps": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        "data": {
            "_target_": "tests.test_experiment_smoke.DummyDataModule",
            "input_dim": 10,
            "n": 64,
        },
        "child_runs": [
            {
                "_target_": "tests.test_experiment_smoke.DummyModel",
                "name": "model_a",
                "input_dim": 10,
            },
            {
                "_target_": "tests.test_experiment_smoke.DummyModel",
                "name": "model_b",
                "input_dim": 10,
            },
        ],
    })


# ---------------------------------------------------------------------------
# Tests — local backend
# ---------------------------------------------------------------------------

def test_experiment_runs_end_to_end(local_mlflow):
    cfg = _make_cfg()
    exp = _TestableExperiment(cfg)
    results = exp.run()

    # 1. Her child run sonuç döndürdü mü?
    assert set(results.keys()) == {"model_a", "model_b"}
    for name, res in results.items():
        assert "error" not in res, f"{name} failed: {res.get('error')}"
        assert res["best_epoch"] is not None
        assert res["best_val_score"] is not None
        assert "test/loss" in res["test_metrics"]
        assert res["mlflow_run_id"]

    # 2. Aggregate çağrıldı mı?
    assert getattr(exp, "aggregate_called", False)

    # 3. MLflow'da parent + 2 child run yazıldı mı, nesting doğru mu?
    client = mlflow.tracking.MlflowClient()
    exp_obj = client.get_experiment_by_name("test_experiment")
    runs = client.search_runs([exp_obj.experiment_id])
    parent_runs = [r for r in runs if r.data.tags.get("mlflow.parentRunId") is None]
    child_runs = [r for r in runs if r.data.tags.get("mlflow.parentRunId") is not None]
    assert len(parent_runs) == 1
    assert len(child_runs) == 2
    assert all(r.data.tags["mlflow.parentRunId"] == parent_runs[0].info.run_id
               for r in child_runs)

    # 4. Summary metric'ler child run'larda mı?
    for cr in child_runs:
        assert "best_epoch" in cr.data.metrics
        assert "final/test/loss" in cr.data.metrics


def test_failing_child_does_not_kill_others(local_mlflow):
    """Bir child instantiate'te patlasa bile diğeri tamamlanmalı."""
    cfg = _make_cfg()
    cfg.child_runs[0]["_target_"] = "nonexistent.Module"

    exp = _TestableExperiment(cfg)
    results = exp.run()

    assert "error" in results["model_a"]
    assert "error" not in results["model_b"]
    assert results["model_b"]["best_epoch"] is not None


# ---------------------------------------------------------------------------
# Tests — DagsHub backend
# ---------------------------------------------------------------------------

@pytest.mark.dagshub
@pytest.mark.usefixtures("dagshub_mlflow")
def test_experiment_runs_end_to_end_on_dagshub():
    """Aynı e2e testi gerçek DagsHub tracking server'a karşı çalıştırır.

    Çalıştırmak için: DAGSHUB_USER_TOKEN=<token> uv run pytest -m dagshub
    """
    cfg = _make_cfg()
    exp = _TestableExperiment(cfg)
    results = exp.run()

    assert set(results.keys()) == {"model_a", "model_b"}
    for name, res in results.items():
        assert "error" not in res, f"{name} failed: {res.get('error')}"
        assert res["best_epoch"] is not None
        assert res["best_val_score"] is not None
        assert "test/loss" in res["test_metrics"]
        assert res["mlflow_run_id"]

    assert getattr(exp, "aggregate_called", False)

    client = mlflow.tracking.MlflowClient()
    exp_obj = client.get_experiment_by_name("ci_smoke")
    runs = client.search_runs([exp_obj.experiment_id])

    our_run_ids = {res["mlflow_run_id"] for res in results.values()}
    our_children = [r for r in runs
                    if r.info.run_id in our_run_ids]
    assert len(our_children) == 2

    our_parent_id = our_children[0].data.tags["mlflow.parentRunId"]
    assert all(r.data.tags["mlflow.parentRunId"] == our_parent_id
               for r in our_children)

    for cr in our_children:
        assert "best_epoch" in cr.data.metrics
        assert "final/test/loss" in cr.data.metrics


@pytest.mark.dagshub
@pytest.mark.usefixtures("dagshub_mlflow")
def test_failing_child_does_not_kill_others_on_dagshub():
    """Hata toleransı testini DagsHub'a karşı da doğrular."""
    cfg = _make_cfg()
    cfg.child_runs[0]["_target_"] = "nonexistent.Module"

    exp = _TestableExperiment(cfg)
    results = exp.run()

    assert "error" in results["model_a"]
    assert "error" not in results["model_b"]
    assert results["model_b"]["best_epoch"] is not None
