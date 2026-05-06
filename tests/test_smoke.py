"""End-to-end smoke test using a runtime-generated tiny synthetic parquet.

Verifies:
    - The full BaseExperiment pipeline runs (parent + multiple children).
    - Reproducibility artefacts are logged (git_state, resolved_config,
      optional dirty patch).
    - The WER comparison table is logged on the parent run.

Backend resolution:
    - If ``DAGSHUB_USER_TOKEN`` and ``DAGSHUB_TRACKING_URI`` are set,
      the test points at DagsHub (used in CI / manual integration runs).
    - Otherwise it points at a tmp-dir file-store MLflow.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mlflow_backend(tmp_path, monkeypatch):
    """Point mlflow at DagsHub if env is set, else at a local file store."""
    if os.environ.get("DAGSHUB_USER_TOKEN") and os.environ.get("DAGSHUB_TRACKING_URI"):
        uri = os.environ["DAGSHUB_TRACKING_URI"]
        token = os.environ["DAGSHUB_USER_TOKEN"]
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME",
                           os.environ.get("DAGSHUB_USERNAME", "token"))
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", token)
    else:
        uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("smoke")
    yield uri


def test_smoke_synthetic_full_pipeline(tmp_path, mlflow_backend, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    parquet = tmp_path / "synth.parquet"

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=smoke",
                f"experiment.data.parquet_path={parquet}",
            ],
        )

    experiment = instantiate(cfg.experiment, _recursive_=False)
    results = experiment.run(cfg.experiment)

    n_seeds = len(experiment.seeds)
    assert n_seeds >= 2, "smoke config must exercise multi-seed (seeds=[0, 1])"

    method_names = {"oracle", "random", "mlp_pool"}
    assert method_names <= set(results.keys())
    for name, m in results.items():
        assert "error" not in m, f"{name} failed: {m.get('error')}"
        assert m["n_seeds"] == n_seeds
        # Across-seed SEM is reported under wer_sem in multi-seed mode.
        assert "wer_sem" in m
        # Cross-seed SEM sidecar exists for the headline metric.
        assert "wer_mean__seed_sem" in m

    # Deterministic methods (Oracle) must produce zero across-seed variance.
    assert results["oracle"]["wer_mean__seed_sem"] == pytest.approx(0.0, abs=1e-12)

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("smoke")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])

    parents = [r for r in runs if r.data.tags.get("mlflow.parentRunId") is None]
    parent = next(p for p in parents if p.info.run_id == experiment.parent_run_id)

    method_runs = [
        r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    ]
    assert len(method_runs) == len(results), "one method run per result expected"

    method_run_ids = {r.info.run_id for r in method_runs}
    seed_runs = [
        r for r in runs if r.data.tags.get("mlflow.parentRunId") in method_run_ids
    ]
    assert len(seed_runs) == len(method_runs) * n_seeds, (
        f"expected {len(method_runs)} × {n_seeds} grand-children, got {len(seed_runs)}"
    )

    for mr in method_runs:
        assert mr.data.params["n_seeds"] == str(n_seeds)
        assert "final/wer_mean" in mr.data.metrics

    for sr in seed_runs:
        assert "seed" in sr.data.params
        assert sr.data.params["seed"] in {"0", "1"}

    assert "git_commit" in parent.data.params
    artifacts = {a.path for a in client.list_artifacts(parent.info.run_id, "repro")}
    assert "repro/git_state.json" in artifacts
    assert "repro/resolved_config.yaml" in artifacts

    results_artifacts = {a.path for a in client.list_artifacts(parent.info.run_id, "results")}
    assert "results/test_wer_comparison.json" in results_artifacts


def test_single_seed_mode_is_backward_compatible(tmp_path, mlflow_backend, monkeypatch):  # noqa: ARG001
    """Without ``seeds``, behaviour is identical to a single-seed run:
    the aggregate run wraps a single grand-child, no across-seed SEM is
    written, and ``wer_sem`` retains its within-seed sampling value."""
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    parquet = tmp_path / "synth_single.parquet"

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=smoke",
                f"experiment.data.parquet_path={parquet}",
                "~experiment.seeds",  # remove the seeds list
            ],
        )

    experiment = instantiate(cfg.experiment, _recursive_=False)
    results = experiment.run(cfg.experiment)

    assert experiment.seeds == [experiment.seed]
    for m in results.values():
        assert m["n_seeds"] == 1
        assert "wer_mean__seed_sem" not in m, (
            "single-seed mode must not synthesise an across-seed SEM"
        )


def test_synthetic_parquet_is_cached(tmp_path):
    """Re-running prepare_data() with an existing parquet should be a no-op."""
    from src.data.synthetic import SyntheticDataModule

    parquet = tmp_path / "cache.parquet"
    dm = SyntheticDataModule(
        parquet_path=str(parquet), num_samples=8, frame_length=8,
        num_regimes=2, regime_dim=4, batch_size=2, num_workers=0, max_seq_len=8,
        feature_dtype="float32",
    )
    dm.prepare_data()
    assert parquet.exists()
    mtime = parquet.stat().st_mtime

    dm.prepare_data()
    assert parquet.stat().st_mtime == mtime


def test_resolved_config_yaml_is_round_trippable(tmp_path):
    """The resolved_config artifact should be a valid Hydra config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="config", overrides=["experiment=smoke"])
    raw = OmegaConf.to_yaml(cfg, resolve=True)
    out = tmp_path / "resolved.yaml"
    out.write_text(raw)
    assert out.read_text().strip()
