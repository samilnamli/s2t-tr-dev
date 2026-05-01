"""Sweep the synthetic regime-switch experiment over the number of regimes.

This is the same pipeline that used to live in ``synthetic_sweep.py``,
refactored to share helpers with :mod:`src.experiments.run` via
:mod:`src.experiments._common` and to use the unified loguru-frontend
logger.

Two subcommands:

* ``run``   — for each ``R`` value in ``--r-values``, generate the
  synthetic parquet, train both the proposed hierarchical router and
  the MLP-pool baseline, and compute training-free baselines. Per-R
  outputs accumulate in ``sweep_results.json``.

* ``figure`` — render the manuscript figure (test-WER vs. R) from
  ``sweep_results.json``.

Usage:
    uv run python -m src.experiments.sweep run \
        --r-values 2,3,4,6,8 \
        --output-dir reports/sweeps/synthetic_R \
        --num-samples 5000 --frame-length 128 \
        --max-epochs 30 --batch-size 32 --seed 42

    uv run python -m src.experiments.sweep figure \
        --results reports/sweeps/synthetic_R/sweep_results.json \
        --output-path reports/manuscript/figures/synthetic_sweep.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
import matplotlib.pyplot as plt
import typer

from src.experiments._common import python_module, run
from src.utils.logging import setup_unified_logging

app = typer.Typer(help="Sweep the synthetic regime-switch experiment over R.")


@dataclass
class _RunPaths:
    """Filesystem layout for one (R, architecture) sweep cell."""

    parquet: Path
    log_dir: Path
    test_json: Path = field(init=False)

    def __post_init__(self) -> None:
        self.test_json = self.log_dir / "test_results.json"


def _generate_dataset(
    parquet_path: Path,
    num_samples: int,
    frame_length: int,
    num_regimes: int,
    regime_dim: int,
    noise_std: float,
    feature_dtype: str,
    write_batch_size: int,
    seed: int,
) -> None:
    """Generate one synthetic parquet via :mod:`src.data.synthetic`."""
    cmd = python_module(
        "src.data.synthetic",
        "--output-path",
        str(parquet_path),
        "--num-samples",
        str(num_samples),
        "--frame-length",
        str(frame_length),
        "--num-regimes",
        str(num_regimes),
        "--regime-dim",
        str(regime_dim),
        "--noise-std",
        str(noise_std),
        "--feature-dtype",
        feature_dtype,
        "--write-batch-size",
        str(write_batch_size),
        "--seed",
        str(seed),
    )
    run(cmd, f"generate synthetic parquet (R={num_regimes})")


def _train_router(
    parquet: Path,
    log_dir: Path,
    arch: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    learning_rate: float,
    primary_weight: float,
    aux_ce_weight: float,
    soft_ce_weight: float,
    soft_ce_temperature: float,
    seed: int,
) -> None:
    """Train one router via the Hydra-driven :mod:`src.training.train`."""
    overrides = {
        "parquet_path": str(parquet),
        "arch": arch,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "primary_weight": primary_weight,
        "aux_ce_weight": aux_ce_weight,
        "soft_ce_weight": soft_ce_weight,
        "soft_ce_temperature": soft_ce_temperature,
        "seed": seed,
        "experiment_name": log_dir.name,
        "log_dir": str(log_dir.parent),
    }
    cmd = python_module("src.training.train", *[f"{k}={v}" for k, v in overrides.items()])
    run(cmd, f"train router (arch={arch}, log_dir={log_dir})")


def _evaluate_baselines(
    parquet: Path,
    save_json: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    """Compute training-free baselines via :mod:`src.training.eval_baselines`."""
    cmd = python_module(
        "src.training.eval_baselines",
        "--parquet-path",
        str(parquet),
        "--train-ratio",
        str(train_ratio),
        "--val-ratio",
        str(val_ratio),
        "--seed",
        str(seed),
        "--split",
        "test",
        "--save-json",
        str(save_json),
    )
    run(cmd, f"eval_baselines on {parquet}")


@app.command("run")
def run_sweep(
    output_dir: str = typer.Option(..., "--output-dir", "-o"),
    r_values: str = typer.Option("2,3,4,6,8", "--r-values"),
    num_samples: int = typer.Option(5000, "--num-samples"),
    frame_length: int = typer.Option(128, "--frame-length"),
    regime_dim: int = typer.Option(32, "--regime-dim"),
    noise_std: float = typer.Option(0.5, "--noise-std"),
    feature_dtype: str = typer.Option("float16", "--feature-dtype"),
    write_batch_size: int = typer.Option(500, "--write-batch-size"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    max_seq_len: int = typer.Option(256, "--max-seq-len"),
    batch_size: int = typer.Option(32, "--batch-size"),
    num_workers: int = typer.Option(2, "--num-workers"),
    max_epochs: int = typer.Option(30, "--max-epochs"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate"),
    primary_weight: float = typer.Option(1.0, "--primary-weight"),
    aux_ce_weight: float = typer.Option(0.0, "--aux-ce-weight"),
    soft_ce_weight: float = typer.Option(0.5, "--soft-ce-weight"),
    soft_ce_temperature: float = typer.Option(0.1, "--soft-ce-temperature"),
    seed: int = typer.Option(42, "--seed"),
    keep_parquets: bool = typer.Option(False, "--keep-parquets/--no-keep-parquets"),
):
    """Generate, train, and evaluate the full pipeline for each R value."""
    setup_unified_logging(level="INFO")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r_list = [int(x) for x in r_values.split(",") if x.strip()]
    logger.info("Sweep R values: {} — output dir: {}", r_list, out_dir)

    sweep_results: Dict = {
        "config": {
            "r_values": r_list,
            "num_samples": num_samples,
            "frame_length": frame_length,
            "regime_dim": regime_dim,
            "noise_std": noise_std,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "primary_weight": primary_weight,
            "aux_ce_weight": aux_ce_weight,
            "soft_ce_weight": soft_ce_weight,
            "soft_ce_temperature": soft_ce_temperature,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "seed": seed,
        },
        "by_r": {},
    }

    for R in r_list:
        cell_dir = out_dir / f"R{R}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        parquet = cell_dir / "combined_features.parquet"
        hier_paths = _RunPaths(parquet=parquet, log_dir=cell_dir / "hier")
        mlp_paths = _RunPaths(parquet=parquet, log_dir=cell_dir / "mlp_pool")
        baselines_json = cell_dir / "baselines.json"

        _generate_dataset(
            parquet_path=parquet,
            num_samples=num_samples,
            frame_length=frame_length,
            num_regimes=R,
            regime_dim=regime_dim,
            noise_std=noise_std,
            feature_dtype=feature_dtype,
            write_batch_size=write_batch_size,
            seed=seed,
        )
        for paths, arch in [
            (hier_paths, "hierarchical_transformer"),
            (mlp_paths, "mlp_pool"),
        ]:
            _train_router(
                parquet=parquet,
                log_dir=paths.log_dir,
                arch=arch,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                num_workers=num_workers,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                primary_weight=primary_weight,
                aux_ce_weight=aux_ce_weight,
                soft_ce_weight=soft_ce_weight,
                soft_ce_temperature=soft_ce_temperature,
                seed=seed,
            )
        _evaluate_baselines(
            parquet=parquet,
            save_json=baselines_json,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        sweep_results["by_r"][str(R)] = {
            "hier_test": json.loads(hier_paths.test_json.read_text()),
            "mlp_pool_test": json.loads(mlp_paths.test_json.read_text()),
            "baselines": json.loads(baselines_json.read_text()),
        }
        with open(out_dir / "sweep_results.json", "w") as f:
            json.dump(sweep_results, f, indent=2)
        logger.info("Sweep cell R={} complete.", R)

        if not keep_parquets and parquet.exists():
            parquet.unlink()
            logger.info(
                "Removed parquet {} to save disk; pass --keep-parquets to retain.", parquet
            )

    logger.info("All sweep cells complete; results at {}", out_dir / "sweep_results.json")


def _series_for_baseline(by_r: Dict, baseline_name: str) -> List[Optional[float]]:
    """Pull the test-split mean WER of a training-free baseline at each R."""
    out: List[Optional[float]] = []
    for _, entry in sorted(by_r.items(), key=lambda kv: int(kv[0])):
        results = entry["baselines"]["results"]["test"]
        out.append(float(results[baseline_name]["wer_mean"]) if baseline_name in results else None)
    return out


def _series_for_router(by_r: Dict, key: str) -> List[Optional[float]]:
    """Pull a metric (e.g. ``selected_wer``) from each R's evaluate JSON."""
    out: List[Optional[float]] = []
    for _, entry in sorted(by_r.items(), key=lambda kv: int(kv[0])):
        v = entry.get(key, {}).get("selected_wer")
        out.append(float(v) if v is not None else None)
    return out


@app.command("figure")
def figure(
    results: str = typer.Option(..., "--results", "-r"),
    output_path: str = typer.Option(..., "--output-path", "-o"),
):
    """Render the manuscript sweep figure from a sweep_results.json file."""
    setup_unified_logging(level="INFO")

    data = json.loads(Path(results).read_text())
    by_r = data["by_r"]
    r_values = sorted(int(R) for R in by_r.keys())

    series = {
        "Random": _series_for_baseline(by_r, "random"),
        "Weighted random": _series_for_baseline(by_r, "weighted_random"),
        "MLP-pool": _series_for_router(by_r, "mlp_pool_test"),
        "Proposed": _series_for_router(by_r, "hier_test"),
        "Oracle": _series_for_baseline(by_r, "oracle"),
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style = {
        "Random": dict(color="#888888", linestyle=":", marker="o"),
        "Weighted random": dict(color="#555555", linestyle="--", marker="s"),
        "MLP-pool": dict(color="#d62728", linestyle="-", marker="^"),
        "Proposed": dict(color="#1f77b4", linestyle="-", marker="D", linewidth=2.0),
        "Oracle": dict(color="#2ca02c", linestyle="-.", marker="x"),
    }
    for name, ys in series.items():
        ax.plot(r_values, ys, label=name, markersize=6, **style.get(name, {}))
    ax.set_xlabel("Number of regimes $R$")
    ax.set_ylabel("Test WER")
    ax.set_xticks(r_values)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True)
    ax.set_title("Synthetic regime-switch sweep — test WER vs. number of regimes")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if out.suffix.lower() == ".pdf":
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote sweep figure to {}", out)


if __name__ == "__main__":
    app()
