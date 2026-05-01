"""Compare hierarchical_transformer vs mlp_pool W&B runs by selected_wer.

Pulls all matching runs from a W&B project, groups by ``seed`` and
``arch``, and prints a sorted table of per-seed differences. Useful as
a quick post-hoc sanity check after a multi-seed sweep, before the
formal aggregation done by :mod:`src.reporting.tables`.

Usage:
    uv run python -m src.scripts.wandb_compare \\
        --entity my-team --project s2t-tr-dev \\
        --seeds 42 2 123
"""

from collections import defaultdict
import logging
from typing import List, Optional

import typer
import wandb

logger = logging.getLogger(__name__)
app = typer.Typer(help="Compare hierarchical_transformer vs mlp_pool W&B runs by selected_wer.")


@app.command()
def compare(
    entity: str = typer.Option(..., "--entity", "-e", help="W&B entity (user or team)."),
    project: str = typer.Option(..., "--project", "-p", help="W&B project name."),
    seeds: List[int] = typer.Option(
        ..., "--seeds", "-s", help="Seeds to compare (repeat for multiple)."
    ),
    metric: str = typer.Option(
        "test/selected_wer", "--metric", help="W&B summary metric to compare."
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="WANDB_API_KEY",
        help="Optional W&B API key (else read from env / ~/.netrc).",
    ),
    seed_key: str = typer.Option("seed", "--seed-key", help="Run config key holding the seed."),
    arch_key: str = typer.Option(
        "arch", "--arch-key", help="Run config key holding the architecture."
    ),
):
    """Print a per-seed (HT vs MLP-pool) comparison sorted by HT-MLP delta."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if api_key:
        wandb.login(key=api_key)
    api = wandb.Api()

    logger.info("Fetching runs from %s/%s ...", entity, project)
    all_runs = api.runs(f"{entity}/{project}")

    target_seeds = set(seeds)
    target_archs = {"hierarchical_transformer", "mlp_pool"}

    data: dict[int, dict[str, float]] = defaultdict(dict)
    for run in all_runs:
        cfg = run.config
        seed = cfg.get(seed_key)
        arch = cfg.get(arch_key)
        if seed not in target_seeds or arch not in target_archs:
            continue
        metric_val = run.summary.get(metric)
        if metric_val is None:
            logger.warning(
                "Run %s (seed=%s, arch=%s) has no %r — skipping", run.id, seed, arch, metric
            )
            continue
        if arch in data[seed]:
            logger.warning(
                "Duplicate run for seed=%s, arch=%s — keeping latest logged", seed, arch
            )
        data[seed][arch] = metric_val
        logger.info("  seed=%-6d arch=%-30s %s=%.4f", seed, arch, metric, metric_val)

    rows = []
    for seed in seeds:
        archs = data.get(seed, {})
        ht = archs.get("hierarchical_transformer")
        mlp = archs.get("mlp_pool")
        if ht is None or mlp is None:
            missing = [
                a for a, v in (("hierarchical_transformer", ht), ("mlp_pool", mlp)) if v is None
            ]
            logger.info("[SKIP] seed=%s — missing: %s", seed, ", ".join(missing))
            continue
        rows.append((seed, ht, mlp, ht - mlp))

    rows.sort(key=lambda r: r[3], reverse=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {'Seed':<8} {'HT':>10} {'MLP':>10} {'Diff (HT-MLP)':>15}")
    print(sep)
    for seed, ht, mlp, diff in rows:
        sign = "+" if diff > 0 else ""
        print(f"  {seed:<8} {ht:>10.4f} {mlp:>10.4f} {sign}{diff:>14.4f}")
    print(sep)
    print("Positive diff => hierarchical_transformer WER is HIGHER (worse).")
    print("Negative diff => hierarchical_transformer WER is LOWER  (better).\n")


if __name__ == "__main__":
    app()
