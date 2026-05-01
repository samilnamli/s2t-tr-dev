# s2t-tr-dev

**Hierarchical Transformer Routing over Frame-Level Encoder Embeddings for Adaptive ASR**

A.~S.~Namli$^{*}$, H.~Karaca$^{*}$, S.~S.~Kozat — Bilkent University / Türk Telekom.
*Manuscript:* [`reports/manuscript/main.pdf`](reports/manuscript/main.pdf)
(working draft; numbers under active reproduction).

---

## What this repo does

We treat **per-clip ASR model selection** as a learning problem. Given
a clip and the cached frame-level encoder hidden states from $K$
pretrained ASR experts (HuBERT-Large, Whisper-base,
Wav2Vec2-Large-Robust), a small **hierarchical transformer router** (~5M
params) emits a probability distribution over the $K$ experts. We
decode only the top-1 expert at inference, so per-clip cost matches
single-model inference while we get the diversity of the expert pool.

Trained with a composite loss:

$$
\mathcal{L}(\theta) = \lambda_{\text{wer}} \mathcal{L}_{\text{wer}} +
\lambda_{\text{hard}} \mathcal{L}_{\text{hard}} +
\lambda_{\text{soft}} \mathcal{L}_{\text{soft}}
$$

(weighted-WER + hard CE on oracle expert + soft CE against a
$\tau$-tempered WER target — see `reports/manuscript/main.tex` §III).

---

## Deliverables (current status)

| # | Experiment | Config | Notebook | Status |
| --- | --- | --- | --- | --- |
| 1 | Synthetic regime-switch | `configs/experiments/synthetic.yaml` | `notebooks/colab/s2t_tr_dev_synthetic_experiment.ipynb` | Smoke only — manuscript-grade run pending |
| 2 | Loss ablation (AMI) | `configs/experiments/ablation_loss.yaml` | `notebooks/colab/s2t_tr_dev_ablation_loss.ipynb` | Pending |
| 3 | Architecture ablation (AMI) | `configs/experiments/ablation_architecture.yaml` | `notebooks/colab/s2t_tr_dev_ablation_architecture.ipynb` | Pending (5 manuscript cells missing) |
| 4 | Main results — AMI | `configs/experiments/main_results_ami.yaml` | `notebooks/colab/s2t_tr_dev_main_results_ami.ipynb` | Trained, awaits cleaned re-run |
| — | Main results — VoxPopuli | `configs/experiments/main_results_voxpopuli*.yaml` | `notebooks/colab/s2t_tr_dev_main_results_voxpopuli.ipynb` | **Parked.** Router collapses to HuBERT under all 4 attempted variants. See `PROJECT_STATE.md §5`. |

The shared roadmap, blockers, and operating principles live in
[`PROJECT_STATE.md`](PROJECT_STATE.md). Read it before proposing changes.

---

## Quickstart

### Local

```bash
# 1. Environment (Python 3.10, all deps via uv)
make create_environment
make requirements                    # uv sync

# 2. Data
make download_ami                    # processed AMI parquet from Drive
make download_voxpopuli              # (parked dataset, fetches anyway)

# 3. One-line experiment run (Hydra-overridable)
uv run python -m src.experiments.run experiments=ablation_loss

# 4. Lint / format
make lint
make format
```

### Colab (the canonical reproducibility path)

Each notebook in `notebooks/colab/` is a strict reproducibility package:

1. `git clone` + `make create_environment` + `uv sync`.
2. `userdata.get("WANDB_API_KEY")`, `userdata.get("HF_TOKEN")` from Colab Secrets.
3. `!make download_<dataset>`.
4. `!uv run python -m src.experiments.run experiments=<name>`.
5. Final cell: `!uv run python -m src.reporting.tables ...` →
   the paper-deliverable table/figure for that experiment, also pushed
   to W&B as an artifact.

**No raw inline Python** is allowed in any notebook cell.

---

## Repo layout

```
configs/                   # Hydra YAML — SSOT (see configs/README.md)
src/
  data/                    # parquet dataset, parquet cache, synthetic generator
  models/                  # selector (hier transformer) + mlp_pool baseline
  training/                # train.py, eval_baselines.py, rover.py, visualize.py
  experiments/             # generic Hydra-driven runners
  reporting/               # main_results.json -> Markdown / LaTeX tables, figures
  scripts/                 # ad-hoc analysis (wandb_compare, push_to_hub, ...)
  utils/                   # logging, ckpt-load helpers
notebooks/colab/           # one Colab notebook per deliverable
reports/manuscript/        # main.tex + figures (auto-tables under figures/auto/)
data/{raw,interim,processed}/   # gitignored; fetched via make download_*
PROJECT_STATE.md           # living roadmap + decision log
```

---

## Stack

- **Compute:** PyTorch 2.11 + Lightning 2.6, mixed-precision (bf16 on A100/L4).
- **Data:** Apache Arrow / parquet via DuckDB + pyarrow, with row-group
  rechunking for low-RAM Colab.
- **Tracking:** W&B as the anchor (git commit, code, config, best
  checkpoint, paper-deliverable artifacts).
- **Env:** `uv` only. Lockfile at `uv.lock`.
- **Logging:** Loguru frontend, stdlib `logging` and 3rd-party loggers
  intercepted into a single sink.
- **Config:** Hydra. See `configs/README.md` for the mandatory
  `experiment_metadata` schema and the zero-mutation rule.

---

## Citing

If this work helps you, please cite the manuscript (preprint coming).
The companion website, HuggingFace Hub model cards, and a GHCR Docker
image are tracked in `PROJECT_STATE.md` §4 (Long-Term).
