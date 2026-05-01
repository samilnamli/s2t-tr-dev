# PROJECT_STATE.md — Living Roadmap

> **Read me first, every time, before proposing any work.**
>
> This file is the single shared context for every human and AI agent that
> touches this repo. Keep it short, blunt, and current. Append-only history
> at the bottom; the top three sections are mutable and reflect *now*.

Last updated: 2026-05-01 (Friday)

---

## 1. Project at a glance

**Title (manuscript working):** Hierarchical Transformer Routing over Frame-Level Encoder Embeddings for Adaptive ASR

**One-liner:** Train a small hierarchical transformer to pick the best
pretrained ASR system (HuBERT-Large / Whisper-base / Wav2Vec2-Large-Robust)
per audio clip, using cached frame-level encoder features and a
weighted-WER + soft-CE loss.

**Manuscript:** `reports/manuscript/main.tex` (last build 2026-04-26).
The numbers in Tables I–IV of `main.tex` are **placeholders** — they will
be replaced from the cleaned re-runs of the four active deliverables
listed in §3.

---

## 2. Operating principles (binding)

1. **SSOT.** YAML configs in `configs/experiments/` are the source of truth.
   Python files in `src/` are static, modular runners that consume them.
2. **Zero mutation.** Never edit a past experiment config in place. If
   `<name>_v1.yaml` needs new params, create `<name>_v2.yaml` extending it
   and explain *why* in `experiment_metadata`.
3. **Explicit rationale.** Every new config carries an `experiment_metadata`
   block: `parent`, `interpretation`, `rationale`, `hypothesis`, `status`.
   Schema is documented in `configs/README.md`.
4. **Modularity.** The difference between two experiments must be a config
   swap. If you need a new Python branch, lift the variation into a config
   field instead.
5. **uv only.** All commands use `uv run` / `uv sync`. No `pip install`,
   no global Python.
6. **Notebook = reproducibility package, not a scratchpad.** Cells contain
   only `! ...` shell calls (`!uv run python -m ...`, `!make ...`),
   `from google.colab import userdata` for secrets, and one final markdown
   cell embedding the paper-deliverable rendered from `src/`. No raw
   inline Python in any cell.
7. **Container assumption.** Code must run inside a fresh Docker container
   on vast.ai with only `uv sync`, the cloned repo, and `WANDB_API_KEY` /
   `HF_TOKEN` available. Use relative paths and standard dirs.
8. **W&B is the anchor.** Every reproducible state is a W&B run with the
   git commit captured, the YAML config logged, and the final test result
   logged from the **single best checkpoint only**. Paper-deliverables
   (figures and tables) are pushed as W&B artifacts.
9. **Approval gate.** Never start implementing or running without explicit
   user approval of a granular sub-checklist for the proposed batch.

---

## 3. Deliverables — current status

We have **four active experiments** for the manuscript. VoxPopuli is
parked (see §5).

| # | Experiment | Config (canonical) | Manuscript ref | Status |
| --- | --- | --- | --- | --- |
| 1 | Synthetic regime-switch | `configs/experiments/synthetic.yaml` (will get `_v2` for manuscript-grade run) | Tab. I `tab:synthetic`, Fig. `fig:synthetic_sweep` | Smoke run only (2 epochs); needs full run + sweep |
| 2 | Loss ablation (AMI) | `configs/experiments/ablation_loss.yaml` (will get `_v2` to fix `τ` mismatch with manuscript) | Tab. III `tab:loss_ablation` | Not yet executed cleanly |
| 3 | Architecture ablation (AMI) | `configs/experiments/ablation_architecture.yaml` (will get `_v2` to add 5 missing manuscript cells) | Tab. IV `tab:arch_ablation` | Not yet executed cleanly |
| 4 | Main results — AMI | `configs/experiments/main_results_ami.yaml` (3 seeds × 2 methods) | Tab. II `tab:main_results` (AMI column) | Trained once; will be re-run on cleaned pipeline |

### Deferred / parked

| # | Experiment | Config | Reason |
| --- | --- | --- | --- |
| 5 | Main results — VoxPopuli | `main_results_voxpopuli{,_v2,_v3,_v4}.yaml` | Router never learns to pick Whisper despite 4 attempts; soft-CE penalises Whisper hallucinations too harshly. Detailed write-up in v2 `_comment_:`. Parked, not deleted, so lineage is preserved. |

---

## 4. Roadmap

### Short-Term (this sprint, blocking the manuscript)

- [ ] **Phase 0** — Foundations: this file + `configs/README.md` schema doc. *(in progress)*
- [ ] **Phase 1.A** — Repo hygiene (cookiecutter README, dead files, untrack
  cache/build artifacts, kill `outputs/` dir fragmentation). *(in progress)*
- [ ] **Phase 1.B** — Consolidate `src/experiments/` into one Hydra-driven
  generic runner (`run.py`) + shared `_common.py`. Delete dead Typer/JSON
  `run_ablation.py`. Refactor `synthetic_sweep.py` to share the same helpers.
- [ ] **Phase 1.C** — Hydra `chdir=false` + fixed `hydra.run.dir`. Unified
  loguru-frontend logging (intercept stdlib + 3rd-party). W&B `save_code=True`,
  `log_model="best"`, explicit git-commit capture. Remove the dead AMI-priors
  fallback in `train.py`. Replace global `torch.load` monkey-patch with a
  context-managed helper.
- [ ] **Phase 1.D** — Reconcile each YAML against the manuscript. Create
  `_v2` configs with the corrected/extended sweep cells per §3. Mark
  VoxPopuli configs deprecated. Generic `make run_experiment EXPERIMENT=…`.
- [ ] **Phase 1.E** — Generic `src/reporting/{tables,figures}.py` that
  consume `main_results.json` and emit Markdown + LaTeX tables and matplotlib
  figures. Used both by the notebooks (final cell) and by the manuscript
  (`\input{auto/...}`).
- [ ] **Phase 1.F** — Regenerate the four Colab notebooks fresh, output-cleared,
  protocol-compliant (`!`-only cells + final `src.reporting` cell + W&B artifact).

### Medium-Term (after Phase 1 lands)

- [ ] Execute the four active deliverables on Colab/vast.ai under the cleaned
  pipeline.
- [ ] Auto-replace placeholder numbers in `main.tex` Tables I–IV via the
  `\input{auto/...}` mechanism.
- [ ] Co-write abstract, conclusion, refreshed discussion based on real numbers.

### Long-Term (post-submission)

- [ ] HuggingFace Hub push helper (`src/scripts/push_to_hub.py`) + auto model card.
- [ ] Data registry (`data/REGISTRY.md`) with Drive IDs + checksums.
- [ ] MkDocs companion site (`docs/`) — abstract, reproducibility steps, per-experiment pages.
- [ ] Multi-stage Dockerfile + GitHub Actions workflow → GHCR (`ghcr.io/huseyin-karaca/s2t-tr-dev:vX.Y.Z`).
- [ ] CI: `ruff format --check`, `ruff check`, `pytest` (synthetic 2-epoch smoke), notebook-execution smoke on PR.

---

## 5. Active blockers / known issues

- **VoxPopuli routing failure (parked).** Across four config versions
  (`v1`–`v4`), the router collapses to picking HuBERT 100% of the time
  despite Whisper being oracle on ~46% of clips. Hypothesis (per the v2
  Turkish rationale): Whisper's outlier WERs (~120% on hallucinated clips)
  generate gradient signal so harsh under soft-CE that the model learns
  to never pick Whisper. v3 tested extreme risk-taking (`primary_weight=0.1`,
  `soft_ce_temperature=0.05`); v4 tested dynamic class-balancing. None
  worked. Next attempt would likely require a different base model
  (e.g. Whisper-large) or per-clip outlier clipping in the loss; out of
  scope for the current submission.
- **Manuscript ↔ config mismatches** documented in audit (2026-05-01):
  - Loss ablation reports `τ=0.1` in `main.tex`, but `ablation_loss.yaml` has `τ=1.5`.
  - Architecture ablation table has 8 cells in `main.tex` but only 4 in `ablation_architecture.yaml` (depth/width sweep cells missing).
  - Synthetic table values are placeholder integers (`49.6`, `32.4`, `21.7`, `18.1`).

---

## 6. Repo map (after Phase 1 lands; current state in parens)

```
configs/
  config.yaml                # global defaults
  README.md                  # (NEW) experiment_metadata schema
  experiments/
    synthetic_v2.yaml
    ablation_loss_v2.yaml
    ablation_architecture_v2.yaml
    main_results_ami.yaml
    deprecated/              # (parked) main_results_voxpopuli{,_v2,_v3,_v4}.yaml
src/
  data/                      # ASRFeatureDataset, parquet_cache, synthetic generator
  models/                    # selector (hier transformer) + mlp_pool baseline
  training/                  # train.py, eval_baselines.py, rover.py, evaluate.py, visualize.py
  experiments/
    run.py                   # (NEW, was main_results.py) generic Hydra runner
    sweep.py                 # (NEW, was synthetic_sweep.py) generic sweeps
    _common.py               # (NEW) shared subprocess helpers
  reporting/                 # (NEW)
    tables.py                # main_results.json -> {.md, .tex}
    figures.py               # main_results.json -> {.pdf, .png}
  scripts/
    wandb_compare.py         # (MOVED from repo root)
  utils/
    logging.py               # (NEW) loguru-frontend setup
notebooks/colab/             # (regenerated fresh, output-cleared)
reports/manuscript/          # main.tex (auto-tables under figures/auto/)
```

---

## 7. Decision log (append-only)

- **2026-05-01** Initial audit; roadmap approved by user.
  - VoxPopuli deferred indefinitely (router collapses to HuBERT under all four config versions).
  - Manuscript values trusted as placeholders; numbers will be regenerated.
  - Notebooks regenerated fresh, not patched in place.
  - Loguru chosen as unified-logging frontend.
  - Hydra: `job.chdir=false` + `hydra.run.dir=${log_dir}/.hydra/${experiment_name}` (option B).
