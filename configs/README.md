# `configs/` — Hydra config single source of truth

> Python in `src/` is a static, modular runner.
> **YAML in this folder is the SSOT.** All experimental knobs live here.

---

## 1. Layout

```
configs/
  config.yaml             # global defaults (model dims, training defaults, paths)
  README.md               # this file
  experiments/
    <name>.yaml           # one file per experiment
    <name>_v2.yaml        # successor (never edit <name>.yaml in place)
    deprecated/<name>.yaml  # parked configs kept for lineage
```

`config.yaml` is loaded for every run. An experiment file is selected via
the Hydra override:

```bash
uv run python -m src.experiments.run experiments=ablation_loss_v2
```

The chosen experiment YAML is mounted under `cfg.experiments` and overrides
matching keys in `config.yaml` for that run only.

---

## 2. Mandatory schema for `configs/experiments/<name>.yaml`

Every new experiment config MUST include:

```yaml
experiment_metadata:
  name: ablation_loss_v2
  parent: ablation_loss          # null if this is the first version
  status: active                 # one of: draft | active | deprecated | failed
  created: 2026-05-01
  author: huseyin-karaca

  interpretation: |
    Plain-language summary of what we learned from the parent run.
    What numbers came back, what surprised us, what we believed we
    understood. Kept short (3-6 sentences).

  rationale: |
    Mathematical / logical reason for the parameter changes in this
    file relative to the parent. Cite equations from the manuscript
    where helpful (e.g. "lowering tau in eq. (12) sharpens the soft
    target so the gradient w.r.t. the oracle expert dominates").

  hypothesis: |
    What we expect to observe if the new params are correct, and what
    counter-evidence would falsify the hypothesis. Numeric thresholds
    where possible (e.g. "test WER on AMI should drop below 0.32; if
    it stays >0.34 the hypothesis is wrong").

  deprecated_for: null           # set to "<successor_name>" when superseded

parquet_path: data/processed/<dataset>/combined_features_with_transcripts.parquet
skip_rover: false                # true for synthetic where ROVER is meaningless
skip_training_free: false

shared:                          # applied to every method below
  train_ratio: 0.8
  val_ratio: 0.1
  max_seq_len: 2000
  batch_size: 128
  num_workers: 4
  max_epochs: 50
  learning_rate: 1.0e-4
  precision: bf16-mixed
  eager_load: true
  allow_tf32: false
  early_stopping_patience: 8
  test_average_epochs: 1

methods:                         # one entry per training run
  - name: hierarchical_proposed
    arch: hierarchical_transformer
    primary_weight: 1.0
    aux_ce_weight: 0.3
    soft_ce_weight: 0.5
    soft_ce_temperature: 0.1
    class_balanced_loss: true
    seed: [42, 2, 123]           # list -> auto-expanded to multiple runs
```

The runner (`src/experiments/run.py`) automatically expands a list-valued
`seed` into one independent run per seed, with `name` suffixed
`-seed<n>`. All other fields may be scalars or arrays per Hydra's rules.

---

## 3. Zero-mutation rule

**Never edit a config that has been used to train a logged W&B run.**
If you need new parameters:

1. `cp configs/experiments/<name>.yaml configs/experiments/<name>_v2.yaml`
2. Edit only `experiment_metadata` and the changed fields in `_v2`.
3. Set `experiment_metadata.parent: <name>` and write `interpretation`,
   `rationale`, `hypothesis`.
4. In the parent file, set `experiment_metadata.deprecated_for: <name>_v2`.
5. Commit both. The W&B run history thus has a stable, machine-readable
   lineage.

When a config is fully retired (e.g. VoxPopuli configs that never
converged), move it to `configs/experiments/deprecated/` rather than
deleting it.

---

## 4. Reading a config from code

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # global defaults are at cfg.<key>
    # experiment-specific block is at cfg.experiments.<key>
    pipeline = cfg.experiments
    metadata = pipeline.experiment_metadata
    methods = pipeline.methods
    ...
```

The runner reads `cfg.experiments.experiment_metadata` and logs the
entire block to W&B under `config.experiment_metadata` so the
interpretation / rationale / hypothesis are searchable from the W&B UI.

---

## 5. Naming conventions

| Pattern | Meaning |
| --- | --- |
| `<task>.yaml` | First version of a new experiment family |
| `<task>_v2.yaml`, `_v3.yaml`, ... | Strict successors; carry `experiment_metadata.parent` |
| `deprecated/<task>.yaml` | Parked / superseded; reference only |

`<task>` is one of: `synthetic`, `ablation_loss`, `ablation_architecture`,
`main_results_<dataset>`. Anything else needs a manuscript section to
back it.
