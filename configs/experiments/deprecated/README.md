# Deprecated experiment configs

> Lineage-only. Not loaded by `src.experiments.run` (Hydra picks up
> only top-level files in `configs/experiments/`). Re-promoting any of
> these requires moving them back up and adding a fresh
> `experiment_metadata` block per `configs/README.md`.

## Why these are here

### `main_results_voxpopuli{,_v2,_v3,_v4}.yaml` — VoxPopuli main results (parked)

Across four config versions on the VoxPopuli dataset, the proposed
hierarchical transformer router collapses to picking HuBERT 100% of
the time, despite Whisper being the per-clip oracle on ~46% of clips.

Detailed timeline (mirrors `_comment_:` fields inside each file):

| Version | Diff vs predecessor | Outcome |
| --- | --- | --- |
| `v1` | First attempt. 16-seed sweep over `mlp_pool_hard_ce` and `hierarchical_transformer_proposed` with the AMI-tuned loss (`λ_wer=1, λ_hard=0.3, λ_soft=0.5, τ=1.5`). | Router collapsed to HuBERT under all 16 seeds; `class_balanced_loss=true` was using AMI-hard-coded priors at the time, which were wrong for VoxPopuli. |
| `v2` | Hard-fix the class-balanced-loss bug (`class_balanced_loss=false`); raise `aux_ce_weight` to 0.5 and lower `τ` to 1.0 to "give a stronger signal" and "sharpen the soft labels". 3 seeds. | Same collapse. Whisper's outlier WERs (~120% on hallucinated clips) generated penalty signal so harsh under the soft-CE that the model learned to never pick Whisper at all. |
| `v3` | Extreme risk-taking: drop `λ_wer` to 0.1 (do not fear the outliers), raise `λ_soft` to 1.0, sharpen `τ` to 0.05 (force the soft target to be a near-hard pointer at the per-clip oracle). 3 seeds. | Same collapse, modest improvement at best. Tested the "Acoustic Predictability" hypothesis: if even an extreme risk-taking config still cannot pick Whisper, the failure is not because the loss is too penalising — it is because Whisper's hallucination errors are not predictable from acoustic features. |
| `v4` | Roll back to v2 hyperparams but enable `class_balanced_loss=true` with the (now-correct) dynamic train-set priors. 3 seeds. | No improvement. The dynamic class balancing helped on AMI in the loss ablation, but not enough to overcome the Whisper-outlier penalty issue on VoxPopuli. |

## Conclusion (as of 2026-05-01)

The VoxPopuli failure is structural: the per-expert WER table on
this corpus has a heavy-tailed distribution for Whisper, and the
soft-CE objective is dominated by the worst-case clips. None of the
four config-side mitigations recovered Whisper as a usable expert.

Reviving this deliverable would likely require one of:

1. Replacing Whisper-base with Whisper-large (fewer hallucinations →
   tail of WER vector compressed → soft-CE penalty manageable);
2. Per-clip outlier clipping in the loss (clip `WER_{n,k}` at, say,
   `1.0` before computing the soft target so a 120% clip is not
   dominating the gradient); or
3. A different routing target, e.g. weighted ROVER as the oracle
   instead of the per-clip arg-min.

None of these are planned for the current submission. See
`PROJECT_STATE.md` §5 for the active blocker entry.
