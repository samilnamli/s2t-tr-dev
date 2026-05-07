#!/usr/bin/env bash
# Overnight exploratory phase launcher.
#
# Runs every explore_* experiment family sequentially. Each invocation is
# isolated (separate MLflow experiment, separate process), so a crash in
# one run does not poison the next. Per-run stdout+stderr go to
# logs/overnight_<timestamp>/<name>.log.
#
# Usage:
#   bash scripts/run_overnight_explore.sh
#   bash scripts/run_overnight_explore.sh --dry-run        # print, don't run
#   bash scripts/run_overnight_explore.sh --skip-voxpopuli # AMI+synthetic only
#
# Order is from highest manuscript value to lowest, so an early kill still
# leaves a usable result set.

set -u  # do NOT use -e; we want to continue past failures

DRY_RUN=0
SKIP_VOX=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-voxpopuli) SKIP_VOX=1 ;;
  esac
done

ts=$(date +%Y-%m-%d_%H-%M-%S)
log_dir="logs/overnight_${ts}"
mkdir -p "${log_dir}"
echo "logs -> ${log_dir}"

# (name, experiment, optional cli overrides)
runs=(
  # ---- AMI ablations: highest manuscript payoff first ---------------
  "ami_loss|explore_ami_loss"
  "ami_temp|explore_ami_temp"
  "ami_arch|explore_ami_arch"
  "ami_mlp_pool_loss|explore_ami_mlp_pool_loss"

  # ---- Synthetic story: cheap, builds the figure --------------------
  "synth_main|explore_synth_main"
  "synth_R2|explore_synth_R2"
  "synth_R8|explore_synth_R8"
  "synth_R16|explore_synth_R16"
  "synth_T64|explore_synth_T64"
  "synth_T256|explore_synth_T256"
  "synth_noise_low|explore_synth_noise_low"
  "synth_noise_high|explore_synth_noise_high"

  # ---- Frame-level claim sanity (3 different seq-len truncations) ---
  "ami_seqlen_200|explore_ami_seqlen|data.max_seq_len=200"
  "ami_seqlen_500|explore_ami_seqlen|data.max_seq_len=500"
  "ami_seqlen_1000|explore_ami_seqlen|data.max_seq_len=1000"

  # ---- Optimization sweep: nice-to-have, runs late ------------------
  "ami_lr_wd|explore_ami_lr_wd"

  # ---- VoxPopuli (3 seeds): a second dataset is huge for the paper --
  "voxpopuli|explore_main_results_voxpopuli"
)

run_one() {
  local name="$1" experiment="$2" overrides="${3:-}"
  local log="${log_dir}/${name}.log"
  local cmd="python run.py experiment=${experiment}"
  if [[ -n "${overrides}" ]]; then
    cmd="${cmd} ${overrides}"
  fi
  # Override MLflow experiment name so all explore_* runs land in
  # discoverable buckets named after the experiment file.
  cmd="${cmd} mlflow.experiment_name=${experiment}"

  echo "=========================================="
  echo "[$(date +%H:%M:%S)] START ${name}"
  echo "  cmd: ${cmd}"
  echo "  log: ${log}"
  echo "=========================================="
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  local start=$(date +%s)
  if eval "${cmd}" > "${log}" 2>&1; then
    local dur=$(( $(date +%s) - start ))
    echo "[$(date +%H:%M:%S)] OK    ${name}  (${dur}s)"
  else
    local rc=$?
    local dur=$(( $(date +%s) - start ))
    echo "[$(date +%H:%M:%S)] FAIL  ${name}  rc=${rc}  (${dur}s)  -- continuing"
  fi
}

for entry in "${runs[@]}"; do
  IFS='|' read -r name experiment overrides <<< "${entry}"
  if [[ "${SKIP_VOX}" == "1" && "${name}" == "voxpopuli" ]]; then
    echo "[skip] ${name} (--skip-voxpopuli)"
    continue
  fi
  run_one "${name}" "${experiment}" "${overrides}"
done

echo
echo "DONE. Per-run logs in ${log_dir}/"
