#!/bin/bash
# SHD 3-axis (acc / params / FR) analysis — SeMoE vs Phase 02 STASA baseline
# plus eval-time pruning of dead experts on each SeMoE checkpoint.
#
# Eval-only — fits inside the spare VRAM the training queue isn't using.
# Outputs to runs/shd_3axis_analysis.log (also stdout).
#
# Usage:  bash scripts/analyze_shd_3axis.sh

set -u
cd "$(dirname "$0")/.."

LOG=runs/shd_3axis_analysis.log
mkdir -p runs
: > "$LOG"

run() {
  echo "$@" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
}

# (seed, semoe_run_dir_timestamp). Matches the finished SeMoE SHD runs.
SEMOE_RUNS=(
  "0:032253"
  "1:033437"
  "2:033437"
)

# Phase 02 STASA baseline 5-seed sweep timestamp (per baseline-repro-report.md)
BASELINE_TS="20260424_233649"

echo "=== [1/3] SeMoE 3-axis ===" | tee -a "$LOG"
for entry in "${SEMOE_RUNS[@]}"; do
  seed=${entry%:*}
  ts=${entry#*:}
  dir="runs/shd_seed${seed}_20260515_${ts}"
  if [ ! -f "$dir/best_acc.pt" ]; then
    echo "MISSING $dir/best_acc.pt — skip" | tee -a "$LOG"
    continue
  fi
  run python -m scripts.measure_3axis \
      --checkpoint "$dir/best_acc.pt" \
      --config "$dir/config.yaml" \
      --dataset shd \
      --variant "semoe_seed${seed}"
done

echo "=== [2/3] STASA baseline (Phase 02, 233649) 3-axis ===" | tee -a "$LOG"
for seed in 0 1 2; do
  dir="runs/shd_seed${seed}_${BASELINE_TS}"
  if [ ! -f "$dir/best_acc.pt" ]; then
    echo "MISSING $dir/best_acc.pt — skip" | tee -a "$LOG"
    continue
  fi
  run python -m scripts.measure_3axis \
      --checkpoint "$dir/best_acc.pt" \
      --config "$dir/config.yaml" \
      --dataset shd \
      --variant "baseline_seed${seed}"
done

echo "=== [3/3] SeMoE expert pruning (threshold 0.05) ===" | tee -a "$LOG"
for entry in "${SEMOE_RUNS[@]}"; do
  seed=${entry%:*}
  ts=${entry#*:}
  dir="runs/shd_seed${seed}_20260515_${ts}"
  if [ ! -f "$dir/best_acc.pt" ]; then
    echo "MISSING $dir/best_acc.pt — skip" | tee -a "$LOG"
    continue
  fi
  run python -m scripts.prune_semoe \
      --checkpoint "$dir/best_acc.pt" \
      --config "$dir/config.yaml" \
      --dataset shd \
      --threshold 0.05
done

echo | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
