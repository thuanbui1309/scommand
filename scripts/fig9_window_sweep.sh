#!/bin/bash
# Fig 9 — SWA window radius sensitivity sweep on SHD K2 hero.
# Sweeps w ∈ {4, 8, 12, 16, 20, 32, 50} × 2 seeds = 14 runs × ~20min = ~5h.
# Generates tmp variant configs that extend semoe-k2-1l.yaml with window_radius override.
#
# Usage (from repo root on server):
#   bash src/scripts/fig9_window_sweep.sh
# Or wrapped in tmux:
#   tmux new -ds fig9 'bash src/scripts/fig9_window_sweep.sh'

set -eu
# Resolve repo root robustly: prefer git, fall back to script-relative.
if root=$(git rev-parse --show-toplevel 2>/dev/null); then cd "$root"
else cd "$(dirname "$0")/.."
fi

WINDOWS=(4 8 12 16 20 32 50)
SEEDS=(0 1)
LOG_DIR=logs
TMP_CFG_DIR=/tmp/fig9_configs
MAIN_LOG=$LOG_DIR/fig9-window-sweep-$(date +%y%m%d).log

mkdir -p "$LOG_DIR" "$TMP_CFG_DIR"
: > "$MAIN_LOG"

echo "=== FIG 9 WINDOW SWEEP START $(date) ===" | tee -a "$MAIN_LOG"
echo "windows=${WINDOWS[*]} seeds=${SEEDS[*]}" | tee -a "$MAIN_LOG"

for w in "${WINDOWS[@]}"; do
  cfg=$TMP_CFG_DIR/semoe-k2-1l-w${w}.yaml
  # Generate complete K2-1L config with window_radius override.
  # Inline (not append to base) to avoid YAML duplicate-key error.
  cat > "$cfg" << YAML
model:
  attention: semoe
  depth: 1
  window_radius: $w
  semoe:
    num_experts: 2
    expert_types: [swa, identity]
    load_balance_weight: 0.01
training:
  grad_clip: 1.0
YAML

  for s in "${SEEDS[@]}"; do
    tag="fig9-w${w}-s${s}"
    echo "" | tee -a "$MAIN_LOG"
    echo "=== LAUNCH $tag $(date +%T) ===" | tee -a "$MAIN_LOG"
    python -m scripts.train --config "$cfg" --dataset shd --seed "$s" 2>&1 \
      | tee -a "$MAIN_LOG" \
      || echo "FAIL $tag" | tee -a "$MAIN_LOG"
  done
done

echo "" | tee -a "$MAIN_LOG"
echo "=== FIG 9 WINDOW SWEEP DONE $(date) ===" | tee -a "$MAIN_LOG"
