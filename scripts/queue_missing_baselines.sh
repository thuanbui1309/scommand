#!/bin/bash
# Reproduce STASA baselines for the (dataset, depth) combos Phase 02 didn't
# cover: SSC 1L, GSC 1L, GSC 2L. Required for honest paired comparison
# vs SeMoE on these combos (currently we only compare against paper Table 1
# numbers, which leaves stack-drift as an unresolved confound).
#
# Existing Phase 02 baselines:
#   SHD 1L (5 seeds, 20260424_233649) ✓
#   SSC 2L (3 seeds, 20260424_233649) ✓
#
# Missing (this script queues):
#   SSC 1L baseline × 3 seeds
#   GSC 2L baseline × 3 seeds
#   GSC 1L baseline × 3 seeds
#
# Usage:
#   tmux new -ds baselines "bash scripts/queue_missing_baselines.sh"

set -u
cd "$(dirname "$0")/.."

LOG=runs/queue_missing_baselines.log
mkdir -p runs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# tag                 dataset  seed  config                              min_vram_mib
QUEUE=$(cat <<'EOF'
baseline-ssc-1l   ssc  0  configs/variant/baseline-1l.yaml    6000
baseline-ssc-1l   ssc  1  configs/variant/baseline-1l.yaml    6000
baseline-ssc-1l   ssc  2  configs/variant/baseline-1l.yaml    6000
baseline-gsc-2l   gsc  0  configs/variant/baseline.yaml       11000
baseline-gsc-2l   gsc  1  configs/variant/baseline.yaml       11000
baseline-gsc-2l   gsc  2  configs/variant/baseline.yaml       11000
baseline-gsc-1l   gsc  0  configs/variant/baseline-1l.yaml    6000
baseline-gsc-1l   gsc  1  configs/variant/baseline-1l.yaml    6000
baseline-gsc-1l   gsc  2  configs/variant/baseline-1l.yaml    6000
EOF
)

log "Missing baseline queue starting with $(echo "$QUEUE" | grep -c .) entries"

while read -r tag dataset seed config min_vram; do
  [ -z "${tag:-}" ] && continue
  session="${tag}_s${seed}"

  if tmux has-session -t "$session" 2>/dev/null; then
    log "SKIP $session (tmux session already running)"
    continue
  fi

  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( free >= min_vram )); then break; fi
    log "WAIT $session — need ${min_vram}MiB, free ${free}MiB"
    sleep 60
  done

  log "LAUNCH $session (free=${free}MiB, need=${min_vram}MiB, config=$config)"
  tmux new -ds "$session" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate scommander && cd ~/vstung/thuanbui/scommand && python -m scripts.train --config $config --dataset $dataset --seed $seed --output-dir runs 2>&1 | tee runs/${tag}_seed${seed}.log"

  sleep 60
done <<< "$QUEUE"

log "Missing baseline queue complete"
