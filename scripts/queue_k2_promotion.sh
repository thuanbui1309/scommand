#!/bin/bash
# Promote the K2 (SWA + identity) hero to SSC + GSC. SHD showed K2 is
# Pareto-dominant over baseline (acc +0.59, params -28%, energy -49%) and
# beats the K4 hero. This queue verifies whether K2 generalises to the
# harder 35-class datasets.
#
# SSC first (lighter, answers the critical question faster), then GSC.
# Runs alongside the GSC-K4 queue (queue_gsc_rerun) — both gate on VRAM.
# 12 runs: K2 × {SSC,GSC} × {2L,1L} × 3 seeds.
#
# Usage:  tmux new -ds k2_promo "bash scripts/queue_k2_promotion.sh"
# Watch:  tail -f runs/queue_k2_promotion.log

set -u
cd "$(dirname "$0")/.."

LOG=runs/queue_k2_promotion.log
mkdir -p runs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# tag                dataset  seed  config                              min_vram_mib
QUEUE=$(cat <<'EOF'
semoe-k2-ssc-2l  ssc  0  configs/variant/semoe-k2.yaml       9000
semoe-k2-ssc-2l  ssc  1  configs/variant/semoe-k2.yaml       9000
semoe-k2-ssc-2l  ssc  2  configs/variant/semoe-k2.yaml       9000
semoe-k2-ssc-1l  ssc  0  configs/variant/semoe-k2-1l.yaml    6000
semoe-k2-ssc-1l  ssc  1  configs/variant/semoe-k2-1l.yaml    6000
semoe-k2-ssc-1l  ssc  2  configs/variant/semoe-k2-1l.yaml    6000
semoe-k2-gsc-2l  gsc  0  configs/variant/semoe-k2.yaml       9000
semoe-k2-gsc-2l  gsc  1  configs/variant/semoe-k2.yaml       9000
semoe-k2-gsc-2l  gsc  2  configs/variant/semoe-k2.yaml       9000
semoe-k2-gsc-1l  gsc  0  configs/variant/semoe-k2-1l.yaml    6000
semoe-k2-gsc-1l  gsc  1  configs/variant/semoe-k2-1l.yaml    6000
semoe-k2-gsc-1l  gsc  2  configs/variant/semoe-k2-1l.yaml    6000
EOF
)

log "K2 promotion queue starting with $(echo "$QUEUE" | grep -c .) entries"

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
    sleep 120
  done

  log "LAUNCH $session (free=${free}MiB, need=${min_vram}MiB, config=$config)"
  tmux new -ds "$session" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate scommander && cd ~/vstung/thuanbui/scommand && python -m scripts.train --config $config --dataset $dataset --seed $seed --output-dir runs 2>&1 | tee runs/${tag}_seed${seed}.log"

  sleep 90
done <<< "$QUEUE"

log "K2 promotion queue complete"
