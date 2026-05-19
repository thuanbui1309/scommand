#!/bin/bash
# Re-run critical cells SIGKILLed by GPU oversubscription OOM (2026-05-19).
# SSC-2L K2 ×3 (the key "does K2 cure SSC headline" question) + the one
# missing GSC-2L K2 seed (seeds 1,2 already DONE → need seed 0).
#
# Conservative VRAM threshold (12000) so a 2L run launches ONLY with real
# headroom — prevents the mid-epoch peak OOM that killed the originals.
# Runs alongside p5final; self-defers when p5final's runs saturate the GPU.
#
# Usage:  tmux new -ds rerun "bash scripts/queue_rerun_critical.sh"
# Watch:  tail -f runs/queue_rerun_critical.log

set -u
cd "$(dirname "$0")/.."

LOG=runs/queue_rerun_critical.log
mkdir -p runs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# tag                dataset  seed  config                          min_vram_mib
QUEUE=$(cat <<'EOF'
semoe-k2-ssc-2l    ssc  0  configs/variant/semoe-k2.yaml      12000
semoe-k2-ssc-2l    ssc  1  configs/variant/semoe-k2.yaml      12000
semoe-k2-ssc-2l    ssc  2  configs/variant/semoe-k2.yaml      12000
semoe-k2-gsc-2l    gsc  0  configs/variant/semoe-k2.yaml      12000
EOF
)

log "Critical re-run queue starting with $(echo "$QUEUE" | grep -c .) entries"

while read -r tag dataset seed config min_vram; do
  [ -z "${tag:-}" ] && continue
  session="${tag}_s${seed}"

  if tmux has-session -t "$session" 2>/dev/null; then
    log "SKIP $session (tmux session already running)"
    continue
  fi

  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    # Global concurrency guard: never let total training sessions exceed 5
    # (prevents the runaway oversubscription that caused the OOM).
    nsess=$(tmux ls 2>/dev/null | grep -cE '^(semoe|baseline)-' || true)
    if (( free >= min_vram && nsess < 5 )); then break; fi
    log "WAIT $session — need ${min_vram}MiB free (have ${free}), sessions ${nsess}/5"
    sleep 120
  done

  log "LAUNCH $session (free=${free}MiB, need=${min_vram}MiB, sessions=${nsess})"
  tmux new -ds "$session" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate scommander && cd ~/vstung/thuanbui/scommand && python -m scripts.train --config $config --dataset $dataset --seed $seed --output-dir runs 2>&1 | tee runs/${tag}_seed${seed}_rerun.log"

  sleep 90
done <<< "$QUEUE"

log "Critical re-run queue complete"
