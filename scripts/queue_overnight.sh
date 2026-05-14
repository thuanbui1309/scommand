#!/bin/bash
# Overnight SeMoE training queue.
#
# Walks the QUEUE list below, waiting for VRAM headroom before launching each
# run into its own tmux session. Each entry: tag dataset seed config min_vram_mib.
# Already-launched sessions (tmux name collision) are skipped.
#
# Usage (run inside the conda env, in the repo root):
#   tmux new -ds queue "bash scripts/queue_overnight.sh"
#
# Watch progress:  tail -f runs/queue.log

set -u
cd "$(dirname "$0")/.."

QUEUE_LOG=runs/queue.log
mkdir -p runs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

# tag         dataset  seed  config                                   min_vram_mib
QUEUE=$(cat <<'EOF'
ssc-2l       ssc      1     configs/variant/semoe-k4.yaml            11000
ssc-2l       ssc      2     configs/variant/semoe-k4.yaml            11000
ssc-1l       ssc      0     configs/variant/semoe-k4-1l.yaml         6000
ssc-1l       ssc      1     configs/variant/semoe-k4-1l.yaml         6000
ssc-1l       ssc      2     configs/variant/semoe-k4-1l.yaml         6000
gsc-2l       gsc      0     configs/variant/semoe-k4.yaml            11000
gsc-2l       gsc      1     configs/variant/semoe-k4.yaml            11000
gsc-2l       gsc      2     configs/variant/semoe-k4.yaml            11000
gsc-1l       gsc      0     configs/variant/semoe-k4-1l.yaml         6000
gsc-1l       gsc      1     configs/variant/semoe-k4-1l.yaml         6000
gsc-1l       gsc      2     configs/variant/semoe-k4-1l.yaml         6000
EOF
)

log "Queue starting with $(echo "$QUEUE" | grep -c .) entries"

while read -r tag dataset seed config min_vram; do
  [ -z "${tag:-}" ] && continue
  session="semoe_${tag}_s${seed}"

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
  tmux new -ds "$session" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate scommander && cd ~/vstung/thuanbui/scommand && python -m scripts.train --config $config --dataset $dataset --seed $seed --output-dir runs 2>&1 | tee runs/semoe_${tag}_seed${seed}.log"

  # Let the new run allocate VRAM before checking the next entry
  sleep 60
done <<< "$QUEUE"

log "Queue complete — all entries launched (or skipped)"
