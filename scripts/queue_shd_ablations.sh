#!/bin/bash
# SHD ablation queue — same pattern as queue_overnight.sh but smaller and
# SHD-only. Runs each ablation × 3 seeds, gating on VRAM headroom.
#
# Tags chosen so status_report.py + analyze scripts can find them by
# config-derived attn name (always "semoe") plus the run dir prefix.
#
# Usage:
#   tmux new -ds shd_abl "bash scripts/queue_shd_ablations.sh"
#
# Watch:  tail -f runs/queue_shd_ablations.log

set -u
cd "$(dirname "$0")/.."

LOG=runs/queue_shd_ablations.log
mkdir -p runs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# tag                 dataset  seed  config                                     min_vram_mib
# Trimmed 2026-05-15: dropped lb1e3, lb5e2, lb1e1 (LB strength sweep). Keep
# no-lb only as the LB necessity ablation (extreme λ=0 vs default λ=0.01).
# Five SeMoE-specific ablations remain: K sweep (K2, K6) + K3-noid (identity
# expert role) + no-lb (LB necessity) + fulld (D_e choice).
QUEUE=$(cat <<'EOF'
shd-k4-no-lb   shd  0  configs/variant/semoe-k4-no-lb.yaml      4000
shd-k4-no-lb   shd  1  configs/variant/semoe-k4-no-lb.yaml      4000
shd-k4-no-lb   shd  2  configs/variant/semoe-k4-no-lb.yaml      4000
shd-k3-noid    shd  0  configs/variant/semoe-k3-no-identity.yaml 3500
shd-k3-noid    shd  1  configs/variant/semoe-k3-no-identity.yaml 3500
shd-k3-noid    shd  2  configs/variant/semoe-k3-no-identity.yaml 3500
shd-k2         shd  0  configs/variant/semoe-k2.yaml             3000
shd-k2         shd  1  configs/variant/semoe-k2.yaml             3000
shd-k2         shd  2  configs/variant/semoe-k2.yaml             3000
shd-k6         shd  0  configs/variant/semoe-k6.yaml             5000
shd-k6         shd  1  configs/variant/semoe-k6.yaml             5000
shd-k6         shd  2  configs/variant/semoe-k6.yaml             5000
shd-k4-fulld   shd  0  configs/variant/semoe-k4-fulld.yaml       4500
shd-k4-fulld   shd  1  configs/variant/semoe-k4-fulld.yaml       4500
shd-k4-fulld   shd  2  configs/variant/semoe-k4-fulld.yaml       4500
EOF
)

log "SHD ablation queue starting with $(echo "$QUEUE" | grep -c .) entries"

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

  sleep 60
done <<< "$QUEUE"

log "SHD ablation queue complete — all entries launched (or skipped)"
