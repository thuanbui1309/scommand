#!/bin/bash
# Re-run all GSC training after the TorchCodec loader fix (commit 79dc23f).
# The original GSC runs (queue_overnight + queue_missing_baselines) all
# crashed at epoch 0 with ImportError: TorchCodec. Dataset is already
# downloaded/extracted — only per-sample load failed — so no re-download.
#
# 12 runs: SeMoE 2L+1L + STASA baseline 2L+1L, × 3 seeds.
#
# Usage:
#   tmux new -ds gsc_rerun "bash scripts/queue_gsc_rerun.sh"
# Watch:  tail -f runs/queue_gsc_rerun.log

set -u
cd "$(dirname "$0")/.."

LOG=runs/queue_gsc_rerun.log
mkdir -p runs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# tag                dataset  seed  config                              min_vram_mib
QUEUE=$(cat <<'EOF'
semoe_gsc-2l    gsc  0  configs/variant/semoe-k4.yaml       11000
semoe_gsc-2l    gsc  1  configs/variant/semoe-k4.yaml       11000
semoe_gsc-2l    gsc  2  configs/variant/semoe-k4.yaml       11000
semoe_gsc-1l    gsc  0  configs/variant/semoe-k4-1l.yaml    6000
semoe_gsc-1l    gsc  1  configs/variant/semoe-k4-1l.yaml    6000
semoe_gsc-1l    gsc  2  configs/variant/semoe-k4-1l.yaml    6000
baseline-gsc-2l gsc  0  configs/variant/baseline.yaml       11000
baseline-gsc-2l gsc  1  configs/variant/baseline.yaml       11000
baseline-gsc-2l gsc  2  configs/variant/baseline.yaml       11000
baseline-gsc-1l gsc  0  configs/variant/baseline-1l.yaml    6000
baseline-gsc-1l gsc  1  configs/variant/baseline-1l.yaml    6000
baseline-gsc-1l gsc  2  configs/variant/baseline-1l.yaml    6000
EOF
)

log "GSC re-run queue starting with $(echo "$QUEUE" | grep -c .) entries"

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

log "GSC re-run queue complete"
