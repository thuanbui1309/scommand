#!/bin/bash
# Phase 05 finalization queue — full hero-candidate sweep on SSC + GSC so we
# can pick the best variant PER DATASET (K2 winning on SHD does not imply it
# wins on SSC/GSC). Tiered by priority: if the deadline hits, Tier 1 (the
# must-have headline cells) completes first; partial results are still usable.
#
# Candidates from SHD ablation top-3: K2 (best SHD), K4 (spec default),
# K6 (extra capacity). SSC-K4 already done (skipped). SSC-1L STASA baseline
# done; SSC-2L STASA baseline from Phase 02. GSC has NO baseline yet — its
# STASA baseline is Tier 1 (blocks all GSC comparison).
#
# Usage:  tmux new -ds p5final "bash scripts/queue_phase05_final.sh"
# Watch:  tail -f runs/queue_phase05_final.log

set -u
cd "$(dirname "$0")/.."

LOG=runs/queue_phase05_final.log
mkdir -p runs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# tag                  dataset  seed  config                            min_vram_mib
# Tier 1 — headline must-have (GSC baseline + K2/K4 2L on SSC/GSC)
# Tier 2 — 1L scaling rows
# Tier 3 — K6 candidate
QUEUE=$(cat <<'EOF'
baseline-gsc-2l    gsc  0  configs/variant/baseline.yaml      11000
baseline-gsc-2l    gsc  1  configs/variant/baseline.yaml      11000
baseline-gsc-2l    gsc  2  configs/variant/baseline.yaml      11000
semoe-k2-ssc-2l    ssc  0  configs/variant/semoe-k2.yaml      9000
semoe-k2-ssc-2l    ssc  1  configs/variant/semoe-k2.yaml      9000
semoe-k2-ssc-2l    ssc  2  configs/variant/semoe-k2.yaml      9000
semoe-k2-gsc-2l    gsc  0  configs/variant/semoe-k2.yaml      9000
semoe-k2-gsc-2l    gsc  1  configs/variant/semoe-k2.yaml      9000
semoe-k2-gsc-2l    gsc  2  configs/variant/semoe-k2.yaml      9000
semoe-k4-gsc-2l    gsc  0  configs/variant/semoe-k4.yaml      11000
semoe-k4-gsc-2l    gsc  1  configs/variant/semoe-k4.yaml      11000
semoe-k4-gsc-2l    gsc  2  configs/variant/semoe-k4.yaml      11000
baseline-gsc-1l    gsc  0  configs/variant/baseline-1l.yaml   6000
baseline-gsc-1l    gsc  1  configs/variant/baseline-1l.yaml   6000
baseline-gsc-1l    gsc  2  configs/variant/baseline-1l.yaml   6000
semoe-k2-ssc-1l    ssc  0  configs/variant/semoe-k2-1l.yaml   6000
semoe-k2-ssc-1l    ssc  1  configs/variant/semoe-k2-1l.yaml   6000
semoe-k2-ssc-1l    ssc  2  configs/variant/semoe-k2-1l.yaml   6000
semoe-k2-gsc-1l    gsc  0  configs/variant/semoe-k2-1l.yaml   6000
semoe-k2-gsc-1l    gsc  1  configs/variant/semoe-k2-1l.yaml   6000
semoe-k2-gsc-1l    gsc  2  configs/variant/semoe-k2-1l.yaml   6000
semoe-k4-gsc-1l    gsc  0  configs/variant/semoe-k4-1l.yaml   6000
semoe-k4-gsc-1l    gsc  1  configs/variant/semoe-k4-1l.yaml   6000
semoe-k4-gsc-1l    gsc  2  configs/variant/semoe-k4-1l.yaml   6000
semoe-k6-ssc-2l    ssc  0  configs/variant/semoe-k6.yaml      9000
semoe-k6-ssc-2l    ssc  1  configs/variant/semoe-k6.yaml      9000
semoe-k6-ssc-2l    ssc  2  configs/variant/semoe-k6.yaml      9000
semoe-k6-gsc-2l    gsc  0  configs/variant/semoe-k6.yaml      9000
semoe-k6-gsc-2l    gsc  1  configs/variant/semoe-k6.yaml      9000
semoe-k6-gsc-2l    gsc  2  configs/variant/semoe-k6.yaml      9000
semoe-k6-ssc-1l    ssc  0  configs/variant/semoe-k6-1l.yaml   6000
semoe-k6-ssc-1l    ssc  1  configs/variant/semoe-k6-1l.yaml   6000
semoe-k6-ssc-1l    ssc  2  configs/variant/semoe-k6-1l.yaml   6000
semoe-k6-gsc-1l    gsc  0  configs/variant/semoe-k6-1l.yaml   6000
semoe-k6-gsc-1l    gsc  1  configs/variant/semoe-k6-1l.yaml   6000
semoe-k6-gsc-1l    gsc  2  configs/variant/semoe-k6-1l.yaml   6000
EOF
)

log "Phase 05 final queue starting with $(echo "$QUEUE" | grep -c .) entries"

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

log "Phase 05 final queue complete"
