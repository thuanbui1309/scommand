"""Inspect SeMoE expert-usage trajectory across training epochs.

Reads ``semoe_expert_usage.csv`` (written by trainer) and prints per-expert
mean/std/min/max plus the final 10 epochs and a "phase change" detection.
Useful to answer "did the gate collapse, and when".

Usage:
    python -m scripts.analyze_gate_stability --run-dir runs/shd_seed0_<ts>
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SeMoE gate stability analysis")
    p.add_argument("--run-dir", required=True, help="Run directory containing semoe_expert_usage.csv")
    p.add_argument("--phase-shift-thresh", type=float, default=0.20,
                   help="Per-expert |Δusage| above this between consecutive epochs is flagged (default 0.20)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    csv_path = os.path.join(args.run_dir, "semoe_expert_usage.csv")
    if not os.path.isfile(csv_path):
        print(f"MISSING: {csv_path}")
        raise SystemExit(1)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"EMPTY: {csv_path}")
        raise SystemExit(1)

    expert_cols = [c for c in rows[0].keys() if c != "epoch"]
    series: dict[str, list[float]] = defaultdict(list)
    epochs: list[int] = []
    for r in rows:
        epochs.append(int(r["epoch"]))
        for c in expert_cols:
            series[c].append(float(r[c]))

    print(f"## {csv_path}")
    print(f"   epochs: {len(rows)}  (range {epochs[0]}..{epochs[-1]})")
    print(f"   columns: {len(expert_cols)}")

    print(f"\n## Per-expert usage stats:")
    print(f"   {'col':<32s} {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s} {'final':>7s}")
    for c in expert_cols:
        v = series[c]
        n = len(v)
        m = sum(v) / n
        sd = (sum((x - m) ** 2 for x in v) / n) ** 0.5
        print(f"   {c:<32s} {m:>7.3f} {sd:>7.3f} {min(v):>7.3f} {max(v):>7.3f} {v[-1]:>7.3f}")

    # Phase-shift detection: when any expert's usage swings by > thresh
    # between consecutive epochs, flag it. Indicates gate instability.
    shifts: list[tuple[int, str, float, float]] = []
    for i in range(1, len(rows)):
        for c in expert_cols:
            d = series[c][i] - series[c][i - 1]
            if abs(d) >= args.phase_shift_thresh:
                shifts.append((epochs[i], c, series[c][i - 1], series[c][i]))

    print(f"\n## Phase shifts (|Δ|≥{args.phase_shift_thresh}) — gate instability events:")
    if not shifts:
        print("   none — gate is stable across training")
    else:
        for ep, c, before, after in shifts[:20]:
            print(f"   epoch {ep:4d}  {c}:  {before:.3f} → {after:.3f}  (Δ={after-before:+.3f})")
        if len(shifts) > 20:
            print(f"   ... +{len(shifts) - 20} more events")

    # Tail
    tail_n = min(10, len(rows))
    print(f"\n## Final {tail_n} epochs:")
    print("   epoch  " + "  ".join(f"{c:>10s}" for c in expert_cols))
    for r in rows[-tail_n:]:
        vals = "  ".join(f"{float(r[c]):>10.3f}" for c in expert_cols)
        print(f"   {int(r['epoch']):>5d}  {vals}")


if __name__ == "__main__":
    main()
