"""Aggregate multi-seed training results into mean ± std summary.

Usage:
    python -m scripts.aggregate_seeds \\
        --runs-glob 'runs/shd_seed*' \\
        --output seed_summary.csv

Reads the final row (last epoch) of each run's ``metrics.csv``, computes
mean ± std over seeds, and emits a summary CSV plus a console table that
includes paper Table 1 comparison columns.

Paper baselines (Table 1):
    SHD: 95.6%   SSC: 80.3%   GSC: 96.3%
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# Paper Table 1 reference accuracies (mean over seeds)
_PAPER_BASELINES = {
    "shd": 95.6,
    "ssc": 80.3,
    "gsc": 96.3,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multi-seed SpikCommander results")
    p.add_argument("--runs-glob", required=True,
                   help="Glob pattern matching run directories, e.g. 'runs/shd_seed*'")
    p.add_argument("--output", default="seed_summary.csv",
                   help="Output CSV path (default: seed_summary.csv)")
    p.add_argument("--dataset", default=None, choices=["shd", "ssc", "gsc", None],
                   help="Dataset name for paper comparison column (auto-detected from glob if omitted)")
    return p.parse_args()


def _read_final_row(metrics_csv: str) -> dict | None:
    """Read the last data row of a metrics.csv. Returns None if unreadable."""
    if not os.path.isfile(metrics_csv):
        return None
    try:
        with open(metrics_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return rows[-1]
    except Exception as e:
        print(f"  WARNING: could not read {metrics_csv}: {e}")
        return None


def main() -> None:
    args = _parse_args()

    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        print(f"No directories matched: {args.runs_glob!r}")
        sys.exit(1)

    print(f"Found {len(run_dirs)} run(s):")

    records = []
    for run_dir in run_dirs:
        metrics_csv = os.path.join(run_dir, "metrics.csv")
        row = _read_final_row(metrics_csv)
        if row is None:
            print(f"  SKIP {run_dir} (no metrics.csv or empty)")
            continue
        try:
            rec = {
                "run_dir": run_dir,
                "epoch": int(row["epoch"]),
                "acc_val": float(row["acc_val"]),
                "loss_val": float(row["loss_val"]),
                "acc_train": float(row["acc_train"]),
                "loss_train": float(row["loss_train"]),
                "lr": float(row["lr"]),
            }
            records.append(rec)
            print(f"  OK  {os.path.basename(run_dir):40s}  "
                  f"epoch={rec['epoch']}  acc_val={100*rec['acc_val']:.2f}%")
        except (KeyError, ValueError) as e:
            print(f"  SKIP {run_dir} (parse error: {e})")

    if not records:
        print("No valid runs to aggregate.")
        sys.exit(1)

    acc_vals = np.array([r["acc_val"] for r in records]) * 100.0  # -> percent
    loss_vals = np.array([r["loss_val"] for r in records])

    mean_acc = float(np.mean(acc_vals))
    std_acc = float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0
    mean_loss = float(np.mean(loss_vals))
    std_loss = float(np.std(loss_vals, ddof=1)) if len(loss_vals) > 1 else 0.0

    # Auto-detect dataset from glob pattern
    dataset = args.dataset
    if dataset is None:
        for ds in ["shd", "ssc", "gsc"]:
            if ds in args.runs_glob.lower():
                dataset = ds
                break

    paper_acc = _PAPER_BASELINES.get(dataset, float("nan")) if dataset else float("nan")
    delta = mean_acc - paper_acc if not np.isnan(paper_acc) else float("nan")

    # Console table
    print(f"\n{'='*60}")
    print(f"  Seeds: {len(records)}")
    print(f"  Acc  val: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Loss val: {mean_loss:.4f} ± {std_loss:.4f}")
    if dataset:
        print(f"  Paper ({dataset.upper()}): {paper_acc:.1f}%")
        sign = "+" if delta >= 0 else ""
        print(f"  Delta vs paper: {sign}{delta:.2f}%")
    print(f"{'='*60}\n")

    # Write summary CSV
    fieldnames = [
        "n_seeds", "dataset",
        "mean_acc_pct", "std_acc_pct",
        "mean_loss_val", "std_loss_val",
        "paper_acc_pct", "delta_pct",
        "run_dirs",
    ]
    row = {
        "n_seeds": len(records),
        "dataset": dataset or "unknown",
        "mean_acc_pct": f"{mean_acc:.4f}",
        "std_acc_pct": f"{std_acc:.4f}",
        "mean_loss_val": f"{mean_loss:.6f}",
        "std_loss_val": f"{std_loss:.6f}",
        "paper_acc_pct": f"{paper_acc:.1f}" if not np.isnan(paper_acc) else "N/A",
        "delta_pct": f"{delta:.4f}" if not np.isnan(delta) else "N/A",
        "run_dirs": "|".join(r["run_dir"] for r in records),
    }

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"Summary written to: {args.output}")


if __name__ == "__main__":
    main()
