"""Auto-generate experiment status table from runs/ directory.

Walks every ``runs/<dataset>_seed*_<timestamp>/`` directory, reads its
``config.yaml`` + ``metrics.csv``, classifies each run by (dataset, depth,
attention type, variant tag), and emits a markdown status report grouping
runs into completed / running / abandoned buckets.

Designed to be re-run periodically — output is fully derived from runs/,
so re-running always reflects the latest disk state.

Usage:
    python -m scripts.status_report                      # stdout
    python -m scripts.status_report --output plans/reports/status-experiments.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SeMoE experiment status report")
    p.add_argument("--runs-dir", default="runs", help="Root dir containing run subdirs (default: runs)")
    p.add_argument("--output", default=None, help="Write markdown report to this path (default: stdout)")
    p.add_argument("--date-prefix", default="20260515",
                   help="Restrict to run dirs with this date prefix in timestamp (default: 20260515)")
    return p.parse_args()


def _read_metrics(metrics_path: str) -> tuple[int | None, float | None, float | None]:
    """Return (last_epoch, best_val_acc, last_val_acc). None on read failure."""
    if not os.path.isfile(metrics_path):
        return None, None, None
    last_epoch: int | None = None
    last_val: float | None = None
    best_val = -1.0
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e = int(row["epoch"])
                v = float(row["acc_val"])
            except (KeyError, ValueError):
                continue
            last_epoch = e
            last_val = v
            if v > best_val:
                best_val = v
    return last_epoch, (best_val if best_val >= 0 else None), last_val


def _classify_run(run_dir: str) -> dict | None:
    """Read config + metrics; return record dict or None on failure."""
    cfg_path = os.path.join(run_dir, "config.yaml")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(cfg_path):
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception:
        return None

    dataset = OmegaConf.select(cfg, "dataset.name", default="?")
    depth = OmegaConf.select(cfg, "model.depth", default=1)
    attn = OmegaConf.select(cfg, "model.attention", default="stasa")
    seed = OmegaConf.select(cfg, "experiment.seed", default=-1)
    target_epochs = OmegaConf.select(cfg, "training.epochs", default=0)

    # Variant signature — distinguishes SeMoE ablations that otherwise share
    # (attn, dataset, depth). Reads the semoe sub-config.
    if attn == "semoe":
        K = OmegaConf.select(cfg, "model.semoe.num_experts", default=4)
        lb = OmegaConf.select(cfg, "model.semoe.load_balance_weight", default=0.01)
        etypes = OmegaConf.select(cfg, "model.semoe.expert_types", default=None)
        edim = OmegaConf.select(cfg, "model.semoe.expert_dim", default=None)
        if etypes is not None and hasattr(etypes, "__iter__"):
            etypes = list(etypes)
        sig = f"K{K}-lb{lb}"
        if etypes and "identity" not in etypes:
            sig += "-noid"
        if edim is not None:
            sig += "-fulld"
        variant = sig
    else:
        variant = "baseline"

    last_epoch, best_val, last_val = _read_metrics(metrics_path)

    # Has best_acc.pt → at least one improving epoch
    has_ckpt = os.path.isfile(os.path.join(run_dir, "best_acc.pt"))

    # Status heuristic:
    #   DONE if last_epoch reached target_epochs - 1
    #   RUNNING if a fresh metrics row appeared in the last 5 minutes
    #   ABANDONED otherwise
    status = "?"
    if last_epoch is not None and last_epoch >= int(target_epochs) - 1:
        status = "DONE"
    else:
        try:
            mtime = os.path.getmtime(metrics_path) if os.path.isfile(metrics_path) else 0
            age_s = datetime.now().timestamp() - mtime
            status = "RUNNING" if age_s < 300 else "ABANDONED"
        except OSError:
            status = "?"

    return {
        "run_dir": os.path.basename(run_dir),
        "dataset": dataset,
        "depth": int(depth) if depth is not None else 1,
        "attn": attn,
        "variant": variant,
        "seed": int(seed) if seed != -1 else -1,
        "target_epochs": int(target_epochs),
        "last_epoch": last_epoch,
        "best_val": best_val,
        "last_val": last_val,
        "has_ckpt": has_ckpt,
        "status": status,
    }


def _format_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{100*v:.2f}"


def _emit_markdown(records: list[dict], date_prefix: str) -> str:
    """Group records by (attn, dataset, depth, variant) then by status."""
    # Group: (attn, dataset, depth, variant) -> [records sorted by seed]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["attn"], r["dataset"], r["depth"], r.get("variant", "?"))
        groups[key].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: (r["seed"], r["run_dir"]))

    out: list[str] = []
    out.append(f"# Experiment Status (auto-generated)")
    out.append(f"")
    out.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    out.append(f"- Filter: runs/*_seed*_{date_prefix}_*")
    out.append(f"- Total runs found: {len(records)}")
    out.append(f"")

    # Summary table
    out.append("## Summary by config")
    out.append("")
    out.append("| attn | dataset | depth | variant | total | DONE | RUN | ABND | best mean | seeds(DONE) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        attn, dataset, depth, variant = key
        recs = groups[key]
        n_done = sum(1 for r in recs if r["status"] == "DONE")
        n_running = sum(1 for r in recs if r["status"] == "RUNNING")
        n_aban = sum(1 for r in recs if r["status"] == "ABANDONED")
        done_recs = [r for r in recs if r["status"] == "DONE" and r["best_val"] is not None]
        if done_recs:
            best_mean = sum(r["best_val"] for r in done_recs) / len(done_recs) * 100
            best_str = f"{best_mean:.2f}"
            seeds_str = ",".join(str(r["seed"]) for r in done_recs)
        else:
            best_str = "-"
            seeds_str = "-"
        out.append(
            f"| {attn} | {dataset} | {depth}L | {variant} | {len(recs)} | "
            f"{n_done} | {n_running} | {n_aban} | {best_str} | {seeds_str} |"
        )
    out.append("")

    # Detailed per-run
    out.append("## Detailed runs (per group)")
    out.append("")
    for key in sorted(groups):
        attn, dataset, depth, variant = key
        out.append(f"### {attn} / {dataset} / {depth}L / {variant}")
        out.append("")
        out.append("| seed | run_dir | epochs | best_val | last_val | status |")
        out.append("|---|---|---|---|---|---|")
        for r in groups[key]:
            ep = "-" if r["last_epoch"] is None else f"{r['last_epoch']+1}/{r['target_epochs']}"
            out.append(
                f"| {r['seed']} | `{r['run_dir']}` | {ep} | "
                f"{_format_pct(r['best_val'])} | {_format_pct(r['last_val'])} | {r['status']} |"
            )
        out.append("")

    return "\n".join(out)


def main() -> None:
    args = _parse_args()

    pattern = os.path.join(args.runs_dir, f"*_seed*_{args.date_prefix}_*")
    run_dirs = sorted(glob.glob(pattern))

    records: list[dict] = []
    for d in run_dirs:
        if not os.path.isdir(d):
            continue
        rec = _classify_run(d)
        if rec is not None:
            records.append(rec)

    md = _emit_markdown(records, args.date_prefix)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Wrote: {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
