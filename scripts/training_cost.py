"""Extract per-run wall-clock training time from runs/*/metrics.csv.

Reads ``time_s`` column (per-epoch seconds) emitted by trainer; sums to get
total training wall-clock per run. Aggregates by (dataset, depth, method)
group and emits paper-ready LaTeX table + JSON summary.

Counts only DONE runs (epoch count >= target_epochs). Uses same group keys
as status_report (attn = stasa | semoe; semoe disambiguated by K).

Usage (run on server where runs/ exists):
    python -m scripts.training_cost
    python -m scripts.training_cost --output-tex write/tables/table-training-cost.tex
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf


def _sum_time_s(metrics_path: str) -> tuple[int, float] | None:
    """Return (n_epochs, total_seconds). None if metrics missing/empty."""
    if not os.path.isfile(metrics_path):
        return None
    n_epochs = 0
    total = 0.0
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                total += float(row["time_s"])
                n_epochs += 1
            except (KeyError, ValueError, TypeError):
                continue
    if n_epochs == 0:
        return None
    return n_epochs, total


def _classify(run_dir: str) -> dict | None:
    """Read config; return (dataset, depth, method) classification + metadata."""
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception:
        return None

    dataset = OmegaConf.select(cfg, "dataset.name", default="?")
    depth = OmegaConf.select(cfg, "model.depth", default=1)
    attn = OmegaConf.select(cfg, "model.attention", default="stasa")
    target_epochs = OmegaConf.select(cfg, "training.epochs", default=0)

    if attn == "semoe":
        K = OmegaConf.select(cfg, "model.semoe.num_experts", default=4)
        method = f"K{K}"
    else:
        method = "STASA"

    return {
        "run_dir": run_dir,
        "dataset": str(dataset).upper(),
        "depth": f"{depth}L",
        "method": method,
        "target_epochs": int(target_epochs),
    }


def collect_runs(runs_dir: str) -> list[dict]:
    """Walk runs_dir, parse DONE runs, return list of rows with wall-clock."""
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*_seed*_2026*"))):
        if not os.path.isdir(run_dir):
            continue
        meta = _classify(run_dir)
        if meta is None:
            continue
        t = _sum_time_s(os.path.join(run_dir, "metrics.csv"))
        if t is None:
            continue
        n_epochs, total_s = t
        if meta["target_epochs"] > 0 and n_epochs < meta["target_epochs"]:
            # Skip abandoned/incomplete runs.
            continue
        rows.append({
            **meta,
            "n_epochs": n_epochs,
            "total_seconds": total_s,
            "total_minutes": total_s / 60,
            "total_hours": total_s / 3600,
            "seconds_per_epoch": total_s / n_epochs if n_epochs else 0,
        })
    return rows


def aggregate(rows: list[dict]) -> dict[tuple, dict]:
    """Group by (dataset, depth, method), return mean wall-clock + count."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["depth"], r["method"])
        groups[key].append(r)
    agg = {}
    for key, items in sorted(groups.items()):
        total_secs = [it["total_seconds"] for it in items]
        per_epoch = [it["seconds_per_epoch"] for it in items]
        n = len(items)
        agg[key] = {
            "n_runs": n,
            "mean_total_seconds": sum(total_secs) / n,
            "mean_total_minutes": sum(total_secs) / n / 60,
            "mean_total_hours": sum(total_secs) / n / 3600,
            "mean_seconds_per_epoch": sum(per_epoch) / n,
            "min_seconds": min(total_secs),
            "max_seconds": max(total_secs),
        }
    return agg


def emit_latex(agg: dict[tuple, dict]) -> str:
    """LaTeX booktabs table — training cost per (dataset, depth, method)."""
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Depth & Method & N runs & Sec / epoch & Wall-clock (h) \\",
        r"\midrule",
    ]
    for (ds, depth, method), info in agg.items():
        lines.append(
            f"{ds} & {depth} & {method} & "
            f"{info['n_runs']} & "
            f"{info['mean_seconds_per_epoch']:.1f} & "
            f"{info['mean_total_hours']:.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def emit_markdown(agg: dict[tuple, dict]) -> str:
    """Markdown summary table for inclusion in reports."""
    lines = [
        "| Dataset | Depth | Method | N runs | Sec/epoch | Wall-clock (h) | Range (h) |",
        "|---|---|---|---|---|---|---|",
    ]
    for (ds, depth, method), info in agg.items():
        lines.append(
            f"| {ds} | {depth} | {method} | "
            f"{info['n_runs']} | "
            f"{info['mean_seconds_per_epoch']:.1f} | "
            f"{info['mean_total_hours']:.2f} | "
            f"{info['min_seconds']/3600:.2f}–{info['max_seconds']/3600:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-tex", default="write/tables/table-training-cost.tex")
    parser.add_argument("--output-json", default="plans/reports/training-cost-260520.json")
    parser.add_argument("--print-md", action="store_true", help="Also print markdown to stdout")
    args = parser.parse_args()

    rows = collect_runs(args.runs_dir)
    print(f"Collected {len(rows)} DONE runs with wall-clock data")
    agg = aggregate(rows)

    tex = emit_latex(agg)
    md = emit_markdown(agg)

    Path(args.output_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_tex).write_text(tex)
    print(f"wrote: {args.output_tex}")

    # JSON for downstream tooling
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json_data = {
        f"{ds}/{depth}/{method}": info
        for (ds, depth, method), info in agg.items()
    }
    Path(args.output_json).write_text(json.dumps(json_data, indent=2))
    print(f"wrote: {args.output_json}")

    if args.print_md:
        print()
        print(md)


if __name__ == "__main__":
    main()
