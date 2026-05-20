"""Paired t-test K2 (SeMoE) vs STASA baseline per (dataset, depth).

Reads per-seed test accuracies hardcoded from Phase 05 measurement logs +
SHD 3-axis report. Computes paired t-test (scipy.stats.ttest_rel) since
seeds are matched across variants.

Outputs:
    - Console table: variant, n, mean Δ, t-stat, p-value, p<0.05?
    - Markdown report at plans/reports/results-{date}-paired-t-test.md

Usage:
    python -m scripts.paired_t_test [--output-md plans/reports/results-260520-paired-t-test.md]
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import NamedTuple

import numpy as np
from scipy import stats


class PerSeed(NamedTuple):
    dataset: str
    depth: str
    stasa: list[float]
    k2: list[float]
    source: str  # log file or report


# Per-seed test accuracies, seeds {0,1,2} paired across STASA and K2.
# Sources: logs/measure-260520.log + logs/measure-stasa-260520.log + results-260515-1240-shd-3axis.md
DATA: list[PerSeed] = [
    PerSeed(
        dataset="SHD", depth="1L",
        stasa=[96.07, 95.27, 95.01],
        k2=[96.38, 95.58, 96.16],
        source="results-260515-1240-shd-3axis.md",
    ),
    PerSeed(
        dataset="SSC", depth="2L",
        stasa=[83.44, 83.33, 83.62],
        k2=[82.62, 82.63, 82.95],
        source="logs/measure-{260520,stasa-260520}.log",
    ),
    PerSeed(
        dataset="SSC", depth="1L",
        stasa=[83.40, 83.03, 83.17],
        k2=[82.52, 82.99, 82.46],
        source="logs/measure-{260520,stasa-260520}.log",
    ),
    PerSeed(
        dataset="GSC", depth="2L",
        stasa=[96.82, 96.81, 96.48],
        k2=[96.57, 96.49, 96.26],
        source="logs/measure-{260520,stasa-260520}.log",
    ),
    PerSeed(
        dataset="GSC", depth="1L",
        stasa=[96.50, 96.49],   # paired with K2 n=2 (seed 2 skipped per scope cut)
        k2=[96.15, 96.37],
        source="logs/measure-{260520,stasa-260520}.log (K2 seed 2 skipped)",
    ),
]


def _ttest(stasa: list[float], k2: list[float]) -> tuple[float, float, float]:
    """Paired t-test K2 − STASA. Returns (mean_delta, t_stat, p_value)."""
    s = np.array(stasa, dtype=float)
    k = np.array(k2, dtype=float)
    delta = k - s
    if len(delta) < 2:
        return float(delta.mean()), float("nan"), float("nan")
    t_stat, p_val = stats.ttest_rel(k, s)
    return float(delta.mean()), float(t_stat), float(p_val)


def _format_console(rows: list[PerSeed]) -> str:
    lines = [
        f"{'Dataset':<8}{'Depth':<6}{'n':<4}{'STASA mean':<12}{'K2 mean':<10}"
        f"{'Δ mean':<10}{'t':<10}{'p-value':<12}{'p<0.05?':<8}"
    ]
    lines.append("-" * 80)
    for r in rows:
        mean_d, t, p = _ttest(r.stasa, r.k2)
        sig = "yes" if (not np.isnan(p)) and p < 0.05 else "no"
        lines.append(
            f"{r.dataset:<8}{r.depth:<6}{len(r.stasa):<4}"
            f"{np.mean(r.stasa):<12.3f}{np.mean(r.k2):<10.3f}"
            f"{mean_d:<+10.3f}{t:<+10.3f}{p:<12.4f}{sig:<8}"
        )
    return "\n".join(lines)


def _format_markdown(rows: list[PerSeed]) -> str:
    today = date.today().strftime("%Y-%m-%d")
    lines = [
        "---",
        "type: results",
        f"date: {today}",
        "slug: paired-t-test-k2-vs-stasa",
        "status: FINAL — Phase 07 statistical significance for paper",
        "---",
        "",
        "# Paired t-test — SeMoE K2 vs STASA baseline (per dataset × depth)",
        "",
        "Pairs are matched on seed ID {0,1,2} across both variants. Reported is "
        "`scipy.stats.ttest_rel(k2_per_seed, stasa_per_seed)`. Delta = K2 − STASA, "
        "negative means K2 lost accuracy. Significance threshold: p<0.05.",
        "",
        "## Result table",
        "",
        "| Dataset | Depth | n | STASA mean | K2 mean | Δ mean (K2−STASA) | t-stat | p-value | p<0.05? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mean_d, t, p = _ttest(r.stasa, r.k2)
        sig = "✅ yes" if (not np.isnan(p)) and p < 0.05 else "no"
        p_str = f"{p:.4f}" if not np.isnan(p) else "n/a"
        t_str = f"{t:+.3f}" if not np.isnan(t) else "n/a"
        lines.append(
            f"| {r.dataset} | {r.depth} | {len(r.stasa)} | "
            f"{np.mean(r.stasa):.3f} | {np.mean(r.k2):.3f} | {mean_d:+.3f} | "
            f"{t_str} | {p_str} | {sig} |"
        )
    lines += [
        "",
        "## Interpretation for paper",
        "",
        "**SHD 1L**: K2 is the only cell where K2 nominally beats STASA (mean Δ > 0). "
        "Whether p<0.05 holds depends on small-n variance; report as 'matching or "
        "exceeding baseline with compression+energy wins'.",
        "",
        "**SSC/GSC**: STASA wins acc by 0.2-0.7pp (small but consistent). "
        "Per Phase 05 framing (compression+energy hero with small acc trade), the "
        "paired t-test will likely show **STASA significantly better on acc** for "
        "SSC/GSC at p<0.05. This is **expected and acceptable** — the paper "
        "headlines params/energy reduction, not acc dominance, on these datasets.",
        "",
        "**Statistical caveat**: n=3 (or 2 for GSC-1L) seeds → low statistical "
        "power; even meaningful effects may not reach p<0.05. Report p-values "
        "honestly in paper; emphasize effect-size magnitudes (Δ in pp) alongside "
        "Pareto improvements on params/energy axes.",
        "",
        "## Per-seed data",
        "",
        "| Dataset | Depth | Seed | STASA | K2 | Δ |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        for i, (s, k) in enumerate(zip(r.stasa, r.k2)):
            lines.append(f"| {r.dataset} | {r.depth} | {i} | {s:.2f} | {k:.2f} | {k-s:+.2f} |")
    lines += [
        "",
        "## Sources",
        "",
    ]
    for r in rows:
        lines.append(f"- {r.dataset} {r.depth}: `{r.source}`")
    lines += [
        "",
        "## Unresolved questions",
        "",
        "- For GSC-1L (n=2), Welch's t-test or Wilcoxon signed-rank may be preferred "
        "given degenerate paired-sample sizes. Current report uses ttest_rel for "
        "consistency.",
        "- Phase 02 used 5-seed for STASA SHD; paired test here truncates to 3-seed "
        "to match K2 sample size. Effect-size comparable; power penalty noted.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional path to write markdown report (e.g. plans/reports/results-260520-paired-t-test.md)",
    )
    args = parser.parse_args()

    print(_format_console(DATA))

    if args.output_md:
        md = _format_markdown(DATA)
        with open(args.output_md, "w") as f:
            f.write(md)
        print(f"\nWrote markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
