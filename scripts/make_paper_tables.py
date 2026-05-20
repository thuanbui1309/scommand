"""Aggregate Phase 05 measurement logs + reports into paper LaTeX tables + master CSV.

Generates (under repo root):
    write/tables/table-01-sota.tex          — main SOTA (SHD/SSC/GSC, acc/params/energy)
    write/tables/table-02-efficiency.tex    — full efficiency (params/FR/SOPs/energy/Δ%)
    write/tables/table-04-ablation-k-sweep.tex — SHD K-sweep + identity/LB/fulld variants
    write/tables/table-09-seed-std.tex      — per-seed std for all hero cells
    write/tables/table-noise-robustness.tex — GSC-2L AWGN SNR sweep
    plans/reports/runs-master-csv-260520.csv — per-seed master CSV

Data sources (hardcoded — all from committed Phase 05 reports + measurement logs):
    - SHD K2/STASA per-seed: results-260515-1240-shd-3axis.md
    - SSC/GSC K2 + STASA: logs/measure-260520.log + measure-stasa-260520.log
    - Noise: logs/noise-260520.log (and results-260520-noise-robustness-gsc.md)
    - K-sweep SHD: status_report (semoe/shd/1L grouped)

Usage:
    python -m scripts.make_paper_tables
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Hero cells: per-seed (acc, params, eff_FR, sops_g, energy_mJ).
# Sources documented per cell.
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    dataset: str
    depth: str            # "1L" or "2L"
    method: str           # "STASA" or "K2"
    per_seed_acc: list[float]
    params: int
    eff_fr_mean: float
    sops_g_mean: float
    energy_mj_mean: float


HEROES: list[Cell] = [
    # SHD (results-260515-1240-shd-3axis.md)
    Cell("SHD", "1L", "STASA",
         per_seed_acc=[96.07, 95.27, 95.01],
         params=190_664, eff_fr_mean=0.187, sops_g_mean=0.0036, energy_mj_mean=0.00319),
    Cell("SHD", "1L", "K2",
         per_seed_acc=[96.38, 95.58, 96.16],
         params=137_666, eff_fr_mean=0.136, sops_g_mean=0.0018, energy_mj_mean=0.00161),

    # SSC 2L (logs/measure-stasa-260520.log + measure-260520.log)
    Cell("SSC", "2L", "STASA",
         per_seed_acc=[83.44, 83.33, 83.62],
         params=2_127_392, eff_fr_mean=0.1356, sops_g_mean=0.0288, energy_mj_mean=0.02591),
    Cell("SSC", "2L", "K2",
         per_seed_acc=[82.62, 82.63, 82.95],
         params=1_702_148, eff_fr_mean=0.0914, sops_g_mean=0.0151, energy_mj_mean=0.01359),

    # SSC 1L
    Cell("SSC", "1L", "STASA",
         per_seed_acc=[83.40, 83.03, 83.17],
         params=1_120_400, eff_fr_mean=0.1339, sops_g_mean=0.01497, energy_mj_mean=0.01346),
    Cell("SSC", "1L", "K2",
         per_seed_acc=[82.52, 82.99, 82.46],
         params=907_778, eff_fr_mean=0.1172, sops_g_mean=0.0106, energy_mj_mean=0.00945),

    # GSC 2L
    Cell("GSC", "2L", "STASA",
         per_seed_acc=[96.82, 96.81, 96.48],
         params=2_127_392, eff_fr_mean=0.1718, sops_g_mean=0.0368, energy_mj_mean=0.03396),
    Cell("GSC", "2L", "K2",
         per_seed_acc=[96.57, 96.49, 96.26],
         params=1_702_148, eff_fr_mean=0.1207, sops_g_mean=0.0200, energy_mj_mean=0.01883),

    # GSC 1L (K2 n=2; STASA n=3)
    Cell("GSC", "1L", "STASA",
         per_seed_acc=[96.50, 96.49, 96.39],
         params=1_120_400, eff_fr_mean=0.1644, sops_g_mean=0.0185, energy_mj_mean=0.01750),
    Cell("GSC", "1L", "K2",
         per_seed_acc=[96.15, 96.37],
         params=907_778, eff_fr_mean=0.1171, sops_g_mean=0.0104, energy_mj_mean=0.01016),
]


# K-sweep ablation on SHD (1L) — mean acc per K-variant from Phase 05 status report.
K_SWEEP_SHD = [
    ("K2 [SWA, identity]", 96.05, "hero"),
    ("K3 [SWA, LRA, smallSWA] no-id", 95.14, "no-identity ablation"),
    ("K4 [SWA, LRA, smallSWA, id]", 95.21, "spec default"),
    ("K4 fulld (no expert compression)", 95.56, "expert width ablation"),
    ("K4 no load-balance (λ=0)", 95.33, "LB-weight ablation"),
    ("K6 [SWA, LRA, smallSWA, id, +2]", 95.67, "extra experts"),
    ("STASA baseline", 95.45, "non-routing baseline"),
]


# Noise robustness from logs/noise-260520.log
NOISE_DATA = {
    "K2-2L": {
        "clean": [96.57, 96.49, 96.26],
        "20":    [96.40, 96.39, 96.25],
        "10":    [94.41, 94.64, 94.34],
        "5":     [86.42, 86.35, 87.05],
        "0":     [59.38, 58.51, 62.10],
    },
    "STASA-2L": {
        "clean": [96.82, 96.81, 96.48],
        "20":    [96.77, 96.54, 96.71],
        "10":    [94.94, 94.75, 95.14],
        "5":     [87.91, 86.72, 88.76],
        "0":     [62.34, 56.84, 62.14],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def fmt(value: float, prec: int = 2) -> str:
    return f"{value:.{prec}f}"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"wrote: {path}")


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------


def table_01_sota() -> str:
    """Main SOTA: SHD/SSC/GSC × {STASA, K2} with mean ± std acc, params, energy."""
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Depth & Method & Acc (\%) & Params & Energy (mJ) & \(\Delta\) Energy \\",
        r"\midrule",
    ]
    for ds in ["SHD", "SSC", "GSC"]:
        for depth in ["1L", "2L"]:
            stasa = next((c for c in HEROES if c.dataset == ds and c.depth == depth and c.method == "STASA"), None)
            k2 = next((c for c in HEROES if c.dataset == ds and c.depth == depth and c.method == "K2"), None)
            if not stasa or not k2:
                continue
            for c in (stasa, k2):
                m, s = mean(c.per_seed_acc), std(c.per_seed_acc)
                line = (
                    f"{ds} & {depth} & {c.method} & "
                    f"{fmt(m)} \\(\\pm\\) {fmt(s)} & "
                    f"{c.params:,} & "
                    f"{c.energy_mj_mean:.5f} & "
                )
                if c.method == "K2":
                    delta_pct = -100 * (stasa.energy_mj_mean - c.energy_mj_mean) / stasa.energy_mj_mean
                    line += f"{delta_pct:+.1f}\\% \\\\"
                else:
                    line += r"--- \\"
                lines.append(line)
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def table_02_efficiency() -> str:
    """Full efficiency: params, eff_FR, SOPs, energy per cell with %Δ vs STASA."""
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Depth & Method & Params (M) & Eff FR & SOPs (G) & Energy (mJ) & \(\Delta\) E (\%) \\",
        r"\midrule",
    ]
    for ds in ["SHD", "SSC", "GSC"]:
        for depth in ["1L", "2L"]:
            stasa = next((c for c in HEROES if c.dataset == ds and c.depth == depth and c.method == "STASA"), None)
            k2 = next((c for c in HEROES if c.dataset == ds and c.depth == depth and c.method == "K2"), None)
            if not stasa or not k2:
                continue
            for c in (stasa, k2):
                de = ""
                if c.method == "K2":
                    de = f"{-100*(stasa.energy_mj_mean - c.energy_mj_mean)/stasa.energy_mj_mean:+.1f}"
                lines.append(
                    f"{ds} & {depth} & {c.method} & "
                    f"{c.params/1e6:.3f} & "
                    f"{c.eff_fr_mean:.4f} & "
                    f"{c.sops_g_mean:.4f} & "
                    f"{c.energy_mj_mean:.5f} & "
                    f"{de} \\\\"
                )
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def table_04_ablation() -> str:
    """SHD K-sweep + identity/LB/fulld variants. Mean acc only (per-seed std elsewhere)."""
    lines = [
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Variant & Acc (\%) & Role \\",
        r"\midrule",
    ]
    for name, acc, role in K_SWEEP_SHD:
        lines.append(f"{name} & {fmt(acc)} & {role} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def table_09_seed_std() -> str:
    """Per-cell mean ± std (separates STASA / K2 std for reviewer scrutiny)."""
    lines = [
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Dataset & Depth & Method & n & Acc mean \(\pm\) std \\",
        r"\midrule",
    ]
    for c in HEROES:
        m, s = mean(c.per_seed_acc), std(c.per_seed_acc)
        lines.append(
            f"{c.dataset} & {c.depth} & {c.method} & "
            f"{len(c.per_seed_acc)} & "
            f"{fmt(m)} \\(\\pm\\) {fmt(s)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def table_noise() -> str:
    """GSC-2L noise robustness (mean ± std per SNR)."""
    snrs = ["clean", "20", "10", "5", "0"]
    lines = [
        r"\begin{tabular}{l" + "r" * len(snrs) + "}",
        r"\toprule",
        r"Method & " + " & ".join([f"SNR {s}{'~dB' if s != 'clean' else ''}" for s in snrs]) + r" \\",
        r"\midrule",
    ]
    for method_key in ["STASA-2L", "K2-2L"]:
        cells = [
            f"{mean(NOISE_DATA[method_key][s]):.2f} \\(\\pm\\) {std(NOISE_DATA[method_key][s]):.2f}"
            for s in snrs
        ]
        lines.append(f"{method_key} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def master_csv() -> list[dict]:
    """Per-seed master rows for runs.csv."""
    rows = []
    for c in HEROES:
        for seed, acc in enumerate(c.per_seed_acc):
            rows.append({
                "dataset": c.dataset,
                "depth": c.depth,
                "method": c.method,
                "seed": seed,
                "test_acc": f"{acc:.2f}",
                "params": c.params,
                "eff_fr_mean": f"{c.eff_fr_mean:.4f}",
                "sops_g_mean": f"{c.sops_g_mean:.4f}",
                "energy_mj_mean": f"{c.energy_mj_mean:.5f}",
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Resolve repo root (script lives at scripts/ or src/scripts/).
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent if (here.parent.name == "src") else here.parent

    write_text(repo_root / "write/tables/table-01-sota.tex", table_01_sota())
    write_text(repo_root / "write/tables/table-02-efficiency.tex", table_02_efficiency())
    write_text(repo_root / "write/tables/table-04-ablation-k-sweep.tex", table_04_ablation())
    write_text(repo_root / "write/tables/table-09-seed-std.tex", table_09_seed_std())
    write_text(repo_root / "write/tables/table-noise-robustness.tex", table_noise())

    csv_path = repo_root / "plans/reports/runs-master-csv-260520.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        rows = master_csv()
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
