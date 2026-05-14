"""SeMoE post-hoc expert pruning + re-evaluation.

Loads a trained SeMoE checkpoint, measures per-expert usage on a target split,
identifies experts whose usage falls below ``--threshold`` (default 0.05),
disables them, and reports:
  - top-1 accuracy before / after pruning
  - parameter count before / after (dropping dead-expert parameters)
  - per-block expert usage table

Pruning is implemented by replacing each dead expert with a zero-output stub —
the gate's argmax can still pick that index, in which case the routed output
is zero for that timestep. We then recompute "effective params" by subtracting
the dropped expert parameter counts.

Usage:
    python -m scripts.prune_semoe \\
        --checkpoint runs/shd_seed0_.../best_acc.pt \\
        --config     runs/shd_seed0_.../config.yaml \\
        --dataset shd --split test [--threshold 0.05]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from scommander.datasets import make_loaders, padded_sequence_mask
from scommander.losses.ce import SumSoftmaxCE
from scommander.models.registry import build_model
from scommander.modules.semoe import SeMoEBlock, collect_semoe_expert_usage
from scommander.utils.seed import set_seed

_N_CLASSES = {"shd": 20, "ssc": 35, "gsc": 35}


class _ZeroExpert(nn.Module):
    """Stub replacement for a pruned expert. Returns zeros shaped like the original."""

    def __init__(self, expert_dim: int) -> None:
        super().__init__()
        self.expert_dim = expert_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _ = x.shape
        return x.new_zeros((T, B, self.expert_dim))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc SeMoE expert pruning")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True, choices=["shd", "ssc", "gsc"])
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Drop experts with mean usage below this fraction (default 0.05)")
    return p.parse_args()


def _measure_usage(model: nn.Module, loader, device: torch.device) -> Dict[str, torch.Tensor]:
    """Average per-block expert usage across the loader."""
    acc: Dict[str, torch.Tensor] = {}
    n = 0
    model.eval()
    with torch.no_grad():
        for x, y, x_len in tqdm(loader, desc="usage"):
            attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)
            x = x.float().to(device)
            _ = model(x, attn_mask)
            for name, vec in collect_semoe_expert_usage(model).items():
                if name not in acc:
                    acc[name] = torch.zeros_like(vec)
                acc[name] += vec
            n += 1
            model.reset()
    return {k: v / max(n, 1) for k, v in acc.items()}


def _evaluate(model: nn.Module, loader, n_classes: int, device: torch.device) -> tuple[float, float]:
    """Return (top1_acc, mean_loss)."""
    loss_fn = SumSoftmaxCE()
    preds, targets, losses = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, x_len in tqdm(loader, desc="eval"):
            attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)
            x = x.float().to(device)
            y_int = y.long().to(device)
            y_onehot = F.one_hot(y_int, n_classes).float()
            logits = model(x, attn_mask)
            losses.append(loss_fn(logits, y_onehot).cpu().item())
            m = torch.sum(F.softmax(logits, dim=-1), dim=0)
            preds.append(m.argmax(dim=-1).cpu())
            targets.append(y_int.cpu())
            model.reset()
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return (preds == targets).float().mean().item(), float(np.mean(losses))


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _prune_block(block: SeMoEBlock, dead_idx: list[int]) -> int:
    """Replace dead experts with zero stubs. Returns number of params dropped."""
    dropped = 0
    for k in dead_idx:
        old = block.experts[k]
        dropped += sum(p.numel() for p in old.parameters() if p.requires_grad)
        block.experts[k] = _ZeroExpert(block.expert_dim).to(
            next(block.gate_linear.parameters()).device
        )
    return dropped


def main() -> None:
    args = _parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.dataset.name = args.dataset
    set_seed(int(cfg.experiment.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = _N_CLASSES[args.dataset]

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    semoe_blocks = [(name, m) for name, m in model.named_modules() if isinstance(m, SeMoEBlock)]
    if not semoe_blocks:
        print("ERROR: no SeMoE blocks in this checkpoint. Run on a SeMoE config.")
        sys.exit(1)

    loaders = make_loaders(args.dataset, cfg)
    if args.dataset == "shd":
        _, test_loader = loaders
        loader = test_loader
    else:
        _, val_loader, test_loader = loaders
        loader = test_loader if args.split == "test" else val_loader

    # ── Stage 1: usage + baseline acc ────────────────────────────────────────
    print("\n=== Stage 1: measure expert usage ===")
    usage = _measure_usage(model, loader, device)
    for name, vec in usage.items():
        formatted = ", ".join(f"e{k}={v:.3f}" for k, v in enumerate(vec.tolist()))
        print(f"  {name}: {formatted}")

    acc0, loss0 = _evaluate(model, loader, n_classes, device)
    params0 = _count_params(model)
    print(f"\nBaseline ({args.split}): acc={100*acc0:.2f}%  loss={loss0:.4f}  params={params0:,}")

    # ── Stage 2: prune ───────────────────────────────────────────────────────
    print(f"\n=== Stage 2: prune experts with usage < {args.threshold} ===")
    total_dropped = 0
    pruning_summary = []
    for name, block in semoe_blocks:
        vec = usage[name]
        dead = [k for k, v in enumerate(vec.tolist()) if v < args.threshold]
        if not dead:
            print(f"  {name}: no experts below threshold")
            continue
        dropped = _prune_block(block, dead)
        total_dropped += dropped
        pruning_summary.append((name, dead, dropped))
        print(f"  {name}: dropped experts {dead} (-{dropped:,} params)")

    if total_dropped == 0:
        print("All experts above threshold — nothing to prune.")
        return

    # ── Stage 3: re-eval pruned model ────────────────────────────────────────
    print("\n=== Stage 3: re-eval after pruning ===")
    acc1, loss1 = _evaluate(model, loader, n_classes, device)
    params1 = params0 - total_dropped
    print(f"\nPruned ({args.split}): acc={100*acc1:.2f}%  loss={loss1:.4f}  params={params1:,}")
    print(f"Δ acc: {100*(acc1-acc0):+.2f} pp  |  Δ params: {-total_dropped:+,} ({-100*total_dropped/params0:+.1f}%)")

    print("\nSummary:")
    for name, dead, dropped in pruning_summary:
        print(f"  {name}: -{len(dead)} experts ({dropped:,} params)")


if __name__ == "__main__":
    main()
