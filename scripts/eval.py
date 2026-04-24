"""Evaluation script for a trained SpikCommander checkpoint.

Usage:
    python -m scripts.eval \\
        --checkpoint runs/shd_seed0_20260424_120000/best_acc.pt \\
        --config runs/shd_seed0_20260424_120000/config.yaml \\
        --dataset shd \\
        --split test

Prints top-1 accuracy and per-class accuracy.

NOTE: Firing rate, SOPs (syops-counter), and energy (paper Eq 19/20)
are stubbed as TODOs — not blocking for Phase 02 P1 gate.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from scommander.datasets import make_loaders, padded_sequence_mask
from scommander.models.registry import build_model
from scommander.losses.ce import SumSoftmaxCE, accuracy_from_logits
from scommander.utils.seed import set_seed

# Per-dataset class counts
_N_CLASSES = {"shd": 20, "ssc": 35, "gsc": 35}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a SpikCommander checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .pt state dict")
    p.add_argument("--config", required=True, help="Path to config.yaml (from run dir)")
    p.add_argument("--dataset", required=True, choices=["shd", "ssc", "gsc"])
    p.add_argument("--split", default="test", choices=["test", "val"],
                   help="Which split to evaluate (default: test)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.dataset.name = args.dataset

    set_seed(int(cfg.experiment.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = _N_CLASSES[args.dataset]

    # Load checkpoint
    state = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Select loader
    loaders = make_loaders(args.dataset, cfg)
    if args.dataset == "shd":
        _, test_loader = loaders
        loader = test_loader
    else:
        _, val_loader, test_loader = loaders
        loader = test_loader if args.split == "test" else val_loader

    loss_fn = SumSoftmaxCE()

    all_preds, all_targets = [], []
    loss_batch = []

    with torch.no_grad():
        for x, y, x_len in tqdm(loader, desc="eval"):
            attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)
            x = x.float().to(device)
            y_int = y.long().to(device)
            y_onehot = F.one_hot(y_int, n_classes).float()

            logits = model(x, attn_mask)                   # (T, B, C)
            loss = loss_fn(logits, y_onehot)
            loss_batch.append(loss.cpu().item())

            # Aggregate over time for predictions
            m = torch.sum(F.softmax(logits, dim=-1), dim=0)  # (B, C)
            preds = m.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_targets.append(y_int.cpu())

            model.reset()

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    top1_acc = (all_preds == all_targets).float().mean().item()
    mean_loss = float(np.mean(loss_batch))

    print(f"\n=== Evaluation ({args.split}) ===")
    print(f"Top-1 Accuracy : {100 * top1_acc:.2f}%")
    print(f"Mean Loss      : {mean_loss:.4f}")

    # Per-class accuracy
    print(f"\nPer-class accuracy ({n_classes} classes):")
    for c in range(n_classes):
        mask = all_targets == c
        if mask.sum() == 0:
            print(f"  Class {c:2d}: N/A (no samples)")
            continue
        class_acc = (all_preds[mask] == c).float().mean().item()
        print(f"  Class {c:2d}: {100 * class_acc:.1f}%  (n={mask.sum().item()})")

    # TODO (Phase 02 P1, non-blocking): firing rate per layer
    # TODO (Phase 02 P1, non-blocking): SOPs via syops-counter
    # TODO (Phase 02 P1, non-blocking): energy via paper Eq 19/20


if __name__ == "__main__":
    main()
