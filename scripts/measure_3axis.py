"""Measure accuracy + param count + mean firing rate for a trained checkpoint.

Usage:
    python -m scripts.measure_3axis \\
        --checkpoint runs/<run_dir>/best_acc.pt \\
        --config runs/<run_dir>/config.yaml \\
        --dataset shd \\
        --split test

Outputs CSV line: variant,acc,params,mean_fr (printed to stdout).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from scommander.datasets import make_loaders, padded_sequence_mask
from scommander.losses.sparsity import FiringRateCollector
from scommander.models.registry import build_model
from scommander.modules.lif import LIFNode
from scommander.modules.semoe import SeMoEBlock, collect_semoe_expert_usage
from scommander.utils.seed import set_seed


_N_CLASSES = {"shd": 20, "ssc": 35, "gsc": 35}


def _build_routing_map(model: torch.nn.Module) -> dict[str, tuple[str, int]]:
    """Return {full_layer_name: (semoe_block_name, expert_idx)} for every LIF
    layer that lives inside a SeMoE expert. Layers outside any SeMoE block
    (gate, input/out projections, MLP, SEE) are absent — they always count
    fully toward effective FR.
    """
    out: dict[str, tuple[str, int]] = {}
    for block_name, block in model.named_modules():
        if not isinstance(block, SeMoEBlock):
            continue
        for expert_idx, expert in enumerate(block.experts):
            for sub_name, sub in expert.named_modules():
                if not isinstance(sub, LIFNode):
                    continue
                # Reconstruct name as it appears in named_modules() at model level
                expert_path = f"{block_name}.experts.{expert_idx}"
                full = f"{expert_path}.{sub_name}" if sub_name else expert_path
                out[full] = (block_name, expert_idx)
    return out


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True, choices=["shd", "ssc", "gsc"])
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--variant", default="", help="Label for output (e.g. c1-c2-sparse-full)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.name = args.dataset
    set_seed(int(cfg.experiment.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = _N_CLASSES[args.dataset]

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loaders = make_loaders(args.dataset, cfg)
    if args.dataset == "shd":
        _, test_loader = loaders
        loader = test_loader
    else:
        _, val_loader, test_loader = loaders
        loader = test_loader if args.split == "test" else val_loader

    routing_map = _build_routing_map(model)

    all_preds, all_targets = [], []
    fr_layer_accumulator: dict[str, list[float]] = {}
    usage_acc: dict[str, torch.Tensor] = {}
    n_batches = 0

    with torch.no_grad():
        for x, y, x_len in tqdm(loader, desc=f"{args.variant or 'eval'}", leave=False):
            attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)
            x = x.float().to(device)
            y_int = y.long().to(device)

            with FiringRateCollector(model) as fr:
                logits = model(x, attn_mask)
            for name, rate in fr.spike_rates.items():
                fr_layer_accumulator.setdefault(name, []).append(rate.item())

            # Per-block expert usage (no-op when no SeMoE block present)
            for bname, vec in collect_semoe_expert_usage(model).items():
                if bname not in usage_acc:
                    usage_acc[bname] = torch.zeros_like(vec)
                usage_acc[bname] += vec
            n_batches += 1

            m = torch.sum(F.softmax(logits, dim=-1), dim=0)
            preds = m.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_targets.append(y_int.cpu())
            model.reset()

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = (all_preds == all_targets).float().mean().item()

    layer_fr = {k: float(np.mean(v)) for k, v in fr_layer_accumulator.items()}
    raw_mean_fr = float(np.mean(list(layer_fr.values()))) if layer_fr else 0.0

    # Routing-aware effective FR: weight each expert layer by its block's
    # mean usage probability. Non-routed layers count fully. Without SeMoE
    # blocks this is identical to raw_mean_fr.
    mean_usage = {bn: (vec / max(n_batches, 1)).cpu().tolist() for bn, vec in usage_acc.items()}
    effective_layer_fr: dict[str, float] = {}
    for name, fr in layer_fr.items():
        if name in routing_map:
            bn, eidx = routing_map[name]
            u = mean_usage.get(bn, [1.0] * (eidx + 1))[eidx]
            effective_layer_fr[name] = fr * u
        else:
            effective_layer_fr[name] = fr
    effective_mean_fr = float(np.mean(list(effective_layer_fr.values()))) if effective_layer_fr else 0.0

    label = args.variant or os.path.basename(os.path.dirname(args.checkpoint))
    print(f"variant,acc,params,raw_mean_fr,effective_mean_fr,n_layers")
    print(f"{label},{100*acc:.2f},{n_params},{raw_mean_fr:.4f},{effective_mean_fr:.4f},{len(layer_fr)}")

    if mean_usage:
        print("\n# Mean expert usage (routing fractions):")
        for bn, vec in sorted(mean_usage.items()):
            print(f"#   {bn}: {[f'{u:.3f}' for u in vec]}")

    # Per-layer FR breakdown (raw, with routing factor inline when applicable)
    print("\n# Per-layer FR (raw  /  effective when routed):")
    for name in sorted(layer_fr):
        raw = layer_fr[name]
        eff = effective_layer_fr[name]
        if name in routing_map:
            bn, eidx = routing_map[name]
            print(f"#   {name}: {raw:.4f}  /  {eff:.4f}  (×u_{eidx}={mean_usage[bn][eidx]:.3f})")
        else:
            print(f"#   {name}: {raw:.4f}")


if __name__ == "__main__":
    main()
