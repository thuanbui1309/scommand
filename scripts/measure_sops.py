"""Measure SOPs (synaptic operations) and theoretical energy in mJ for a
trained checkpoint, accounting for SeMoE routing.

Approach (per paper Appendix E):
  - SOP_layer = FR_input × FLOP_layer × routing_weight
  - Energy   = E_AC * Σ SOPs (spike inputs) + E_MAC * first_conv_FLOPs (real input on GSC)
  - 45nm hardware: E_AC = 0.9 pJ, E_MAC = 4.6 pJ

Routing: for any Conv/Linear layer that lives inside a SeMoE expert, multiply
its SOPs by the block's mean usage probability for that expert. Non-routed
layers (gate, input/out projection, MLP, SEE) count fully.

Input-FR proxy: per layer we use the overall ``effective_mean_fr`` of the
model (already routing-weighted) — same simplification used in measure_3axis.
This collapses per-layer causality but matches what the paper's syops-counter
ultimately reports as a single energy number.

Usage:
    python -m scripts.measure_sops \\
        --checkpoint runs/<run>/best_acc.pt \\
        --config runs/<run>/config.yaml \\
        --dataset shd
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
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

# Paper Appendix E, 45 nm hardware (Horowitz 2014)
_E_AC_PJ = 0.9
_E_MAC_PJ = 4.6


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOPs + theoretical energy measurement")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True, choices=["shd", "ssc", "gsc"])
    p.add_argument("--variant", default="", help="Output label")
    return p.parse_args()


def _build_routing_map(model: nn.Module) -> dict[str, tuple[str, int]]:
    """Map any submodule under a SeMoE expert to (block_name, expert_idx)."""
    out: dict[str, tuple[str, int]] = {}
    for block_name, block in model.named_modules():
        if not isinstance(block, SeMoEBlock):
            continue
        for expert_idx, expert in enumerate(block.experts):
            for sub_name, _sub in expert.named_modules():
                expert_path = f"{block_name}.experts.{expert_idx}"
                full = f"{expert_path}.{sub_name}" if sub_name else expert_path
                out[full] = (block_name, expert_idx)
    return out


def _flops_for(layer: nn.Module, output_shape: torch.Size, batch_size: int) -> int:
    """Per-sample FLOPs for one forward pass through this layer.

    Output shape is the forward output; we strip batch and any leading
    time/multistep axes by dividing by ``batch_size`` × extra leading dims.
    """
    if isinstance(layer, nn.Conv1d):
        # output_shape: (..., C_out, L) — collapse all leading dims (T, B, ...) into 1
        L = output_shape[-1]
        Cin = layer.in_channels // layer.groups
        Cout = layer.out_channels
        K = layer.kernel_size[0]
        per_step = Cin * Cout * K * L
        # Account for any extra leading dim beyond batch (e.g. T in step_mode='m')
        leading = int(np.prod(output_shape[:-2])) if len(output_shape) > 2 else 1
        return per_step * leading // batch_size
    if isinstance(layer, nn.Conv2d):
        H, W = output_shape[-2], output_shape[-1]
        Cin = layer.in_channels // layer.groups
        Cout = layer.out_channels
        Kh, Kw = layer.kernel_size
        per_step = Cin * Cout * Kh * Kw * H * W
        leading = int(np.prod(output_shape[:-3])) if len(output_shape) > 3 else 1
        return per_step * leading // batch_size
    if isinstance(layer, nn.Linear):
        # output_shape: (..., out_features); per-sample FLOPs = product(spatial dims) × in × out
        spatial = int(np.prod(output_shape[:-1]))
        return spatial * layer.in_features * layer.out_features // batch_size
    return 0


def _capture_flops_one_batch(model: nn.Module, x: torch.Tensor, attn_mask: torch.Tensor,
                              batch_size: int) -> dict[str, int]:
    """Return {layer_name: per-sample FLOPs} via forward hooks."""
    flops_table: dict[str, int] = {}
    handles: list[Any] = []

    def make_hook(name: str, mod: nn.Module):
        def fn(_module, _input, output):
            if isinstance(output, torch.Tensor):
                flops_table[name] = _flops_for(mod, output.shape, batch_size)
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(make_hook(name, mod)))

    with torch.no_grad():
        _ = model(x, attn_mask)
    model.reset()

    for h in handles:
        h.remove()
    return flops_table


def main() -> None:
    args = _parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.name = args.dataset
    set_seed(int(cfg.experiment.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = _N_CLASSES[args.dataset]

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    routing_map = _build_routing_map(model)

    # Pick test loader
    loaders = make_loaders(args.dataset, cfg)
    if args.dataset == "shd":
        _, test_loader = loaders
    else:
        _, _, test_loader = loaders

    # Pass 1: capture per-layer FLOPs from one batch (shapes don't depend on data values)
    flops_per_layer: dict[str, int] = {}
    first_batch = next(iter(test_loader))
    x, _y, x_len = first_batch
    attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)
    x_dev = x.float().to(device)
    flops_per_layer = _capture_flops_one_batch(model, x_dev, attn_mask, batch_size=x.shape[0])

    # Pass 2: full eval — accumulate per-layer FR, per-block usage, accuracy
    fr_acc: dict[str, list[float]] = {}
    usage_acc: dict[str, torch.Tensor] = {}
    n_batches = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y, x_len in tqdm(test_loader, desc="eval+fr+usage", leave=False):
            attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)
            x = x.float().to(device)
            y_int = y.long().to(device)
            with FiringRateCollector(model) as fr:
                logits = model(x, attn_mask)
            for name, rate in fr.spike_rates.items():
                fr_acc.setdefault(name, []).append(rate.item())
            for bn, vec in collect_semoe_expert_usage(model).items():
                if bn not in usage_acc:
                    usage_acc[bn] = torch.zeros_like(vec)
                usage_acc[bn] += vec
            n_batches += 1
            m = torch.sum(F.softmax(logits, dim=-1), dim=0)
            all_preds.append(m.argmax(dim=-1).cpu())
            all_targets.append(y_int.cpu())
            model.reset()

    layer_fr = {k: float(np.mean(v)) for k, v in fr_acc.items()}
    mean_usage = {bn: (vec / max(n_batches, 1)).cpu().tolist() for bn, vec in usage_acc.items()}
    raw_mean_fr = float(np.mean(list(layer_fr.values()))) if layer_fr else 0.0

    # Effective FR — same definition as measure_3axis: routed expert FRs weighted by usage
    effective_layer_fr: dict[str, float] = {}
    for name, fr in layer_fr.items():
        if name in routing_map:
            bn, eidx = routing_map[name]
            u = mean_usage.get(bn, [1.0] * (eidx + 1))[eidx]
            effective_layer_fr[name] = fr * u
        else:
            effective_layer_fr[name] = fr
    effective_mean_fr = float(np.mean(list(effective_layer_fr.values()))) if effective_layer_fr else 0.0

    # Accuracy
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = (all_preds == all_targets).float().mean().item()

    # Effective FLOPs: sum per-layer FLOPs × routing_weight
    effective_flops = 0
    raw_flops = 0
    routed_flops_breakdown: dict[str, int] = {}
    for name, f in flops_per_layer.items():
        raw_flops += f
        weight = 1.0
        if name in routing_map:
            bn, eidx = routing_map[name]
            weight = mean_usage.get(bn, [1.0] * (eidx + 1))[eidx]
        effective_flops += int(f * weight)
        routed_flops_breakdown[name] = int(f * weight)

    # Identify first-conv MAC operation for GSC (real-valued Mel input).
    # The very first conv in the SEE encoder consumes raw FP input — counts as MAC.
    # All other ops consume spike input → AC.
    first_mac_flops = 0
    if args.dataset == "gsc":
        # Find the lexically-first Conv1d — should be `see.pwconv` or similar
        candidates = [n for n in flops_per_layer if n.startswith("see")]
        if candidates:
            first_layer = sorted(candidates)[0]
            first_mac_flops = flops_per_layer[first_layer]

    # SOPs (AC) = effective_FLOPs (excluding first MAC layer) × effective_mean_fr
    ac_flops = effective_flops - first_mac_flops
    sops = effective_mean_fr * ac_flops
    energy_pj = sops * _E_AC_PJ + first_mac_flops * _E_MAC_PJ
    energy_mj = energy_pj * 1e-9  # pJ -> mJ

    label = args.variant or os.path.basename(os.path.dirname(args.checkpoint))
    print("variant,acc,params,raw_flops_g,effective_flops_g,sops_g,first_mac_flops_g,effective_fr,energy_mj")
    print(
        f"{label},{100*acc:.2f},{n_params},"
        f"{raw_flops/1e9:.4f},{effective_flops/1e9:.4f},{sops/1e9:.4f},"
        f"{first_mac_flops/1e9:.4f},{effective_mean_fr:.4f},{energy_mj:.6f}"
    )

    # Diagnostics
    print(f"\n# raw_mean_fr: {raw_mean_fr:.4f}")
    print(f"# effective_mean_fr: {effective_mean_fr:.4f}")
    print(f"# effective_flops_savings_vs_raw: {100*(raw_flops-effective_flops)/raw_flops:.1f}%")
    if mean_usage:
        print(f"\n# Mean expert usage:")
        for bn, vec in sorted(mean_usage.items()):
            print(f"#   {bn}: {[f'{u:.3f}' for u in vec]}")
    print(f"\n# Top-10 layers by effective FLOPs:")
    for name, f in sorted(routed_flops_breakdown.items(), key=lambda x: -x[1])[:10]:
        print(f"#   {name}: {f/1e6:.2f} M")


if __name__ == "__main__":
    main()
