"""Noise robustness eval: AWGN injection at multiple SNR levels.

For each SNR in --snr-list (and clean baseline), adds Gaussian noise to input
spike features and evaluates top-1 accuracy. Useful for paper noise-robustness
table (Phase 08).

SNR (signal-to-noise ratio in dB) defines noise std as:
    sigma_noise = sigma_signal / 10^(SNR/20)
where sigma_signal is the per-batch std of the raw input.

For "clean" (no noise), pass SNR=inf or include 'clean' literal in --snr-list.

Usage:
    python -m scripts.noise_eval \\
        --checkpoint runs/gsc_seed0_20260519_195830/best_acc.pt \\
        --config runs/gsc_seed0_20260519_195830/config.yaml \\
        --dataset gsc \\
        --snr-list clean 20 10 5 0 \\
        --variant K2-2L_s0
"""

from __future__ import annotations

import argparse
import math
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
from scommander.losses.ce import SumSoftmaxCE
from scommander.utils.seed import set_seed

_N_CLASSES = {"shd": 20, "ssc": 35, "gsc": 35}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise robustness eval (AWGN SNR sweep)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True, choices=["shd", "ssc", "gsc"])
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument(
        "--snr-list",
        nargs="+",
        default=["clean", "20", "10", "5", "0"],
        help="SNR levels in dB. Use 'clean' for no noise. Default: clean 20 10 5 0",
    )
    p.add_argument("--variant", default="", help="Label for output CSV row")
    p.add_argument("--seed", type=int, default=0, help="Noise RNG seed for reproducibility")
    return p.parse_args()


def _parse_snr(s: str) -> float:
    if s.lower() == "clean":
        return float("inf")
    return float(s)


def _add_awgn(x: torch.Tensor, snr_db: float, rng: torch.Generator) -> torch.Tensor:
    """Add additive Gaussian noise at given SNR in dB. snr=inf returns x unchanged."""
    if math.isinf(snr_db):
        return x
    sigma_signal = x.float().std().item()
    if sigma_signal == 0:
        return x
    sigma_noise = sigma_signal / (10 ** (snr_db / 20.0))
    noise = torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype) * sigma_noise
    return x + noise


def main() -> None:
    args = _parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.dataset.name = args.dataset
    set_seed(int(cfg.experiment.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = _N_CLASSES[args.dataset]

    state = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(state)
    model.eval()

    loaders = make_loaders(args.dataset, cfg)
    if args.dataset == "shd":
        _, test_loader = loaders
        loader = test_loader
    else:
        _, val_loader, test_loader = loaders
        loader = test_loader if args.split == "test" else val_loader

    # Fixed noise RNG for paired comparison across SNR runs of same variant.
    rng = torch.Generator(device=device).manual_seed(args.seed)

    results: list[tuple[str, float]] = []  # (snr_label, accuracy)

    for snr_str in args.snr_list:
        snr_db = _parse_snr(snr_str)
        # Reset RNG so each SNR sees same underlying noise sequence (paired).
        rng.manual_seed(args.seed)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y, x_len in tqdm(loader, desc=f"SNR={snr_str}dB"):
                attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)
                x = x.float().to(device)
                x_noisy = _add_awgn(x, snr_db, rng)
                y_int = y.long().to(device)

                logits = model(x_noisy, attn_mask)
                m = torch.sum(F.softmax(logits, dim=-1), dim=0)
                preds = m.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_targets.append(y_int.cpu())
                model.reset()

        preds_t = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)
        acc = (preds_t == targets_t).float().mean().item() * 100
        results.append((snr_str, acc))
        print(f"  SNR={snr_str:>6}dB  acc={acc:.2f}%")

    # CSV one-liner for easy aggregation
    label = args.variant or os.path.basename(os.path.dirname(args.checkpoint))
    csv_header = "variant," + ",".join(f"snr_{s}" for s in args.snr_list)
    csv_row = label + "," + ",".join(f"{a:.2f}" for _, a in results)
    print()
    print(csv_header)
    print(csv_row)


if __name__ == "__main__":
    main()
