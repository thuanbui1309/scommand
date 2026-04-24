"""Training entry point for SpikCommander baseline.

Usage:
    python -m scripts.train \\
        --config configs/variant/baseline.yaml \\
        --dataset {shd|ssc|gsc} \\
        --seed N \\
        [--output-dir runs/]

Config load order (each layer overrides the previous):
    base.yaml -> dataset/<name>.yaml -> variant/baseline.yaml -> CLI seed

Run directory: ``{output_dir}/{dataset}_seed{seed}_{timestamp}/``
Outputs: ``config.yaml``, ``metrics.csv``, ``best_acc.pt``, ``best_loss.pt``,
         ``epoch_NNNN.pt`` (every 25 epochs).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# Allow running as `python -m scripts.train` from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf

from scommander.datasets import make_loaders
from scommander.models.registry import build_model
from scommander.training import train, build_scheduler
from scommander.utils.seed import set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpikCommander baseline")
    p.add_argument("--config", required=True, help="Variant config path (e.g. configs/variant/baseline.yaml)")
    p.add_argument("--dataset", required=True, choices=["shd", "ssc", "gsc"], help="Dataset name")
    p.add_argument("--seed", type=int, required=True, help="Random seed")
    p.add_argument("--output-dir", default="runs", help="Root directory for run outputs (default: runs/)")
    return p.parse_args()


def _load_config(variant_path: str, dataset_name: str, seed: int):
    """Merge: base -> dataset/<name>.yaml -> variant -> CLI overrides."""
    # Resolve paths relative to this script's location (src/)
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(src_dir, "configs")

    base_cfg = OmegaConf.load(os.path.join(cfg_dir, "base.yaml"))
    dataset_cfg = OmegaConf.load(os.path.join(cfg_dir, "dataset", f"{dataset_name}.yaml"))

    # variant_path may be absolute or relative to cwd
    if not os.path.isabs(variant_path):
        variant_path = os.path.join(src_dir, variant_path)
    variant_cfg = OmegaConf.load(variant_path)

    # CLI overrides
    cli_cfg = OmegaConf.create({"experiment": {"seed": seed}})

    cfg = OmegaConf.merge(base_cfg, dataset_cfg, variant_cfg, cli_cfg)

    # Inject dataset name so loaders and model builder can read it
    cfg.dataset.name = dataset_name

    return cfg


def main() -> None:
    args = _parse_args()

    cfg = _load_config(args.config, args.dataset, args.seed)

    # Seed before anything touches RNG
    set_seed(cfg.experiment.seed)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_seed{args.seed}_{timestamp}"

    # output_dir: prefer CLI arg, fallback to cfg
    output_root = args.output_dir or str(cfg.experiment.output_dir)
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Persist merged config
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"Run dir: {run_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data loaders
    loaders = make_loaders(args.dataset, cfg)
    if args.dataset == "shd":
        train_loader, test_loader = loaders
        val_loader = test_loader   # SHD: test doubles as validation
    else:
        train_loader, val_loader, test_loader = loaders

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg)

    # Train
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        run_dir=run_dir,
    )

    print(f"\nDone. best_acc={100*results['best_acc']:.2f}%  best_loss={results['best_loss']:.4f}")
    print(f"Outputs: {run_dir}")


if __name__ == "__main__":
    main()
