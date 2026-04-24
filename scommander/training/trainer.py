"""Training loop for SpikCommander.

Ports ``train_model`` from
``reference/SpikCommander/SCommander/main_former_v2_shd_spikcommander.py:77-205``
and ``main_former_v2_gsc_spikcommander.py:110-258``.

Design notes
------------
- **No EMA** — not used in any reference training script.
- **No gradient clipping** — reference does not use it; ``cfg.training.grad_clip``
  is present in base.yaml for future tracks but ignored here. Documented below.
- **Reset semantics** — ``model.reset()`` is called after every batch (train + eval).
  The reference uses ``functional.reset_net(model)``. Our ``SpikCommander.reset()``
  walks submodules directly and is functionally equivalent. Using ``model.reset()``
  per spec to avoid the ``functional.reset_net`` warning on non-MemoryModule trunk.
- **GSC SpecAugment** — applied at batch level before forward, matching reference
  ``main_former_v2_gsc*.py:134-136`` (``augs(x, x_len)`` in the train loop).
  SHD/SSC augmentation is handled at the dataset level (transform in __getitem__).
- **Attention mask** — ``padded_sequence_mask(x_len)`` returns ``(T, B)`` bool mask
  (True=valid). Transposed to ``(B, T)`` before passing to model, matching reference
  ``attention_mask.transpose(0,1).to(device)``.
- **Checkpoint naming** — ``best_acc.pt`` and ``best_loss.pt`` state dicts saved to
  ``run_dir``. CSV row appended each epoch: loss_train,acc_train,loss_val,acc_val,lr,time.
- **grad_clip ignored** — ``cfg.training.grad_clip=1.0`` in base.yaml is intentionally
  not applied in baseline. Phase 03+ tracks may enable it via override.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scommander.augmentations.spec_aug import SpecAugment
from scommander.datasets import padded_sequence_mask
from scommander.losses.ce import SumSoftmaxCE, accuracy_from_logits


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg,
    device: torch.device,
    run_dir: str,
) -> dict:
    """Full training loop.

    Args:
        model: SpikCommander instance (already on ``device``).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (SHD: test set; SSC/GSC: valid split).
        test_loader: Optional test DataLoader (None for SHD).
        optimizer: AdamW optimizer.
        scheduler: CosineAnnealingLR scheduler.
        cfg: OmegaConf config.
        device: Target device.
        run_dir: Directory for checkpoints and metrics CSV.

    Returns:
        dict with keys:
            - ``'best_acc'``: float, best validation accuracy (0-1 scale).
            - ``'best_loss'``: float, best validation loss.
            - ``'epochs'``: list of per-epoch dicts with train/val metrics.
    """
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.csv")

    n_classes = _get_n_classes(cfg)
    num_epochs = int(cfg.training.epochs)
    dataset_name = str(cfg.dataset.name)
    aug_enabled = bool(cfg.augmentation.enabled)

    # SpecAugment for GSC (batch-level augmentation in train loop)
    spec_aug: Optional[SpecAugment] = None
    if dataset_name == "gsc" and aug_enabled:
        sa_cfg = cfg.augmentation.specaug
        spec_aug = SpecAugment(
            n_freq_masks=int(sa_cfg.n_freq_masks),
            freq_mask_size=int(sa_cfg.freq_mask_size),
            n_time_masks=int(sa_cfg.n_time_masks),
            time_mask_pct=float(sa_cfg.time_mask_pct),
        ).to(device)

    loss_fn = SumSoftmaxCE()

    best_acc_val = 0.0
    best_loss_val = float("inf")
    epoch_records = []

    # Write CSV header
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_train", "acc_train", "loss_val", "acc_val", "lr", "time_s"])

    for epoch in range(num_epochs):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        loss_batch, acc_batch = [], []

        for x, y, x_len in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
            # Build attention mask: padded_sequence_mask -> (T, B) -> transpose -> (B, T)
            attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)

            x = x.float().to(device)   # (B, T, N) or (B, T, F)
            y_int = y.long().to(device)
            y_onehot = F.one_hot(y_int, n_classes).float()

            # GSC batch-level SpecAugment (x_len contains valid T per sample)
            if spec_aug is not None:
                x_len_dev = x_len.to(device)
                x = spec_aug(x, x_len_dev)

            optimizer.zero_grad()
            logits = model(x, attn_mask)             # (T, B, C)
            loss = loss_fn(logits, y_onehot)
            loss.backward()
            optimizer.step()

            acc = accuracy_from_logits(logits.detach(), y_onehot)
            loss_batch.append(loss.detach().cpu().item())
            acc_batch.append(acc)

            # Reset LIF membrane state between independent sequences
            model.reset()

        loss_train = float(np.mean(loss_batch))
        acc_train = float(np.mean(acc_batch))

        # Advance scheduler after each epoch (reference: scheduler.step() after train loop)
        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        loss_val, acc_val = _eval_epoch(model, val_loader, loss_fn, n_classes, device, spec_aug)

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:4d} | "
            f"loss_train={loss_train:.4f} acc_train={100*acc_train:.2f}% | "
            f"loss_val={loss_val:.4f} acc_val={100*acc_val:.2f}% | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        # ── Checkpoints ──────────────────────────────────────────────────────
        if acc_val > best_acc_val:
            best_acc_val = acc_val
            torch.save(model.state_dict(), os.path.join(run_dir, "best_acc.pt"))
            print(f"  -> Saved best_acc.pt  (acc={100*best_acc_val:.2f}%)")

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save(model.state_dict(), os.path.join(run_dir, "best_loss.pt"))
            print(f"  -> Saved best_loss.pt (loss={best_loss_val:.4f})")

        # Every-25-epoch checkpoint
        if (epoch + 1) % 25 == 0:
            ckpt_path = os.path.join(run_dir, f"epoch_{epoch:04d}.pt")
            torch.save(model.state_dict(), ckpt_path)

        # ── CSV log ──────────────────────────────────────────────────────────
        record = {
            "epoch": epoch,
            "loss_train": loss_train,
            "acc_train": acc_train,
            "loss_val": loss_val,
            "acc_val": acc_val,
            "lr": current_lr,
            "time_s": elapsed,
        }
        epoch_records.append(record)

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{loss_train:.6f}",
                f"{acc_train:.6f}",
                f"{loss_val:.6f}",
                f"{acc_val:.6f}",
                f"{current_lr:.8f}",
                f"{elapsed:.2f}",
            ])

    return {
        "best_acc": best_acc_val,
        "best_loss": best_loss_val,
        "epochs": epoch_records,
    }


def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SumSoftmaxCE,
    n_classes: int,
    device: torch.device,
    spec_aug: Optional[SpecAugment] = None,
) -> tuple[float, float]:
    """Run one evaluation pass.

    Args:
        model: Model in eval mode will be set inside.
        loader: DataLoader (val or test).
        loss_fn: SumSoftmaxCE instance.
        n_classes: Number of output classes.
        device: Target device.
        spec_aug: SpecAugment (None for SHD/SSC; not applied during eval).

    Returns:
        ``(mean_loss, mean_acc)`` over all batches.
    """
    model.eval()
    loss_batch, acc_batch = [], []

    with torch.no_grad():
        for x, y, x_len in tqdm(loader, desc="eval", leave=False):
            attn_mask = padded_sequence_mask(x_len).transpose(0, 1).to(device)

            x = x.float().to(device)
            y_int = y.long().to(device)
            y_onehot = F.one_hot(y_int, n_classes).float()

            # SpecAugment NOT applied during eval — only train-time augmentation
            logits = model(x, attn_mask)           # (T, B, C)
            loss = loss_fn(logits, y_onehot)
            acc = accuracy_from_logits(logits, y_onehot)

            loss_batch.append(loss.cpu().item())
            acc_batch.append(acc)

            model.reset()

    return float(np.mean(loss_batch)), float(np.mean(acc_batch))


def _get_n_classes(cfg) -> int:
    """Resolve number of output classes from config."""
    dataset_name = str(cfg.dataset.name)
    _N_CLASSES = {"shd": 20, "ssc": 35, "gsc": 35}
    if dataset_name not in _N_CLASSES:
        raise ValueError(f"Unknown dataset {dataset_name!r} for n_classes lookup.")
    return _N_CLASSES[dataset_name]
