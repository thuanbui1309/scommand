"""SeMoE Phase 05 unit tests.

Covers: registry plumbing, forward shape, gate STE backward, load-balance
loss positivity + zero-floor, expert usage stats, attention_mask handling,
and end-to-end SpikCommander assembly with attention=semoe.

CPU-only (backend='torch'); the cupy LIF kernel needs CUDA.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import scommander.models  # noqa: F401 — populate registry
from scommander.models.registry import REGISTRY, build_model, resolve
from scommander.modules.semoe import (
    SeMoEBlock,
    collect_semoe_aux_loss,
    collect_semoe_expert_usage,
)


@pytest.fixture
def tbd_input() -> torch.Tensor:
    """SHD-like spike tensor in multi-step layout."""
    return torch.randn(20, 4, 128)


def _semoe_kwargs() -> dict:
    return dict(
        dim=128,
        num_heads=8,
        attention_window=20,
        num_experts=4,
        expert_types=("swa", "lra", "swa_local", "identity"),
        small_window=5,
        load_balance_weight=0.01,
        neuron_cfg={"backend": "torch"},
    )


def test_registry_has_semoe() -> None:
    assert "semoe" in REGISTRY["attention"], REGISTRY["attention"]


def test_semoe_forward_shape(tbd_input: torch.Tensor) -> None:
    block = SeMoEBlock(**_semoe_kwargs())
    out = block(tbd_input)
    assert out.shape == tbd_input.shape, f"got {tuple(out.shape)}"


def test_semoe_aux_loss_is_scalar_and_finite(tbd_input: torch.Tensor) -> None:
    block = SeMoEBlock(**_semoe_kwargs())
    _ = block(tbd_input)
    aux = block.last_aux_loss
    assert aux.dim() == 0, f"aux loss must be scalar, got shape {tuple(aux.shape)}"
    assert torch.isfinite(aux), aux
    # Switch-Transformer formula: K · Σ p_k f_k. With K experts and uniform
    # routing the lower bound is 1.0; argmax routing pushes it ≥ 1 still.
    assert aux.item() >= 0.0, aux


def test_semoe_expert_usage_sums_to_one(tbd_input: torch.Tensor) -> None:
    block = SeMoEBlock(**_semoe_kwargs())
    _ = block(tbd_input)
    usage = block.last_expert_usage
    assert usage.shape == (4,), f"usage shape {tuple(usage.shape)}"
    assert pytest.approx(usage.sum().item(), abs=1e-5) == 1.0


def test_semoe_ste_backward_routes_grad_to_gate(tbd_input: torch.Tensor) -> None:
    """STE: hard mask in forward, soft mask in backward → gate gets a gradient."""
    block = SeMoEBlock(**_semoe_kwargs())
    out = block(tbd_input.requires_grad_(False))
    loss = out.sum() + block.last_aux_loss
    loss.backward()
    grad = block.gate_linear.weight.grad
    assert grad is not None, "gate linear weight has no grad"
    assert torch.isfinite(grad).all(), grad
    # Some routing branches must contribute non-trivially; entire-zero would
    # mean the soft path collapsed.
    assert grad.abs().sum().item() > 0.0, grad


def test_semoe_attention_mask_zeroes_padded_routing(tbd_input: torch.Tensor) -> None:
    """Pad mask should zero out aux-loss contribution of pad timesteps."""
    block = SeMoEBlock(**_semoe_kwargs())
    T, B, _ = tbd_input.shape
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, T // 2:] = False  # mark second half as pad
    _ = block(tbd_input, attention_mask=mask)
    usage = block.last_expert_usage
    # With half the steps masked, total assignment mass should be ~0.5.
    assert pytest.approx(usage.sum().item(), abs=1e-5) == 0.5


def test_collect_semoe_aux_loss_aggregates_across_blocks(tbd_input: torch.Tensor) -> None:
    """Two SeMoE blocks in a Sequential → aux loss is the weighted sum."""
    block_a = SeMoEBlock(**_semoe_kwargs())
    block_b = SeMoEBlock(**_semoe_kwargs())
    container = nn.Sequential(block_a, block_b)
    _ = container(tbd_input)
    total = collect_semoe_aux_loss(container)
    expected = block_a.load_balance_weight * block_a.last_aux_loss \
        + block_b.load_balance_weight * block_b.last_aux_loss
    assert torch.allclose(total, expected, atol=1e-6)


def test_collect_semoe_aux_loss_returns_zero_when_no_block() -> None:
    """No SeMoEBlock present → zero scalar so trainer can add unconditionally."""
    container = nn.Linear(8, 8)
    out = collect_semoe_aux_loss(container)
    assert out.dim() == 0
    assert out.item() == 0.0


def test_expert_usage_helper_yields_one_entry_per_block(tbd_input: torch.Tensor) -> None:
    block = SeMoEBlock(**_semoe_kwargs())
    container = nn.Sequential(block)
    _ = container(tbd_input)
    usage = collect_semoe_expert_usage(container)
    assert len(usage) == 1, usage
    [(_name, vec)] = usage.items()
    assert vec.shape == (4,)


def test_semoe_invalid_expert_type_raises() -> None:
    bad = dict(_semoe_kwargs())
    bad["expert_types"] = ("swa", "bogus", "swa_local", "identity")
    with pytest.raises(ValueError, match="Unknown expert kind"):
        SeMoEBlock(**bad)


def test_semoe_expert_count_mismatch_raises() -> None:
    bad = dict(_semoe_kwargs())
    bad["num_experts"] = 5  # but expert_types has length 4
    with pytest.raises(ValueError, match="expert_types length"):
        SeMoEBlock(**bad)


def test_build_model_picks_semoe_when_configured() -> None:
    """End-to-end: build_model resolves attention=semoe and forwards (B,T,F)→(T,B,C)."""
    cfg = {
        "dataset": {"name": "shd"},
        "model": {
            "arch": "spikcommander",
            "depth": 1,
            "dim": 128,
            "n_heads": 8,
            "expansion": 4,
            "window_radius": 20,
            "attention": "semoe",
            "semoe": {
                "num_experts": 4,
                "expert_types": ["swa", "lra", "swa_local", "identity"],
                "small_window": 5,
                "load_balance_weight": 0.01,
            },
        },
        "neuron": {"tau": 2.0, "v_threshold": 1.0, "v_reset": 0.5,
                   "surrogate": "atan", "alpha": 5.0, "backend": "torch"},
        "training": {"dropout": 0.0},
    }
    model = build_model(cfg)
    # At least one SeMoE block must be wired into the trunk
    semoe_blocks = [m for m in model.modules() if isinstance(m, SeMoEBlock)]
    assert len(semoe_blocks) == 1, semoe_blocks

    x = torch.randn(2, 20, 140)         # (B, T, F_raw)
    logits = model(x)
    assert logits.shape == (20, 2, 20), f"got {tuple(logits.shape)}"

    aux = collect_semoe_aux_loss(model)
    assert torch.isfinite(aux), aux
    assert aux.item() > 0.0, aux
