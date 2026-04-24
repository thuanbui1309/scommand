"""Shape tests for Phase 01 building blocks.

Covers: registry plumbing, LIF forward, SEE (B,T,F_raw) -> (T,B,D),
SumAggregator (T,B,D) -> (B,D), ClassifierHead (B,D) -> (B,C).

Deliberately does NOT test STASA / SCR-MLP / full trunk yet — those land in
a follow-up phase-01 session.
"""
from __future__ import annotations

import pytest
import torch

# Import the package triggers @register decorators.
import scommander.models  # noqa: F401
import scommander.modules  # noqa: F401
from scommander.models.registry import REGISTRY, resolve


@pytest.fixture
def shd_batch() -> torch.Tensor:
    """Tiny SHD-like batch: (B, T, F_raw)."""
    return torch.randn(4, 20, 140)


def test_registry_keys_populated() -> None:
    assert "lif" in REGISTRY["neuron"], REGISTRY["neuron"]
    assert "see" in REGISTRY["encoder"], REGISTRY["encoder"]
    assert "sum_over_time" in REGISTRY["aggregator"], REGISTRY["aggregator"]
    assert "linear_head" in REGISTRY["classifier"], REGISTRY["classifier"]


def test_resolve_missing_raises() -> None:
    with pytest.raises(KeyError, match="not registered"):
        resolve("neuron", "no_such_neuron")


def test_lif_forward_preserves_shape() -> None:
    lif_cls = resolve("neuron", "lif")
    lif = lif_cls(backend="torch")  # torch backend works without CuPy
    # SpikingJelly multi-step expects (T, B, ...)
    x = torch.randn(6, 4, 32)
    y = lif(x)
    assert y.shape == x.shape
    # Spikes are 0/1 (hard threshold)
    assert torch.all((y == 0) | (y == 1))


def test_see_forward_shape(shd_batch: torch.Tensor) -> None:
    see_cls = resolve("encoder", "see")
    see = see_cls(in_features=140, out_features=128, kernel_size=7,
                  neuron_cfg={"backend": "torch"})
    out = see(shd_batch)                        # expected (T, B, D)
    B, T, F = shd_batch.shape
    assert out.shape == (T, B, 128), f"got {tuple(out.shape)}, want ({T},{B},128)"


def test_sum_aggregator_collapses_time() -> None:
    agg_cls = resolve("aggregator", "sum_over_time")
    agg = agg_cls()
    x = torch.randn(10, 4, 64)   # (T, B, D)
    out = agg(x)
    assert out.shape == (4, 64)
    assert torch.allclose(out, x.sum(dim=0))


def test_classifier_head_shape() -> None:
    head_cls = resolve("classifier", "linear_head")
    head = head_cls(in_features=128, num_classes=20)
    x = torch.randn(4, 128)
    logits = head(x)
    assert logits.shape == (4, 20)


def test_classifier_head_param_count() -> None:
    """Phase 01 param count sanity: linear_head(D=128, C=20) = 128*20 + 20 = 2580."""
    head_cls = resolve("classifier", "linear_head")
    head = head_cls(in_features=128, num_classes=20, bias=True)
    n = sum(p.numel() for p in head.parameters())
    assert n == 128 * 20 + 20
