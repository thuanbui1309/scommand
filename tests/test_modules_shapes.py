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


# ── Phase 01 extension: SCR-MLP, STASA, SpikCommander trunk ─────────────────

@pytest.fixture
def shd_batch_tbd() -> torch.Tensor:
    """Fake spike tensor in multi-step layout (T, B, D)."""
    return torch.randn(20, 4, 128)


def test_scr_mlp_shape(shd_batch_tbd: torch.Tensor) -> None:
    """SCR-MLP must preserve (T, B, D) shape."""
    mlp_cls = resolve("mlp", "scr_mlp")
    mlp = mlp_cls(in_features=128, expansion_ratio=4.0, neuron_cfg={"backend": "torch"})
    out = mlp(shd_batch_tbd)
    assert out.shape == shd_batch_tbd.shape, f"got {tuple(out.shape)}"


def test_stasa_shape(shd_batch_tbd: torch.Tensor) -> None:
    """STASA must preserve (T, B, D) shape."""
    attn_cls = resolve("attention", "stasa")
    attn = attn_cls(dim=128, num_heads=8, attention_window=20, neuron_cfg={"backend": "torch"})
    out = attn(shd_batch_tbd)
    assert out.shape == shd_batch_tbd.shape, f"got {tuple(out.shape)}"


def test_stasa_long_range_factory_injection(shd_batch_tbd: torch.Tensor) -> None:
    """STASA factory injection: nn.Identity() replaces LRABranch; forward must not crash."""
    import torch.nn as nn

    attn_cls = resolve("attention", "stasa")
    # Factory returns nn.Identity; it won't match the LRABranch (q,k,v,scale) signature
    # but we test that the *module* is stored and forward runs with a valid LRA-compatible
    # shim. Use a lambda that returns a module accepting (q, k, v, global_scale).
    class _IdentityLRA(nn.Module):
        def forward(self, q, k, v, global_scale):
            return v  # pass-through — shape matches (T, B, H, Dh)

    attn = attn_cls(
        dim=128,
        num_heads=8,
        attention_window=20,
        long_range_branch_factory=lambda: _IdentityLRA(),
        neuron_cfg={"backend": "torch"},
    )
    assert isinstance(attn.lra_module, _IdentityLRA), "factory module not stored"
    out = attn(shd_batch_tbd)
    assert out.shape == shd_batch_tbd.shape


def test_trunk_forward_shape_shd(shd_batch: torch.Tensor) -> None:
    """SpikCommander SHD config: (B, T, F) -> (B, C)."""
    model_cls = resolve("model", "spikcommander")
    model = model_cls(
        in_features=140,
        num_classes=20,
        dim=128,
        n_heads=8,
        depth=1,
        window_radius=20,
        neuron_cfg={"backend": "torch"},
    )
    model.reset()
    out = model(shd_batch)
    assert out.shape == (4, 20), f"got {tuple(out.shape)}"


def test_trunk_backward_grad_flows_shd(shd_batch: torch.Tensor) -> None:
    """Gradients must flow through the full trunk on a SHD-sized batch."""
    model_cls = resolve("model", "spikcommander")
    model = model_cls(
        in_features=140,
        num_classes=20,
        dim=128,
        n_heads=8,
        depth=1,
        window_radius=20,
        neuron_cfg={"backend": "torch"},
    )
    model.reset()
    logits = model(shd_batch)
    loss = logits.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().max() > 0
        for p in model.parameters()
    )
    assert has_grad, "no parameter received a non-zero gradient"


def test_trunk_param_count_within_band() -> None:
    """SHD trunk param count diagnostic gate.

    Spec §3 reports paper baseline at 0.19M but the modular implementation
    (with both pw-conv stages in SCR-MLP and dual proj in STASA) yields ~296K.
    The spec itself notes this discrepancy and defers exact recalibration to
    Phase 01 Step 10.  This test guards against catastrophic explosion (>500K)
    or collapse (<100K) while the exact figure is validated against the server.
    Print the actual value for the server-side CI log.
    """
    model_cls = resolve("model", "spikcommander")
    model = model_cls(
        in_features=140,
        num_classes=20,
        dim=128,
        n_heads=8,
        depth=1,
        window_radius=20,
        neuron_cfg={"backend": "torch"},
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"SHD params: {n:,}")
    assert 100_000 <= n <= 500_000, (
        f"param count {n:,} outside sanity band [100_000, 500_000]"
    )
