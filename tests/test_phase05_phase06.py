"""Phase 05 (PLIF, Spiking Mamba) + Phase 06 (sparsity) module tests."""
from __future__ import annotations

import pytest
import torch

import scommander.models  # noqa: F401  # populates registry
import scommander.modules  # noqa: F401
from scommander.models.registry import resolve
from scommander.modules.lif import make_lif
from scommander.losses.sparsity import FiringRateCollector, FiringRatePenalty


# ── PLIF ────────────────────────────────────────────────────────────────────

def test_plif_registered() -> None:
    assert resolve("neuron", "plif") is not None


def test_plif_forward_shape() -> None:
    plif_cls = resolve("neuron", "plif")
    plif = plif_cls(backend="torch")
    x = torch.randn(6, 4, 32)
    y = plif(x)
    assert y.shape == x.shape
    assert torch.all((y == 0) | (y == 1))


def test_plif_tau_is_learnable() -> None:
    plif_cls = resolve("neuron", "plif")
    plif = plif_cls(backend="torch")
    learnable = [p for p in plif.parameters() if p.requires_grad]
    assert len(learnable) >= 1, "PLIF must have at least one learnable param (tau-related)"


def test_make_lif_dispatches_plif() -> None:
    lif = make_lif({"type": "lif", "backend": "torch"})
    plif = make_lif({"type": "plif", "backend": "torch"})
    assert type(lif).__name__ == "LIFNode"
    assert type(plif).__name__ == "PLIFNode"


# ── Sparsity collector + penalty ────────────────────────────────────────────

def test_fr_collector_captures_neurons() -> None:
    model_cls = resolve("model", "spikcommander")
    model = model_cls(
        in_features=140, num_classes=20, dim=128, n_heads=8, depth=1,
        window_radius=20, neuron_cfg={"backend": "torch"},
    )
    model.reset()
    x = torch.randn(2, 20, 140)
    with FiringRateCollector(model) as fr:
        _ = model(x)
    assert len(fr.spike_rates) > 0, "no LIF nodes captured"
    for name, rate in fr.spike_rates.items():
        assert 0.0 <= rate.item() <= 1.0, f"{name} FR out of [0,1]: {rate.item()}"


def test_fr_penalty_zero_spikes() -> None:
    pen = FiringRatePenalty(lam=1.0)
    rates = {"l0": torch.tensor(0.0), "l1": torch.tensor(0.0)}
    assert pen(rates).item() == 0.0


def test_fr_penalty_uniform_spikes() -> None:
    pen = FiringRatePenalty(lam=0.5)
    rates = {"l0": torch.tensor(0.4), "l1": torch.tensor(0.6)}
    assert abs(pen(rates).item() - 0.5 * 0.5) < 1e-6


def test_fr_penalty_hinge_mode() -> None:
    pen = FiringRatePenalty(lam=1.0, target_fr=0.3)
    rates = {"l0": torch.tensor(0.5), "l1": torch.tensor(0.2)}  # 0.5-0.3=0.2; 0.2-0.3<0
    expected = (0.2 + 0.0) / 2
    assert abs(pen(rates).item() - expected) < 1e-6


# ── Spiking Mamba (gated; requires mamba_ssm + CUDA) ────────────────────────

def _mamba_available() -> bool:
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm + CUDA required")
def test_spiking_mamba_registered_and_forward() -> None:
    cls = resolve("long_range_branch", "spiking_mamba")
    branch = cls(dim=128, num_heads=8, neuron_cfg={"backend": "cupy"}).cuda()
    T, B, H, Dh = 8, 2, 8, 16  # 8*16 = 128 = dim
    q = torch.randn(T, B, H, Dh, device="cuda")
    k = torch.randn(T, B, H, Dh, device="cuda")
    v = torch.randn(T, B, H, Dh, device="cuda")
    out = branch(q, k, v, global_scale=0.1)
    assert out.shape == (T, B, H, Dh)
    assert torch.all((out == 0) | (out == 1))


@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm + CUDA required")
def test_spiking_mamba_gate_initial_near_identity() -> None:
    cls = resolve("long_range_branch", "spiking_mamba")
    branch = cls(dim=64, num_heads=8, gate_init_bias=-3.0,
                 neuron_cfg={"backend": "cupy"}).cuda()
    h = torch.randn(2, 8, 64, device="cuda")
    gate = torch.sigmoid(branch.gate_proj(h))
    assert gate.mean().item() < 0.1, f"gate not near 0 at init: {gate.mean().item():.3f}"
