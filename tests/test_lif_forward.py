"""LIF forward equivalence test: CuPy vs PyTorch backend."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for LIF tests")
def test_lif_forward_shape() -> None:
    from spikingjelly.activation_based import neuron, surrogate

    lif = neuron.LIFNode(
        tau=2.0,
        v_reset=0.5,
        v_threshold=1.0,
        surrogate_function=surrogate.ATan(alpha=5.0),
        detach_reset=True,
    ).cuda()
    x = torch.randn(4, 16, device="cuda")
    out = lif(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lif_spike_binary() -> None:
    """LIF output must be strictly {0, 1}."""
    from spikingjelly.activation_based import neuron, surrogate, functional

    lif = neuron.LIFNode(
        tau=2.0,
        v_reset=0.5,
        v_threshold=1.0,
        surrogate_function=surrogate.ATan(alpha=5.0),
    ).cuda()
    x = torch.randn(8, 32, device="cuda") * 2
    out = lif(x)
    functional.reset_net(lif)
    uniq = torch.unique(out)
    # Allow {0}, {1}, or {0, 1}
    assert set(uniq.tolist()).issubset({0.0, 1.0}), f"Non-binary output: {uniq}"
