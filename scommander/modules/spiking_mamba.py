"""Spiking Mamba long-range branch (Phase 05 C1).

Drop-in replacement for ``LRABranch`` via the ``long_range_branch.spiking_mamba``
registry slot. Implements the Mamba-Spike soft-gating pattern from the C1 spec
in ``plans/260422-0220-spikcommander-improvement/phase-05-track-c-hybrid.md``:

    h     = v                                   # real-valued in SSM space
    ssm   = Mamba(h)                            # selective SSM scan
    gate  = sigmoid(gate_proj(h))               # init near 0 -> identity-like
    fused = gate * ssm + (1 - gate) * h
    out   = lif(fused)                          # back to spike-driven outputs

Operates on the V tensor `(T, B, H, Dh)` (flattened to `(B, T, D)` for the
SSM scan) so the existing STASA factory injection works unchanged. The Q/K
inputs are unused — Mamba's selectivity provides its own input-dependent
gating, making explicit attention scores redundant.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from scommander.models.registry import register
from scommander.modules.lif import make_lif


try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    _MAMBA_AVAILABLE = False


@register("long_range_branch", "spiking_mamba_lite")
class SpikingMambaLite(nn.Module):
    """Lighter Spiking Mamba: d_state=8, expand=1 (cuts SSM params ~50%).

    Subclassed in spirit from SpikingMambaBranch but with smaller defaults
    to test the param/FR tradeoff curve.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 1,
        gate_init_bias: float = -3.0,
        neuron_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm required")
        assert dim % num_heads == 0
        self.dim, self.num_heads = dim, num_heads
        self.dh = dim // num_heads
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init_bias)
        self.lif = make_lif(neuron_cfg)

    def forward(self, q, k, v, global_scale):
        T, B, H, Dh = v.shape
        D = H * Dh
        h = v.permute(1, 0, 2, 3).reshape(B, T, D).contiguous().float()
        ssm = self.mamba(h)
        gate = torch.sigmoid(self.gate_proj(h))
        fused = gate * ssm + (1.0 - gate) * h
        fused_tbd = fused.permute(1, 0, 2).contiguous()
        spike = self.lif(fused_tbd)
        return spike.view(T, B, H, Dh)


@register("long_range_branch", "spiking_mamba")
class SpikingMambaBranch(nn.Module):
    """Mamba-Spike soft-gated SSM branch.

    Args:
        dim: Total model dimension D (= num_heads * Dh).
        num_heads: Attention head count (kept for signature compat with LRABranch).
        d_state: SSM state size N (Mamba default 16). Cheap; small impact on params.
        d_conv: Mamba local conv width (default 4).
        expand: SSM expansion ratio (default 2 → 2D internal).
        gate_init_bias: Initial bias for gate_proj (default -3.0). sigmoid(-3) ≈ 0.047
            so the branch starts near-identity → stable warmup.
        neuron_cfg: Forwarded to ``make_lif`` for the output spiking neuron.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        gate_init_bias: float = -3.0,
        neuron_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for SpikingMambaBranch. "
                "Install via `pip install mamba-ssm`."
            )
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.dh = dim // num_heads
        self.d_state = d_state

        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.gate_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init_bias)

        self.lif = make_lif(neuron_cfg)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_scale: float,
    ) -> torch.Tensor:
        """Args:
            q, k: unused (signature parity with LRABranch).
            v: ``(T, B, H, Dh)`` value-branch tensor (real-valued post V-conv).
            global_scale: unused.

        Returns:
            ``(T, B, H, Dh)`` spike-driven output.
        """
        T, B, H, Dh = v.shape
        D = H * Dh

        # (T, B, H, Dh) -> (B, T, D)
        h = v.permute(1, 0, 2, 3).reshape(B, T, D).contiguous().float()

        ssm = self.mamba(h)                       # (B, T, D)
        gate = torch.sigmoid(self.gate_proj(h))   # (B, T, D), starts near 0
        fused = gate * ssm + (1.0 - gate) * h     # (B, T, D)

        # back to (T, B, D) -> LIF (multi-step) -> reshape to (T, B, H, Dh)
        fused_tbd = fused.permute(1, 0, 2).contiguous()
        spike = self.lif(fused_tbd)
        return spike.view(T, B, H, Dh)
