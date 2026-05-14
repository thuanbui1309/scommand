"""SeMoE: Spike-aware Temporal Mixture-of-Experts.

Replaces STASA's monolithic dual-branch attention with K specialised experts
plus a lightweight spike-driven gate that routes each (timestep, batch) pair
to a single expert via top-1 argmax + straight-through estimator.

Phase 05 paper lead. Spec: ``plans/reports/spec-260515-0245-semoe-design.md``.

Shape contract (matches STASA so it is drop-in via the ``attention`` registry):
    Input:  ``(T, B, D)``
    Output: ``(T, B, D)``

Auxiliary loss:
    Each forward stores ``self.last_aux_loss`` — a scalar tensor with the
    Switch-Transformer load-balancing penalty (Fedus 2021). The trainer
    aggregates these from every SeMoE block via ``collect_semoe_aux_loss``
    and adds them (multiplied by ``load_balance_weight``) to the total loss.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer

from scommander.models.registry import register
from scommander.modules.lif import make_lif


# ── Expert primitives ───────────────────────────────────────────────────────
#
# Each expert maps spikes (T, B, D) → activations (T, B, D_e). Experts are
# intentionally thinner than full STASA (D_e ≤ D) so the K-expert sum stays
# under the baseline parameter budget.

class _SWAExpert(nn.Module):
    """Sliding-window attention expert (no LRA branch)."""

    def __init__(
        self,
        dim: int,
        expert_dim: int,
        num_heads: int,
        window: int,
        neuron_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        assert expert_dim % num_heads == 0, "expert_dim must divide num_heads"
        self.dim = dim
        self.expert_dim = expert_dim
        self.num_heads = num_heads
        self.dh = expert_dim // num_heads
        self.window = window
        self.scale = 1.0 / math.sqrt(self.dh * (2 * window + 1))

        # In-projection: D -> D_e for Q, K, V (single conv1d stacked × 3 lifs)
        self.qkv_conv = nn.Conv1d(dim, expert_dim * 3, kernel_size=1, bias=False)
        self.qkv_bn = layer.BatchNorm1d(expert_dim * 3, step_mode="m")
        self.q_lif = make_lif(neuron_cfg)
        self.k_lif = make_lif(neuron_cfg)
        self.v_lif = make_lif(neuron_cfg)

        self.local_attn_lif = make_lif(neuron_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        T, B, _ = x.shape
        x_bdt = x.permute(1, 2, 0).contiguous()
        qkv = self.qkv_conv(x_bdt)                     # (B, 3*D_e, T)
        qkv = qkv.permute(2, 0, 1).contiguous()        # (T, B, 3*D_e)
        qkv = self.qkv_bn(qkv.unsqueeze(-1)).squeeze(-1)
        q_, k_, v_ = qkv.chunk(3, dim=-1)              # each (T, B, D_e)
        q = self.q_lif(q_).reshape(T, B, self.num_heads, self.dh)
        k = self.k_lif(k_).reshape(T, B, self.num_heads, self.dh)
        v = self.v_lif(v_).reshape(T, B, self.num_heads, self.dh)

        # Sliding window: pad T then unfold (B, H, T, Dh)
        q_bhdt = q.permute(1, 2, 0, 3).contiguous()
        k_bhdt = k.permute(1, 2, 0, 3).contiguous()
        w = self.window
        q_pad = F.pad(q_bhdt, (0, 0, w, w))
        k_pad = F.pad(k_bhdt, (0, 0, w, w))
        q_win = q_pad.unfold(2, 2 * w + 1, 1).permute(0, 1, 2, 4, 3).sum(dim=3)
        k_win = k_pad.unfold(2, 2 * w + 1, 1).permute(0, 1, 2, 4, 3).sum(dim=3)
        q_sum = q_win.permute(2, 0, 1, 3).contiguous()  # (T, B, H, Dh)
        k_sum = k_win.permute(2, 0, 1, 3).contiguous()
        gate = self.local_attn_lif((q_sum + k_sum) * self.scale)
        out = gate * v                                  # (T, B, H, Dh)
        return out.reshape(T, B, self.expert_dim)


class _LRAExpert(nn.Module):
    """Long-range attention expert: global Q+K sum gates V."""

    def __init__(
        self,
        dim: int,
        expert_dim: int,
        num_heads: int,
        neuron_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        assert expert_dim % num_heads == 0
        self.expert_dim = expert_dim
        self.num_heads = num_heads
        self.dh = expert_dim // num_heads

        self.qkv_conv = nn.Conv1d(dim, expert_dim * 3, kernel_size=1, bias=False)
        self.qkv_bn = layer.BatchNorm1d(expert_dim * 3, step_mode="m")
        self.q_lif = make_lif(neuron_cfg)
        self.k_lif = make_lif(neuron_cfg)
        self.v_lif = make_lif(neuron_cfg)
        self.global_attn_lif = make_lif(neuron_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _ = x.shape
        x_bdt = x.permute(1, 2, 0).contiguous()
        qkv = self.qkv_conv(x_bdt).permute(2, 0, 1).contiguous()
        qkv = self.qkv_bn(qkv.unsqueeze(-1)).squeeze(-1)
        q_, k_, v_ = qkv.chunk(3, dim=-1)
        q = self.q_lif(q_).reshape(T, B, self.num_heads, self.dh)
        k = self.k_lif(k_).reshape(T, B, self.num_heads, self.dh)
        v = self.v_lif(v_).reshape(T, B, self.num_heads, self.dh)

        scale = 1.0 / math.sqrt(self.dh * T)
        q_sum_all = q.sum(dim=0, keepdim=True)
        k_sum_all = k.sum(dim=0, keepdim=True)
        gate_all = self.global_attn_lif((q_sum_all + k_sum_all) * scale)
        out = gate_all * v                              # (T, B, H, Dh)
        return out.reshape(T, B, self.expert_dim)


class _IdentityExpert(nn.Module):
    """Pass-through expert: project D→D_e then a single LIF (cheap, for silent frames)."""

    def __init__(
        self,
        dim: int,
        expert_dim: int,
        neuron_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.expert_dim = expert_dim
        self.proj = nn.Conv1d(dim, expert_dim, kernel_size=1, bias=False)
        self.bn = layer.BatchNorm1d(expert_dim, step_mode="m")
        self.lif = make_lif(neuron_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _ = x.shape
        x_bdt = x.permute(1, 2, 0).contiguous()
        out = self.proj(x_bdt).permute(2, 0, 1).contiguous()
        out = self.bn(out.unsqueeze(-1)).squeeze(-1)
        return self.lif(out)


def _build_expert(
    kind: str,
    dim: int,
    expert_dim: int,
    num_heads: int,
    window: int,
    small_window: int,
    neuron_cfg: dict[str, Any] | None,
) -> nn.Module:
    """Factory mapping expert-type string → expert module."""
    k = kind.lower()
    if k == "swa":
        return _SWAExpert(dim, expert_dim, num_heads, window, neuron_cfg)
    if k == "swa_local":
        return _SWAExpert(dim, expert_dim, num_heads, small_window, neuron_cfg)
    if k == "lra":
        return _LRAExpert(dim, expert_dim, num_heads, neuron_cfg)
    if k == "identity":
        return _IdentityExpert(dim, expert_dim, neuron_cfg)
    raise ValueError(f"Unknown expert kind {kind!r}. Supported: swa, swa_local, lra, identity")


# ── SeMoE block ─────────────────────────────────────────────────────────────

@register("attention", "semoe")
class SeMoEBlock(nn.Module):
    """Spike-aware Mixture-of-Experts attention block.

    Args:
        dim: Model dimension D (matches STASA contract).
        num_heads: Heads per expert (D_e must divide num_heads).
        attention_window: SWA window radius for the full-window expert.
        num_experts: K (default 4).
        expert_types: Per-slot kind in ``{swa, swa_local, lra, identity}``.
            Must have length == num_experts.
        small_window: Window for ``swa_local`` slot (default 5).
        load_balance_weight: λ stored on the block; trainer reads it back when
            adding ``last_aux_loss`` to the total loss.
        expert_dim: D_e per expert; defaults to ``D // 2`` per design spec.
        neuron_cfg: Forwarded to every ``make_lif`` call.
        dropout_rate: Output dropout (0 = disabled).
        long_range_branch_factory: Accepted for STASA-compatibility (ignored).

    Shape:
        Input:  ``(T, B, D)``
        Output: ``(T, B, D)``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_window: int = 20,
        value_branch_kernel: int = 9,        # accepted for STASA-compat; unused
        use_long_range: bool = True,         # accepted for STASA-compat; unused
        long_range_branch_factory: Any = None,  # ignored — SeMoE owns its routing
        num_experts: int = 4,
        expert_types: Sequence[str] = ("swa", "lra", "swa_local", "identity"),
        small_window: int = 5,
        load_balance_weight: float = 0.01,
        expert_dim: int | None = None,
        neuron_cfg: dict[str, Any] | None = None,
        use_bn: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.K = int(num_experts)
        if len(expert_types) != self.K:
            raise ValueError(
                f"expert_types length {len(expert_types)} != num_experts {self.K}"
            )
        self.expert_types = tuple(expert_types)

        # D_e default: half of D, rounded down to a num_heads multiple
        if expert_dim is None:
            expert_dim = max(num_heads, (dim // 2 // num_heads) * num_heads)
        if expert_dim % num_heads != 0:
            raise ValueError(f"expert_dim {expert_dim} must be divisible by num_heads {num_heads}")
        self.expert_dim = expert_dim
        self.load_balance_weight = float(load_balance_weight)

        # Spike-driven gate: depthwise temporal conv → LIF → linear-to-K
        self.gate_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.gate_bn = layer.BatchNorm1d(dim, step_mode="m") if use_bn else nn.Identity()
        self.gate_lif = make_lif(neuron_cfg)
        self.gate_linear = nn.Linear(dim, self.K, bias=True)

        # Experts
        self.experts = nn.ModuleList([
            _build_expert(
                kind=expert_types[k],
                dim=dim,
                expert_dim=expert_dim,
                num_heads=num_heads,
                window=attention_window,
                small_window=small_window,
                neuron_cfg=neuron_cfg,
            )
            for k in range(self.K)
        ])

        # Output projection D_e → D (with BN + LIF, mirrors STASA proj path)
        self.out_proj = nn.Conv1d(expert_dim, dim, kernel_size=1, bias=False)
        self.out_bn = layer.BatchNorm1d(dim, step_mode="m") if use_bn else nn.Identity()
        self.out_lif = make_lif(neuron_cfg)

        self._use_dp = dropout_rate > 0.0
        if self._use_dp:
            self.out_dropout = layer.Dropout(dropout_rate, step_mode="m")

        # Aux loss + diagnostics, refreshed every forward
        self.register_buffer("last_aux_loss_value", torch.zeros(1), persistent=False)
        self.register_buffer("last_expert_usage", torch.zeros(self.K), persistent=False)
        self.last_aux_loss: torch.Tensor = self.last_aux_loss_value

    # ── gate ─────────────────────────────────────────────────────────────────

    def _gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the gate; return (mask, soft, hard).

        mask is the STE-friendly tensor (hard forward, soft backward) used to
        weight expert outputs. soft/hard are the raw soft + one-hot tensors
        for the load-balance loss and usage stats.
        """
        T, B, _ = x.shape
        x_bdt = x.permute(1, 2, 0).contiguous()
        g = self.gate_conv(x_bdt).permute(2, 0, 1).contiguous()
        g = self.gate_bn(g.unsqueeze(-1)).squeeze(-1)
        g = self.gate_lif(g)                                  # (T, B, D)
        logits = self.gate_linear(g)                          # (T, B, K)

        soft = F.softmax(logits, dim=-1)
        hard = F.one_hot(logits.argmax(dim=-1), num_classes=self.K).to(soft.dtype)
        # STE: forward uses hard mask, backward routes through soft
        mask = hard + (soft - soft.detach())
        return mask, soft, hard

    @staticmethod
    def _load_balance_loss(soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
        """Switch Transformer auxiliary loss: K · Σ_k p_k · f_k.

        Encourages the gate's soft probabilities (p_k) and the actual hard
        assignment rates (f_k) to spread evenly across experts.
        """
        K = soft.shape[-1]
        p_k = soft.mean(dim=(0, 1))                           # (K,)
        f_k = hard.mean(dim=(0, 1))                           # (K,)
        return K * (p_k * f_k).sum()

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Route, run experts, combine, project. (T, B, D) → (T, B, D)."""
        T, B, D = x.shape
        mask, soft, hard = self._gate(x)                      # mask: (T, B, K)

        if attention_mask is not None:
            # Zero out padded timesteps so they don't bias load-balance stats
            mt = attention_mask.transpose(0, 1).unsqueeze(-1).to(mask.dtype)  # (T, B, 1)
            mask = mask * mt
            soft = soft * mt
            hard = hard * mt

        # Experts (PyTorch eager runs all; routing materialises at masked sum)
        outs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # outs: (T, B, D_e, K)
        routed = (outs * mask.unsqueeze(-2)).sum(dim=-1)      # (T, B, D_e)

        # Project back D_e → D, BN, LIF
        routed_bdt = routed.permute(1, 2, 0).contiguous()
        out = self.out_proj(routed_bdt).permute(2, 0, 1).contiguous()
        out = self.out_bn(out.unsqueeze(-1)).squeeze(-1)
        out = self.out_lif(out)

        if self._use_dp:
            out = self.out_dropout(out)

        # Stash aux loss + usage stats for the trainer to pick up.
        # Note: assignment (not in-place copy) — keeps autograd live.
        self.last_aux_loss = self._load_balance_loss(soft, hard)
        with torch.no_grad():
            self.last_expert_usage = hard.float().mean(dim=(0, 1))

        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, K={self.K}, expert_dim={self.expert_dim}, "
            f"types={self.expert_types}, lb_weight={self.load_balance_weight}"
        )


# ── Trainer-side helpers ─────────────────────────────────────────────────────

def collect_semoe_aux_loss(model: nn.Module) -> torch.Tensor:
    """Sum (load_balance_weight · last_aux_loss) across every SeMoE block.

    Returns a zero tensor on the model's device when no SeMoE block has run
    a forward pass yet, so trainers can unconditionally add it to the total
    loss without branching on attention type.
    """
    total: torch.Tensor | None = None
    for m in model.modules():
        if isinstance(m, SeMoEBlock):
            term = m.load_balance_weight * m.last_aux_loss
            total = term if total is None else total + term
    if total is None:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return total


def collect_semoe_expert_usage(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return {block_path: usage_vector} for every SeMoE block. For diagnostics."""
    out: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, SeMoEBlock):
            out[name] = m.last_expert_usage.detach().cpu()
    return out
