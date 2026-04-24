"""STASA: Spiking Temporal-Aware Self-Attention.

Implements paper Eqs 6-9 (Appendix C):
    SWA branch: local sliding-window attention (bidirectional, window=2w+1)
    LRA branch:  global spike-driven attention (sum Q/K over all T)
    V-branch:    depthwise 2D conv on multi-head values (implicit temporal bias)

LRA temporal mask finding (from grepping reference spikcommander_backbone.py):
    No explicit learnable mask matrix found.  ``MSTASA_v_branch.__init__`` has
    ``self.v_dw`` (DepthwiseConv2d) and ``self.v_pw`` (Conv2d pointwise) on V,
    plus ``self.v_dw_lif``.  No ``self.mask``, ``self.alpha``, ``self.beta``, or
    ``nn.Parameter`` acting as a temporal-decay matrix exists in the reference.
    Conclusion: temporal bias is *implicit* via V-branch depthwise conv — the
    conv kernel spans the head-dim axis and captures position-sensitive patterns
    in the spiking value stream.  LRA is purely spike-driven global attention
    (Q_sum + K_sum scaled, gated by LIF, Hadamard with V).  No extra params.

Shape contract (all public methods):
    Input/Output: ``(T, B, D)``
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer

from scommander.models.registry import register
from scommander.modules.lif import make_lif


# ── LRA branch ───────────────────────────────────────────────────────────────

@register("long_range_branch", "lra")
class LRABranch(nn.Module):
    """Global spike-driven attention branch (Long-Range Aware).

    Computes a single global gate by summing Q and K over all T:
        gate_all = LIF( (Q.sum(T) + K.sum(T)) * β_global )   shape (1, B, H, Dh)
        out      = gate_all * V                                broadcast over T

    No explicit temporal mask matrix — temporal bias comes from V-branch conv
    in the parent STASA module (see module docstring for rationale).

    Args:
        dim: Total model dimension D.
        num_heads: Number of attention heads H. Must divide D.
        global_scale: β_global = 1 / sqrt(Dh * T). Caller supplies T-dependent value.
        neuron_cfg: Passed to ``make_lif``.

    Shape:
        Inputs:  q, k, v each ``(T, B, H, Dh)``
        Output:  ``(T, B, H, Dh)``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        neuron_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.global_attn_lif = make_lif(neuron_cfg)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_scale: float,
    ) -> torch.Tensor:
        """Global attention forward.

        Args:
            q, k, v: ``(T, B, H, Dh)``
            global_scale: β_global scalar, 1 / sqrt(Dh * T).

        Returns:
            ``(T, B, H, Dh)``
        """
        # Sum over T -> (1, B, H, Dh); broadcast back to (T, B, H, Dh) via gate * v
        q_sum_all = q.sum(dim=0, keepdim=True)   # (1, B, H, Dh)
        k_sum_all = k.sum(dim=0, keepdim=True)
        gate_all = self.global_attn_lif((q_sum_all + k_sum_all) * global_scale)
        return gate_all * v                       # broadcast (1→T, B, H, Dh)


# ── STASA (dual-branch container) ────────────────────────────────────────────

@register("attention", "stasa")
class STASA(nn.Module):
    """Spiking Temporal-Aware Self-Attention (SWA + LRA dual branch).

    Args:
        dim: Model dimension D.
        num_heads: Attention heads H (D must be divisible by H).
        attention_window: SWA radius w; window size = 2w+1 (paper default 20).
        value_branch_kernel: Temporal kernel for V-branch depthwise 2D conv
            (9 for SHD/GSC, 7 for SSC per reference config; default 9).
        use_long_range: Enable LRA branch (default True).
        long_range_branch_factory: Optional callable returning an LRABranch-compatible
            module. Used by Track C to inject SpikingMamba. If provided, factory is
            called with no arguments; the result must accept (q, k, v, global_scale).
        neuron_cfg: Passed through to every ``make_lif`` call.
        use_bn: BatchNorm1d throughout (default True).
        dropout_rate: Dropout rate; 0 = disabled.

    Shape:
        Input:  ``(T, B, D)``
        Output: ``(T, B, D)``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_window: int = 20,
        value_branch_kernel: int = 9,
        use_long_range: bool = True,
        long_range_branch_factory: Callable | None = None,
        neuron_cfg: dict[str, Any] | None = None,
        use_bn: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.dh = dim // num_heads          # per-head dim
        self.attention_window = attention_window
        self.use_long_range = use_long_range

        # Precomputed scales; global_scale depends on T so computed in forward
        self.local_scale = 1.0 / math.sqrt(self.dh * (2 * attention_window + 1))

        # ── Q path ───────────────────────────────────────────────────────────
        # Conv1d in (B, D, T) space; BN + LIF in (T, B, D) space
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.q_bn = layer.BatchNorm1d(dim, step_mode="m") if use_bn else nn.Identity()
        self.q_lif = make_lif(neuron_cfg)

        # ── K path ───────────────────────────────────────────────────────────
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.k_bn = layer.BatchNorm1d(dim, step_mode="m") if use_bn else nn.Identity()
        self.k_lif = make_lif(neuron_cfg)

        # ── V path ───────────────────────────────────────────────────────────
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.v_bn = layer.BatchNorm1d(dim, step_mode="m") if use_bn else nn.Identity()
        self.v_lif = make_lif(neuron_cfg)

        # ── SWA branch gate LIF ──────────────────────────────────────────────
        self.local_attn_lif = make_lif(neuron_cfg)

        # ── LRA branch ──────────────────────────────────────────────────────
        if use_long_range:
            if long_range_branch_factory is not None:
                # Track C injection: factory called with no args; must expose same
                # forward(q, k, v, global_scale) -> (T, B, H, Dh) signature.
                self.lra_module = long_range_branch_factory()
            else:
                self.lra_module = LRABranch(dim, num_heads, neuron_cfg=neuron_cfg)

        # ── V-branch: DW2d pointwise + depthwise on per-head values ─────────
        # V reshaped to (B, H, T, Dh); treat as (B, H, T, Dh) 4D image where
        # DW2d kernel=(value_branch_kernel, 1) convolves T-axis within each head.
        # Reference: v_pw then v_dw then v_dw_lif (lines 255-259 backbone).
        vk = value_branch_kernel
        self.v_pw = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)
        self.v_dw = nn.Conv2d(
            num_heads, num_heads,
            kernel_size=(vk, 1),
            stride=1,
            padding=((vk - 1) // 2, 0),
            groups=num_heads,
            bias=False,
        )
        self.v_dw_lif = make_lif(neuron_cfg)

        # ── Output projections ────────────────────────────────────────────────
        # Two-stage: mattn then proj; both are Conv1d in (B, D, T) space.
        self.mattn_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.mattn_bn = layer.BatchNorm1d(dim, step_mode="m") if use_bn else nn.Identity()
        self.mattn_lif = make_lif(neuron_cfg)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.proj_bn = layer.BatchNorm1d(dim, step_mode="m") if use_bn else nn.Identity()
        self.proj_lif = make_lif(neuron_cfg)

        self._use_dp = dropout_rate > 0.0
        if self._use_dp:
            self.attn_dropout = layer.Dropout(dropout_rate, step_mode="m")
            self.mattn_dropout = layer.Dropout(dropout_rate, step_mode="m")
            self.proj_dropout = layer.Dropout(dropout_rate, step_mode="m")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _qkv_proj(
        self, conv: nn.Conv1d, bn: nn.Module, lif: nn.Module, x_bdt: torch.Tensor, T: int, B: int
    ) -> torch.Tensor:
        """Conv1d (B,D,T) → BN+LIF (T,B,D) → reshape to (T,B,H,Dh)."""
        out = conv(x_bdt)                          # (B, D, T)
        out = out.permute(2, 0, 1).contiguous()    # (T, B, D)
        out = bn(out.unsqueeze(-1)).squeeze(-1)    # BN 4D wrap
        out = lif(out)                             # (T, B, D)
        return out.reshape(T, B, self.num_heads, self.dh)  # (T, B, H, Dh)

    def _conv1d_tbd(self, conv: nn.Conv1d, bn: nn.Module, lif: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Project (T,B,D) via Conv1d in (B,D,T) space, return (T,B,D)."""
        T, B, D = x.shape
        x_bdt = x.permute(1, 2, 0).contiguous()
        x_bdt = conv(x_bdt)
        x = x_bdt.permute(2, 0, 1).contiguous()   # (T, B, D)
        x = bn(x.unsqueeze(-1)).squeeze(-1)
        x = lif(x)
        return x

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Dual-branch STASA forward.

        Args:
            x: ``(T, B, D)`` spiking input.
            attention_mask: Optional ``(B, T)`` bool mask (True=attend, False=pad).

        Returns:
            ``(T, B, D)``
        """
        T, B, D = x.shape
        global_scale = 1.0 / math.sqrt(self.dh * T)

        # Q/K/V projections  (T, B, H, Dh)
        x_bdt = x.permute(1, 2, 0).contiguous()   # (B, D, T)
        q = self._qkv_proj(self.q_conv, self.q_bn, self.q_lif, x_bdt, T, B)
        k = self._qkv_proj(self.k_conv, self.k_bn, self.k_lif, x_bdt, T, B)
        v_tbd_h = self._qkv_proj(self.v_conv, self.v_bn, self.v_lif, x_bdt, T, B)
        # v_tbd_h: (T, B, H, Dh)

        # Optional attention mask: zero-out padded positions in Q, K
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)   # (B, 1, T, 1)
            # q, k are (T, B, H, Dh) — convert mask to (T, B, 1, 1)
            mask_tbhd = mask.permute(2, 0, 1, 3)               # (T, B, 1, 1)
            q = q.masked_fill(~mask_tbhd, 0.0)
            k = k.masked_fill(~mask_tbhd, 0.0)

        # ── SWA branch  (local window attention) ─────────────────────────────
        # q, k: (T, B, H, Dh) — pad along T then unfold
        # Rearrange to (B, H, T, Dh) for F.pad + unfold on dim=2
        q_bhdt = q.permute(1, 2, 0, 3).contiguous()  # (B, H, T, Dh)
        k_bhdt = k.permute(1, 2, 0, 3).contiguous()
        w = self.attention_window
        # Pad T dimension: (left_T=w, right_T=w) — pad last 2 dims pair-wise right-to-left
        q_pad = F.pad(q_bhdt, (0, 0, w, w))           # (B, H, T+2w, Dh)
        k_pad = F.pad(k_bhdt, (0, 0, w, w))

        # Unfold window along T: → (B, H, T, Dh, 2w+1)
        q_win = q_pad.unfold(2, 2 * w + 1, 1)         # (B, H, T, Dh, 2w+1)
        k_win = k_pad.unfold(2, 2 * w + 1, 1)
        # Permute to (B, H, T, 2w+1, Dh) then sum over window dim
        q_win = q_win.permute(0, 1, 2, 4, 3).contiguous()  # (B, H, T, 2w+1, Dh)
        k_win = k_win.permute(0, 1, 2, 4, 3).contiguous()
        q_sum = q_win.sum(dim=3)                            # (B, H, T, Dh)
        k_sum = k_win.sum(dim=3)

        # Back to (T, B, H, Dh) for LIF (multi-step layout)
        q_sum = q_sum.permute(2, 0, 1, 3).contiguous()
        k_sum = k_sum.permute(2, 0, 1, 3).contiguous()
        gate_local = self.local_attn_lif((q_sum + k_sum) * self.local_scale)
        out_local = gate_local * v_tbd_h               # (T, B, H, Dh)

        # ── LRA branch  (global attention) ───────────────────────────────────
        if self.use_long_range:
            out_global = self.lra_module(q, k, v_tbd_h, global_scale)
            attn = out_local + out_global              # (T, B, H, Dh)
        else:
            attn = out_local

        # ── V-branch: DW2d on per-head values ────────────────────────────────
        # v_tbd_h (T, B, H, Dh) → (B, H, T, Dh) as 4D image for Conv2d
        v_bhdt = v_tbd_h.permute(1, 2, 0, 3).contiguous()  # (B, H, T, Dh)
        v_bhdt = self.v_pw(v_bhdt)
        v_bhdt = self.v_dw(v_bhdt)                           # (B, H, T, Dh)
        v_mask = v_bhdt.permute(2, 0, 1, 3).contiguous()     # (T, B, H, Dh) for LIF
        v_mask = self.v_dw_lif(v_mask)
        v_mask = v_mask.reshape(T, B, D)                      # (T, B, D)

        # ── Merge attn branches → project ────────────────────────────────────
        attn = attn.reshape(T, B, D)
        if self._use_dp:
            attn = self.attn_dropout(attn)

        x = self._conv1d_tbd(self.mattn_conv, self.mattn_bn, self.mattn_lif, attn)
        if self._use_dp:
            x = self.mattn_dropout(x)
        x = x + v_mask                                        # add V-branch (paper Eq 9)

        x = self._conv1d_tbd(self.proj_conv, self.proj_bn, self.proj_lif, x)
        if self._use_dp:
            x = self.proj_dropout(x)

        return x                                              # (T, B, D)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.num_heads}, dh={self.dh}, "
            f"window={self.attention_window}, lra={self.use_long_range}"
        )
