"""SpikCommander: full trunk model.

Assembles SEE encoder → L × (STASA + SCR-MLP) blocks → aggregator → classifier.

Shape contract:
    Input:  ``(B, T, F_raw)``   raw binned spike features
    Output: ``(B, C)``          pre-softmax class logits

Reset semantics:
    SpikingJelly neurons maintain per-sample membrane state across forward calls.
    Call ``model.reset()`` (or equivalently ``functional.reset_net(model)``) between
    independent sequences — mandatory in training loops and evaluation.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from spikingjelly.activation_based import base as _sj_base

from scommander.models.registry import register, resolve


class _Block(nn.Module):
    """Single transformer-style block: STASA + SCR-MLP with residual connections."""

    def __init__(self, attn: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(T, B, D) -> (T, B, D) with pre-norm residuals."""
        x = x + self.attn(x, attention_mask=attention_mask)
        x = x + self.mlp(x)
        return x


@register("model", "spikcommander")
class SpikCommander(nn.Module):
    """SpikCommander baseline trunk.

    Args:
        in_features: F_raw (binned feature dim; 140 for all datasets post-binning).
        num_classes: Number of output classes (20 SHD; 35 SSC/GSC).
        dim: Model dimension D (128 SHD; 256 SSC/GSC).
        n_heads: Attention heads H (8 SHD; 16 SSC/GSC).
        depth: Number of (STASA + SCR-MLP) blocks L.
        window_radius: SWA window half-width w (default 20 for T=100).
        expansion: SCR-MLP expansion ratio α (default 4.0).
        long_range_branch_name: Registry key for the LRA branch (default ``'lra'``).
            Set to ``'spiking_mamba'`` for Track C (Phase 05).
        neuron_cfg: Dict forwarded to every ``make_lif`` call.
            Use ``{'backend': 'torch'}`` for CPU-only environments.
        dropout_rate: Dropout probability (0 = disabled).

    Shape:
        Input:  ``(B, T, F_raw)``
        Output: ``(B, C)``
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dim: int,
        n_heads: int,
        depth: int,
        window_radius: int = 20,
        expansion: float = 4.0,
        long_range_branch_name: str = "lra",
        neuron_cfg: dict[str, Any] | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth

        # SEE encoder: (B, T, F_raw) → (T, B, D)
        self.see = resolve("encoder", "see")(
            in_features=in_features,
            out_features=dim,
            kernel_size=7,
            neuron_cfg=neuron_cfg,
            dropout_rate=dropout_rate,
        )

        # Build STASA factory that injects the chosen long_range_branch
        def _make_lra_factory() -> Callable:
            """Return a zero-argument factory that builds the long-range branch."""
            lra_cls = resolve("long_range_branch", long_range_branch_name)

            def factory() -> nn.Module:
                return lra_cls(dim=dim, num_heads=n_heads, neuron_cfg=neuron_cfg)

            return factory

        # L blocks of (STASA + SCR-MLP)
        stasa_cls = resolve("attention", "stasa")
        mlp_cls = resolve("mlp", "scr_mlp")
        self.blocks = nn.ModuleList([
            _Block(
                attn=stasa_cls(
                    dim=dim,
                    num_heads=n_heads,
                    attention_window=window_radius,
                    long_range_branch_factory=_make_lra_factory(),
                    neuron_cfg=neuron_cfg,
                    dropout_rate=dropout_rate,
                ),
                mlp=mlp_cls(
                    in_features=dim,
                    expansion_ratio=expansion,
                    neuron_cfg=neuron_cfg,
                    dropout_rate=dropout_rate,
                ),
            )
            for _ in range(depth)
        ])

        # Sum-over-time aggregator: (T, B, D) → (B, D)
        self.aggregator = resolve("aggregator", "sum_over_time")()

        # Linear classifier: (B, D) → (B, C)
        self.head = resolve("classifier", "linear_head")(
            in_features=dim,
            num_classes=num_classes,
        )

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all LIF membrane states between independent sequences.

        Walks submodules directly and resets each ``MemoryModule`` — avoids
        the ``functional.reset_net`` warning on the non-MemoryModule trunk.
        """
        for m in self.modules():
            if isinstance(m, _sj_base.MemoryModule):
                m.reset()

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full model forward pass.

        Args:
            x: ``(B, T, F_raw)`` raw spike feature tensor.
            attention_mask: Optional ``(B, T)`` bool mask (True=attend, False=pad).

        Returns:
            ``(B, C)`` pre-softmax logits.
        """
        # (B, T, F_raw) → (T, B, D)
        x = self.see(x)

        # L blocks: residuals inside _Block.forward
        for blk in self.blocks:
            x = blk(x, attention_mask=attention_mask)

        # (T, B, D) → (B, D)
        x = self.aggregator(x)

        # (B, D) → (B, C)
        return self.head(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"
