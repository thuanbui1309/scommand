"""SEE (Spiking Embedding Extractor).

Implements paper Eqs 4-5:
    X'  = SN(BN(DConv(PConv(X))))           (Eq 4)
    X'' = SN(BN(Linear(X'))) + X'            (Eq 5)

Shape flow (mirrors reference ``spikcommder.SEE``):

    (B, T, F_raw)
        --transpose-->      (B, F_raw, T)
        --PointwiseConv1d-->(B, D, T)           # plain nn.Conv1d, k=1
        --DepthwiseConv1d-->(B, D, T)           # plain nn.Conv1d, k=7, groups=D
        --transpose-->      (T, B, D)
        --BN1d(m) + LIF -->  (T, B, D)          # spikingjelly multi-step space
        --(keep residual)-->
        --Linear(m) + BN + LIF-->(T, B, D)
        --residual add-->   (T, B, D)

Output is in ``(T, B, D)`` spikingjelly multi-step layout; downstream trunk
consumes the same layout.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from scommander.models.registry import register
from scommander.modules.lif import make_lif


@register("encoder", "see")
class SEE(nn.Module):
    """Spiking Embedding Extractor.

    Args:
        in_features: F_raw (140 for SHD/SSC binned; 140 for GSC Mel default).
        out_features: D (128 for SHD; 256 for SSC/GSC).
        kernel_size: Depthwise temporal kernel (paper default 7).
        use_bn: Enable BatchNorm1d (default True).
        use_dw_bias: Bias on depthwise conv (reference config-controlled).
        dropout_rate: 0 = disabled.
        neuron_cfg: Passed through to ``make_lif``.

    Shape:
        Input:  ``(B, T, F_raw)``
        Output: ``(T, B, D)``   # multi-step spikingjelly layout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 7,
        use_bn: bool = True,
        use_dw_bias: bool = False,
        dropout_rate: float = 0.0,
        neuron_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        D = out_features
        pad = (kernel_size - 1) // 2

        # Conv stages run in (B, C, T); plain nn.Conv1d, no spikingjelly wrapping.
        self.pwconv = nn.Conv1d(in_features, D, kernel_size=1, stride=1, padding=0, bias=True)
        self.dwconv = nn.Conv1d(D, D, kernel_size=kernel_size, stride=1, padding=pad,
                                groups=D, bias=use_dw_bias)

        # Multi-step spiking stages consume (T, B, D).
        self.bn1 = layer.BatchNorm1d(D, step_mode="m") if use_bn else nn.Identity()
        self.lif1 = make_lif(neuron_cfg)
        self.dropout1 = layer.Dropout(dropout_rate, step_mode="m") if dropout_rate > 0 else nn.Identity()

        self.linear = layer.Linear(D, D, bias=False, step_mode="m")
        self.bn2 = layer.BatchNorm1d(D, step_mode="m") if use_bn else nn.Identity()
        self.lif2 = make_lif(neuron_cfg)
        self.dropout2 = layer.Dropout(dropout_rate, step_mode="m") if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F_raw) -> (B, F_raw, T)
        x = x.transpose(1, 2).contiguous()
        x = self.pwconv(x)                 # (B, D, T)
        x = self.dwconv(x)                 # (B, D, T)

        # (B, D, T) -> (T, B, D)  — now in spikingjelly multi-step layout
        x = x.permute(2, 0, 1).contiguous()

        x = self.bn1(x)                    # (T, B, D)
        x = self.lif1(x)
        x = self.dropout1(x)

        residual = x
        x = self.linear(x)                 # (T, B, D) — Linear acts on last dim
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.dropout2(x)
        x = x + residual                   # Eq 5

        return x                           # (T, B, D)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"
