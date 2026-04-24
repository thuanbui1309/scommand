"""SCR-MLP: Spiking Contextual Refinement MLP.

Implements paper Eqs 10-11:
    X'   = Linear(D→αD) + BN + LIF        [expand]
    X''  = DWConv(H1) cat H2              [selective refine]
    X''' = Linear(αD→D) + BN + LIF        [reduce]

Shape flow (all in multi-step layout):
    (T, B, D) -> expand -> (T, B, αD) -> split/refine -> (T, B, αD) -> reduce -> (T, B, D)

Reference: spikcommander_backbone.py SCRMLP (lines 289-430).
The gating logic: split expanded hidden into two halves H1, H2; apply depthwise
conv only to one half; concatenate back. Replicates the selective refinement
in the reference's ``outputs, res = x.chunk(2, dim=-1)`` pattern.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from scommander.models.registry import register
from scommander.modules.lif import make_lif


@register("mlp", "scr_mlp")
class SCRMLP(nn.Module):
    """Spiking Contextual Refinement MLP.

    Args:
        in_features: Input/output channel dim D.
        hidden_features: Expanded hidden dim αD; default = round(D * expansion_ratio).
        expansion_ratio: α; used only when hidden_features is None.
        kernel_size: Depthwise conv temporal kernel (paper default 31).
        neuron_cfg: Passed to ``make_lif``; use ``{'backend': 'torch'}`` for CPU.
        use_bn: BatchNorm1d after each linear stage (default True).
        dropout_rate: Applied after each spiking stage (0 = disabled).

    Shape:
        Input:  ``(T, B, D)``
        Output: ``(T, B, D)``
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        expansion_ratio: float = 4.0,
        kernel_size: int = 31,
        neuron_cfg: dict[str, Any] | None = None,
        use_bn: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        D = in_features
        hD = hidden_features if hidden_features is not None else int(round(D * expansion_ratio))
        self.in_features = D
        self.hidden_features = hD

        # ── Stage 1: expand  (D → αD) ───────────────────────────────────────
        # Pointwise conv on the conv-space side: (B, D, T) — plain nn.Conv1d
        self.pw1 = nn.Conv1d(D, D, kernel_size=1, bias=True)
        self.bn1 = layer.BatchNorm1d(D, step_mode="m") if use_bn else nn.Identity()
        self.lif1 = make_lif(neuron_cfg)

        # Linear expansion: acts on last dim of (T, B, D) -> (T, B, αD)
        self.fc1 = layer.Linear(D, hD, bias=False, step_mode="m")
        self.bn_fc1 = layer.BatchNorm1d(hD, step_mode="m") if use_bn else nn.Identity()
        self.lif_fc1 = make_lif(neuron_cfg)

        # ── Stage 2: selective contextual refine  (αD/2 branch) ─────────────
        # DW conv operates in (B, αD/2, T) layout; plain nn.Conv1d with groups
        self.dw = nn.Conv1d(
            hD // 2, hD // 2,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=hD // 2,
            bias=False,
        )
        self.bn2 = layer.BatchNorm1d(hD // 2, step_mode="m") if use_bn else nn.Identity()
        self.lif2 = make_lif(neuron_cfg)

        # ── Stage 3: reduce  (αD → D) ────────────────────────────────────────
        self.fc2 = layer.Linear(hD, D, bias=False, step_mode="m")
        self.bn_fc2 = layer.BatchNorm1d(D, step_mode="m") if use_bn else nn.Identity()
        self.lif_fc2 = make_lif(neuron_cfg)

        # Pointwise conv reduce: (B, D, T)
        self.pw2 = nn.Conv1d(D, D, kernel_size=1, bias=True)
        self.bn3 = layer.BatchNorm1d(D, step_mode="m") if use_bn else nn.Identity()
        self.lif3 = make_lif(neuron_cfg)

        # Optional dropout after each LIF stage
        self._use_dp = dropout_rate > 0.0
        if self._use_dp:
            self.dp1 = layer.Dropout(dropout_rate, step_mode="m")
            self.dp2 = layer.Dropout(dropout_rate, step_mode="m")
            self.dp3 = layer.Dropout(dropout_rate, step_mode="m")
            self.dp_fc1 = layer.Dropout(dropout_rate, step_mode="m")
            self.dp_fc2 = layer.Dropout(dropout_rate, step_mode="m")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _bn_3d(self, bn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply spikingjelly BN1d (needs 4D) to 3D (T, B, D) tensor."""
        return bn(x.unsqueeze(-1)).squeeze(-1)

    def _pw_forward(self, pw: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """Apply plain Conv1d on (T, B, D) by routing through (B, D, T) layout."""
        T, B, D = x.shape
        x_bdt = x.permute(1, 2, 0).contiguous()   # (B, D, T)
        x_bdt = pw(x_bdt)
        return x_bdt.permute(2, 0, 1).contiguous()  # (T, B, D)

    def _dw_forward(self, x: torch.Tensor) -> torch.Tensor:
        """DW conv on (T, B, αD/2) half via (B, αD/2, T)."""
        T, B, C = x.shape
        x_bct = x.permute(1, 2, 0).contiguous()
        x_bct = self.dw(x_bct)
        return x_bct.permute(2, 0, 1).contiguous()  # (T, B, C)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(T, B, D)`` spike tensor.

        Returns:
            ``(T, B, D)`` spike tensor.
        """
        # Stage 1a: pointwise conv + BN + LIF  (T, B, D)
        x = self._pw_forward(self.pw1, x)
        x = self._bn_3d(self.bn1, x)
        x = self.lif1(x)
        if self._use_dp:
            x = self.dp1(x)

        # Stage 1b: linear expand  (T, B, D) -> (T, B, αD)
        x = self.fc1(x)
        x = self._bn_3d(self.bn_fc1, x)
        x = self.lif_fc1(x)
        if self._use_dp:
            x = self.dp_fc1(x)

        # Stage 2: selective refine — split, DW on first half, cat back
        # Reference: ``outputs, res = x.chunk(2, dim=-1)``
        outputs, res = x.chunk(2, dim=-1)           # each (T, B, αD/2)
        outputs = self._dw_forward(outputs)
        outputs = self._bn_3d(self.bn2, outputs)
        outputs = self.lif2(outputs)
        if self._use_dp:
            outputs = self.dp2(outputs)
        x = torch.cat((res, outputs), dim=-1)       # (T, B, αD)

        # Stage 3a: linear reduce  (T, B, αD) -> (T, B, D)
        x = self.fc2(x)
        x = self._bn_3d(self.bn_fc2, x)
        x = self.lif_fc2(x)
        if self._use_dp:
            x = self.dp_fc2(x)

        # Stage 3b: pointwise conv + BN + LIF  (T, B, D)
        x = self._pw_forward(self.pw2, x)
        x = self._bn_3d(self.bn3, x)
        x = self.lif3(x)
        if self._use_dp:
            x = self.dp3(x)

        return x

    def extra_repr(self) -> str:
        return f"in={self.in_features}, hidden={self.hidden_features}"
