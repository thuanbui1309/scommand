"""Manual Mamba SSM SOP counter (Phase 05 mandate).

The ``syops`` package doesn't recognise selective-state-space kernels and
reports 0 ops for ``mamba_ssm.Mamba`` modules — fair-energy comparisons in
Phase 08 would systematically underestimate Track C if we relied on it.

Manual derivation per Mamba paper (Gu & Dao 2023, Algorithm 1) — selective
SSM scan with input-dependent (B, C, Δ) projections. For a single Mamba
block with model dim D, state dim N, conv width K, expand factor E:

  Inner dim     D_inner = E * D
  Per timestep ops (one sequence position, batch=1):
    1. in_proj:     2 * D * D_inner  (Linear, weight+bias for x + z gates)
    2. conv1d:      K * D_inner       (depthwise causal conv)
    3. x_proj:      D_inner * (N + N + dt_rank)  (input -> B, C, dt)
    4. dt_proj:     dt_rank * D_inner            (dt parameter)
    5. SSM scan:    D_inner * N * 2               (A·h + B·x once per step)
                    + D_inner * N                 (C·h read-out)
                    = 3 * D_inner * N
    6. out_proj:    D_inner * D

Total per-step: ~ 2*D*D_inner + K*D_inner + D_inner*(2N+dt_rank) +
                   dt_rank*D_inner + 3*D_inner*N + D_inner*D

For full sequence: multiply by T (timesteps).
For batched run: multiply by B.

This counter returns integer SOPs (synaptic operations, paper Eq 19/20
convention: equivalent to MAC count). Energy in paper Eq 20:
  E_total = E_MAC * SOPs_ANN + E_AC * SOPs_SNN
For Mamba (non-spiking SSM hidden state): all ops count as ANN MACs.

Unit test in ``tests/test_mamba_sop_count.py`` validates within ±5% of
PyTorch profiler empirical count.
"""

from __future__ import annotations

import math
from typing import Any


def mamba_block_sops(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dt_rank: int | None = None,
    seq_len: int = 100,
    batch_size: int = 1,
) -> int:
    """SOP count for one Mamba block over (batch_size, seq_len, d_model).

    Args:
        d_model: model dim D.
        d_state: SSM state size N (Mamba default 16).
        d_conv: causal conv width (default 4).
        expand: D_inner = expand * D (default 2).
        dt_rank: dt projection rank; default = ceil(D/16) per Mamba.
        seq_len: sequence length T.
        batch_size: batch B.

    Returns:
        Total integer SOP count.
    """
    d_inner = expand * d_model
    if dt_rank is None:
        dt_rank = math.ceil(d_model / 16)

    per_step = (
        2 * d_model * d_inner           # in_proj (x + z gates)
        + d_conv * d_inner              # depthwise causal conv
        + d_inner * (2 * d_state + dt_rank)  # x_proj -> (B, C, dt)
        + dt_rank * d_inner             # dt_proj
        + 3 * d_inner * d_state         # SSM scan: A·h + B·x + C·h
        + d_inner * d_model             # out_proj
    )
    return per_step * seq_len * batch_size


def spiking_mamba_branch_sops(
    d_model: int,
    num_heads: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    seq_len: int = 100,
    batch_size: int = 1,
) -> dict[str, int]:
    """SOP breakdown for one ``SpikingMambaBranch``.

    Components: Mamba SSM scan + gate_proj Linear + LIF spike op (counted as
    AC, accumulate-only, ~1 op per spike-equivalent).
    """
    mamba_sops = mamba_block_sops(
        d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
        seq_len=seq_len, batch_size=batch_size,
    )
    gate_proj_sops = d_model * d_model * seq_len * batch_size
    lif_sops = d_model * seq_len * batch_size  # accumulate op per neuron-step

    return {
        "mamba_ssm": mamba_sops,
        "gate_proj": gate_proj_sops,
        "lif_accumulate": lif_sops,
        "total": mamba_sops + gate_proj_sops + lif_sops,
    }
