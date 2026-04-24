"""SOPs (Synaptic Operations) + energy counting for SNN variants.

Wraps syops-counter with spike-aware hooks. Handles the SpikCommander energy
formulation (Eq 19/20 of baseline paper):
    E = E_AC * (sum of SOPs in spike-driven layers)
        [+ E_MAC * FLOPs of first layer if input is real-valued (GSC case)]

Energy constants (45nm hardware, per paper):
    E_AC  = 0.9 pJ
    E_MAC = 4.6 pJ
"""
from __future__ import annotations

from dataclasses import dataclass

E_AC_PJ = 0.9  # AC op energy (pJ), 45nm hardware
E_MAC_PJ = 4.6  # MAC op energy (pJ), 45nm hardware


@dataclass
class EnergyReport:
    params_m: float  # millions of parameters
    sops_g: float  # synaptic ops, billions
    flops_g: float  # MAC ops, billions (for first layer if real-valued input)
    energy_mj: float  # total estimated energy, millijoules
    firing_rates: dict[str, float]  # per-layer mean firing rate


def compute_energy(sops_g: float, flops_g: float = 0.0) -> float:
    """Compute total theoretical energy in millijoules.

    Args:
        sops_g: total SOPs in billions (spike-driven AC operations).
        flops_g: total FLOPs in billions for real-valued MAC ops (e.g., GSC first layer).

    Returns:
        energy in millijoules.
    """
    # Convert G ops to total ops, multiply by pJ, convert pJ to mJ (1e-9)
    sop_energy_pj = sops_g * 1e9 * E_AC_PJ
    flop_energy_pj = flops_g * 1e9 * E_MAC_PJ
    return (sop_energy_pj + flop_energy_pj) * 1e-9


def count_model(model, input_shape: tuple, time_steps: int) -> EnergyReport:
    """Count SOPs, FLOPs, firing rates for an SNN model.

    Placeholder. Phase 01 wires in real syops-counter integration + SNN hooks.
    Phase 05 adds manual Spiking Mamba SOP derivation as a required gate.
    """
    raise NotImplementedError(
        "Implemented in Phase 01. Integrates syops-counter with firing-rate hooks."
    )
