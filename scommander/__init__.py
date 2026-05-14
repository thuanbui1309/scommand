"""SCommander - Spiking Transformer with SSM-Attention Hybrid for SCR."""

# spikingjelly's cupy LIF/PLIF kernel still references np.int (removed in
# numpy >= 1.20). Restore the alias before any submodule imports spikingjelly.
import numpy as _np
if not hasattr(_np, "int"):
    _np.int = int  # type: ignore[attr-defined]

__version__ = "0.0.1"
