"""Shared modules: LIF wrappers, SEE, STASA, SCR-MLP, aggregator, classifier.

Importing this package triggers `@register(...)` side-effects — any
modules not imported here will be absent from the REGISTRY.
"""

from scommander.modules.aggregator import SumAggregator
from scommander.modules.classifier import ClassifierHead
from scommander.modules.lif import LIFNode, make_lif
from scommander.modules.plif import PLIFNode
from scommander.modules.scr_mlp import SCRMLP
from scommander.modules.see import SEE
from scommander.modules.stasa import LRABranch, STASA

# Spiking Mamba requires mamba_ssm; import is best-effort so the package
# still works on CPU-only / mamba-less environments (registry slot simply
# stays empty and resolve("long_range_branch", "spiking_mamba") will raise
# a clean KeyError with hint).
try:
    from scommander.modules.spiking_mamba import SpikingMambaBranch
except ImportError:
    SpikingMambaBranch = None

__all__ = [
    "ClassifierHead",
    "LIFNode",
    "LRABranch",
    "PLIFNode",
    "SCRMLP",
    "SEE",
    "STASA",
    "SpikingMambaBranch",
    "SumAggregator",
    "make_lif",
]
