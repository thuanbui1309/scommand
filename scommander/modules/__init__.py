"""Shared modules: LIF wrappers, SEE, STASA, SCR-MLP, aggregator, classifier.

Importing this package triggers `@register(...)` side-effects — any
modules not imported here will be absent from the REGISTRY.
"""

from scommander.modules.aggregator import SumAggregator
from scommander.modules.classifier import ClassifierHead
from scommander.modules.lif import LIFNode, make_lif
from scommander.modules.see import SEE

__all__ = ["ClassifierHead", "LIFNode", "SEE", "SumAggregator", "make_lif"]
