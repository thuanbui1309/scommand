"""Models: baseline SpikCommander + Track A/B/C variants.

Importing this package is the canonical entry point; it populates the
global component registry by touching every module that declares
`@register(...)`.
"""

from scommander.models.registry import REGISTRY, build_model, register, resolve

# Importing scommander.modules triggers @register side-effects for every
# building block (LIF, SEE, STASA, SCR-MLP, aggregator, classifier). Must
# happen BEFORE spikcommander import so build_model can resolve them.
import scommander.modules  # noqa: F401 — side-effect: populates REGISTRY

from scommander.models.spikcommander import SpikCommander  # noqa: F401 — triggers @register

__all__ = ["REGISTRY", "SpikCommander", "build_model", "register", "resolve"]
