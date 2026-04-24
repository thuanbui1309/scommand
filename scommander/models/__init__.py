"""Models: baseline SpikCommander + Track A/B/C variants.

Importing this package is the canonical entry point; it populates the
global component registry by touching every module that declares
`@register(...)`.
"""

from scommander.models.registry import REGISTRY, build_model, register, resolve

__all__ = ["REGISTRY", "build_model", "register", "resolve"]
