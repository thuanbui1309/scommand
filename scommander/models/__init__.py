"""Models: baseline SpikCommander + Track A/B/C variants.

Importing this package is the canonical entry point; it populates the
global component registry by touching every module that declares
`@register(...)`.
"""

from scommander.models.registry import REGISTRY, build_model, register, resolve
from scommander.models.spikcommander import SpikCommander  # noqa: F401 — triggers @register

__all__ = ["REGISTRY", "SpikCommander", "build_model", "register", "resolve"]
