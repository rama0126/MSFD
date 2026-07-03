"""
Package initializer for the `models` module.
Exports lowercase aliases and dynamic attribute access for model classes.
"""

from .init_continual_model import ContinualModel


__all__ = [

]

# PEP 562: support `from models import <name>` and attribute access
def __getattr__(name):
    name_lower = name.lower()
    if name_lower == 'msfd':
        from .msfd import MSFD
        return MSFD
    if name_lower == 'base':
        return ContinualModel
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
