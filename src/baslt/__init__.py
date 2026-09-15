"""Baslt: compile simulation runs into small review artifacts with verified fidelity contracts.

Importing this package is deliberately cheap. Heavy dependencies such as numpy are only
imported when a public function is first used, so importing baslt inside a pipeline costs
almost nothing.
"""

from __future__ import annotations

from importlib import import_module

from ._version import __version__

# Public name -> (module, attribute). Resolved on first access.
_LAZY: dict[str, tuple[str, str]] = {}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'baslt' has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
