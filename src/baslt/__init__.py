"""Baslt: compile simulation runs into small review artifacts with verified fidelity contracts.

Importing this package is deliberately cheap. Heavy dependencies such as numpy are only
imported when a public function is first used, so importing baslt inside a pipeline costs
almost nothing.
"""

from __future__ import annotations

from importlib import import_module

from ._version import __version__

# Public name -> (module, attribute). Resolved on first access.
_LAZY: dict[str, tuple[str, str]] = {
    "compile": ("baslt.api", "compile"),
    "inspect": ("baslt.api", "inspect"),
    "verify_artifact": ("baslt.api", "verify"),
    "read_artifact": ("baslt.container.reader", "read_artifact"),
    "Artifact": ("baslt.container.reader", "Artifact"),
    "CompileResult": ("baslt.api", "CompileResult"),
    "init_policy": ("baslt.api", "init_policy"),
    "explain": ("baslt.advice", "explain"),
    "check": ("baslt.reqs.api", "check"),
    "lint_requirements": ("baslt.reqs.api", "lint"),
    "init_requirements": ("baslt.reqs.api", "init_template"),
}

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
