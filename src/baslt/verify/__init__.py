"""Independent verifier: checks artifacts against the fidelity contracts without compiler code.

Names resolve lazily, so `import baslt.verify` stays free of numpy and costs almost nothing.
"""

from __future__ import annotations

from importlib import import_module

# Public name -> the module that defines it. Resolved on first access.
_LAZY: dict[str, str] = {
    "Check": "baslt.verify.structure",
    "VerifyResult": "baslt.verify.checks",
    "structure_checks": "baslt.verify.structure",
    "verify_artifact": "baslt.verify.checks",
    "render": "baslt.verify.table",
    "to_json": "baslt.verify.table",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'baslt.verify' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
