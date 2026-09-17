"""Requirement checks: test simulation runs against a requirements table and show what went wrong.

    baslt requirements init -o reqs.xlsx          a starter workbook with every kind of check
    baslt requirements lint reqs.xlsx             read it and report every problem, without running anything
    baslt check run.h5 -r reqs.xlsx -o out        PASS / WARN / FAIL per requirement, Excel, HTML report

See docs/requirements.md for the table layout, the check types and their exact semantics.
"""

from __future__ import annotations

from importlib import import_module

_LAZY = {
    "load": "baslt.reqs.api",
    "lint": "baslt.reqs.api",
    "init_template": "baslt.reqs.api",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'baslt.reqs' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
