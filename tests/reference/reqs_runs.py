"""Small runs and contexts for requirement-check tests."""

from __future__ import annotations

import numpy as np

from baslt.reqs.config import load_config
from baslt.reqs.evaluate import RunContext
from baslt.signals import Run, SourceMeta, normalize_signal


def run(**signals) -> Run:
    """`name=(t, v)` or `name=(t, v, unit)` or `name=(t, v, unit, kind)`; names use '__' for '/'."""
    out = {}
    for key, spec in signals.items():
        t, v, *rest = spec
        unit = rest[0] if rest else None
        kind = rest[1] if len(rest) > 1 else None
        name = key.replace("__", "/")
        sig, _ = normalize_signal(name, np.asarray(t, dtype=float), np.asarray(v), path=name, unit=unit, kind=kind)
        out[name] = sig
    return Run(signals=out, meta=SourceMeta(path=None, format="numpy", size_bytes=None))


def context(signals: dict, mapping: dict | None = None, params: dict | None = None) -> RunContext:
    return RunContext(run(**signals), load_config(mapping or {}), params=params)


def evaluate(ctx: RunContext, text: str, grid_from: str | None = None):
    """Series (or Tri) of an expression on its own grid, or its single value."""
    bound = ctx.bind(text)
    if bound.type.series:
        names = bound.signals if grid_from is None else [grid_from]
        grid = ctx.grid_for(names)
        return ctx.value(bound.root, grid)
    return ctx.scalar(bound.root, None)
