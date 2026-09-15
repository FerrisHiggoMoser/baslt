"""Shared types and helpers for hard-requirement operators.

Every operator module exposes

    evaluate(sig: Signal, params: Mapping[str, object], bits: Mapping[str, int]) -> OpResult

`params` are already bound: values and deltas in the signal's unit as floats, durations in seconds,
enums as strings. `bits` maps the operator's role names to legend bit numbers:

    global_extrema, window_extrema, state_transitions, threshold_crossing: {"main": b}
    local_extrema: {"peak": b1, "base": b2}
    violation: {"edge": b1, "worst": b2}

The returned SampleSet only contains indices of `sig` tagged with those bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..sampleset import SampleSet

if TYPE_CHECKING:
    import numpy as np

    from ..signals import Signal

EVIDENCE_LIMIT = 256

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_NOT_APPLICABLE = "not_applicable"


@dataclass(slots=True)
class OpResult:
    samples: SampleSet
    evidence: dict
    status: str = STATUS_PASS
    notes: list[str] = field(default_factory=list)
    raw: object | None = None  # operator-specific cache reused by closure repair


def finite_mask(v: np.ndarray) -> np.ndarray:
    """True where every component of the sample is finite. Integer and bool values are always finite."""
    import numpy as np

    if v.dtype.kind in "biu":
        return np.ones(v.shape[0], dtype=bool)
    if v.ndim == 1:
        return np.isfinite(v)
    return np.isfinite(v).all(axis=1)


def extent_samples(n: int, bit: int) -> SampleSet:
    """First and last sample of a signal."""
    if n <= 0:
        return SampleSet.empty()
    return SampleSet.from_points([0, n - 1], bit)


def gap_samples(v: np.ndarray, bit: int) -> SampleSet:
    """First and last sample of every non-finite run, plus the finite samples just before and after it."""
    import numpy as np

    n = v.shape[0]
    fin = finite_mask(v)
    if n == 0 or fin.all():
        return SampleSet.empty()
    bad = ~fin
    edges = np.diff(np.concatenate([[False], bad, [False]]).astype(np.int8))
    run_starts = np.flatnonzero(edges == 1)
    run_stops = np.flatnonzero(edges == -1)  # exclusive
    pts = np.concatenate([run_starts - 1, run_starts, run_stops - 1, run_stops])
    pts = pts[(pts >= 0) & (pts < n)]
    return SampleSet.from_points(pts, bit)


def capped(items: list, limit: int = EVIDENCE_LIMIT) -> list:
    return items[:limit]


def fnum(x) -> float | int | str:
    """A JSON-friendly scalar: Python float/int (non-finite floats stay floats; canonical_json converts them)."""
    import numpy as np

    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def value_of(v: np.ndarray, i: int):
    """JSON-friendly value of sample i (list for vector signals)."""
    if v.ndim == 1:
        return fnum(v[i])
    return [fnum(c) for c in v[i]]


def components(v: np.ndarray):
    """Yield (component_index, 1-D view) for scalar or vector values."""
    if v.ndim == 1:
        yield 0, v
    else:
        for k in range(v.shape[1]):
            yield k, v[:, k]
