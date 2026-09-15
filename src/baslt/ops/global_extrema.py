"""global_extrema: retain the lowest-index finite maximum and minimum of every component.

Contract (docs/contracts.md, "global_extrema"): for each component, the retained samples include the source
sample with the maximum finite value and the one with the minimum finite value, lowest index on ties.

Finiteness follows the contract's convention: a sample is finite when every component is finite
(`_common.finite_mask`). One sample mask is built for the whole signal and each component's max and min are taken
over the finite samples only, so a vector sample with a NaN or infinity in any component is never picked. When
there is no finite sample (including an empty signal) the status is `not_applicable`, nothing is retained and every
component reports `max`/`min` as None. A constant signal passes; its max and min are both the first finite sample.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..sampleset import SampleSet
from ._common import STATUS_NOT_APPLICABLE, STATUS_PASS, OpResult, capped, components, finite_mask, fnum

if TYPE_CHECKING:
    from ..signals import Signal


def component_extrema(x: np.ndarray, finite: np.ndarray | None = None) -> tuple[int | None, int | None]:
    """(argmax, argmin) of a 1-D array over the samples flagged in `finite`, lowest index on ties.

    `finite` defaults to the finiteness of `x` itself. Returns (None, None) when no sample is flagged.
    """
    if x.shape[0] == 0:
        return None, None
    if finite is None:
        finite = finite_mask(x)
    if finite.all():
        return int(np.argmax(x)), int(np.argmin(x))
    rows = np.flatnonzero(finite)
    if rows.shape[0] == 0:
        return None, None
    # Indexing keeps the dtype (exact for large integers); argmax/argmin return the first hit, and rows is increasing.
    sub = x[rows]
    return int(rows[np.argmax(sub)]), int(rows[np.argmin(sub)])


def _point(sig: Signal, col: np.ndarray, i: int | None) -> dict | None:
    if i is None:
        return None
    return {"index": int(i), "t": fnum(sig.t[i]), "value": fnum(col[i])}


def evaluate(sig: Signal, params: Mapping[str, object], bits: Mapping[str, int]) -> OpResult:
    bit = int(bits["main"])
    v = sig.v
    fin = finite_mask(v)
    picks: list[int] = []
    items: list[dict] = []
    for k, col in components(v):
        hi, lo = component_extrema(col, fin)
        if hi is not None:
            picks.extend((hi, lo))
        items.append({"component": k, "max": _point(sig, col, hi), "min": _point(sig, col, lo)})

    notes: list[str] = []
    if not picks:
        notes.append("empty signal" if v.shape[0] == 0 else "no finite sample")
    status = STATUS_NOT_APPLICABLE if notes else STATUS_PASS
    samples = SampleSet.from_points(picks, bit) if picks else SampleSet.empty()
    return OpResult(samples=samples, evidence={"components": capped(items)}, status=status, notes=notes)
