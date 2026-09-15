"""state_transitions: retain the first sample, every change of value, and the last sample.

Contract (docs/contracts.md, "state_transitions"): the first sample, every sample whose value differs from the
previous sample (NaN equals NaN), and the last sample are retained. Hold reconstruction equals the source value at
every source timestamp, and docs/verification.md compares values bitwise.

Value equality is therefore bitwise with one exception taken from the contract: any two NaNs are equal. So 0.0 and
-0.0 differ, while NaNs with different payloads do not. A NaN-payload change is not counted as a transition, but
the sample is still retained so that hold reconstruction reproduces the source bits exactly. The retained set is
the first sample, the last sample and every sample whose bit pattern differs from the previous sample. That is a
superset of the contract's samples and adds only NaN-payload changes.

Works for bool, integer and float scalar signals; vector signals raise ValueError. Evidence:
`transitions` (number of samples whose value differs from the previous one, as defined above), `distinct_count`
(exact number of distinct values under the same equality) and `distinct_values` (the distinct values sorted
ascending, -0.0 before 0.0, one NaN last; capped list). An empty signal is `not_applicable`; constant and all-NaN
signals pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..sampleset import SampleSet
from ._common import EVIDENCE_LIMIT, STATUS_NOT_APPLICABLE, STATUS_PASS, OpResult, capped, fnum

if TYPE_CHECKING:
    from ..signals import Signal


def _bits(v: np.ndarray) -> np.ndarray:
    """Unsigned-integer view of float values with the same width, so `!=` compares bit patterns."""
    c = np.ascontiguousarray(v)
    size = c.dtype.itemsize
    if size in (1, 2, 4, 8):
        return c.view(f"u{size}")
    return c.view(np.uint8).reshape(c.shape[0], size)  # extended precision: compare byte rows


def _differs(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    d = a_bits != b_bits
    return d.any(axis=1) if d.ndim == 2 else d


def change_masks(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of length n - 1 comparing v[j + 1] with v[j]: `(value_changed, bits_changed)`.

    `value_changed` uses the contract's equality (bitwise, except that any two NaNs are equal). `bits_changed` is
    True whenever the bit patterns differ; it equals `value_changed` except at NaN-payload changes.
    """
    if v.dtype.kind != "f":
        changed = v[1:] != v[:-1]
        return changed, changed
    bits = _bits(v)
    bits_changed = _differs(bits[1:], bits[:-1])
    nan = np.isnan(v)
    value_changed = bits_changed & ~(nan[1:] & nan[:-1])
    return value_changed, bits_changed


def change_mask(v: np.ndarray) -> np.ndarray:
    """Boolean mask of length n - 1: True at j when v[j + 1] differs from v[j] (bitwise, NaN equals NaN)."""
    return change_masks(v)[0]


def distinct_values(v: np.ndarray) -> np.ndarray:
    """Distinct values under the contract's equality, sorted ascending with -0.0 before 0.0 and one NaN last."""
    if v.dtype.kind != "f":
        return np.unique(v)
    nan = np.isnan(v)
    finite_or_inf = np.ascontiguousarray(v[~nan])
    if finite_or_inf.dtype.itemsize in (1, 2, 4, 8):
        uniq = np.unique(_bits(finite_or_inf)).view(finite_or_inf.dtype)
    else:
        rows = np.unique(_bits(finite_or_inf), axis=0)
        uniq = np.ascontiguousarray(rows).view(finite_or_inf.dtype).reshape(-1)
    # Only 0.0 and -0.0 are equal in value with different bits; the sign key puts -0.0 first.
    uniq = uniq[np.lexsort((~np.signbit(uniq), uniq))]
    if nan.any():
        uniq = np.concatenate([uniq, v[np.flatnonzero(nan)[:1]]])
    return uniq


def evaluate(sig: Signal, params: Mapping[str, object], bits: Mapping[str, int]) -> OpResult:
    bit = int(bits["main"])
    v = sig.v
    if v.ndim != 1 or sig.kind == "vector":
        raise ValueError(f"state_transitions needs a scalar signal; {sig.name!r} has shape {tuple(v.shape)}")
    if v.dtype.kind not in "biuf":
        raise ValueError(f"state_transitions supports bool, integer and float values, got dtype {v.dtype}")
    n = int(v.shape[0])
    if n == 0:
        evidence = {"transitions": 0, "distinct_count": 0, "distinct_values": []}
        return OpResult(SampleSet.empty(), evidence, STATUS_NOT_APPLICABLE, ["empty signal"])

    value_changed, bits_changed = change_masks(v)
    retained = np.flatnonzero(bits_changed).astype(np.int64) + 1
    idx = np.concatenate([np.array([0, n - 1], dtype=np.int64), retained])
    distinct = distinct_values(v)
    evidence = {
        "transitions": int(np.count_nonzero(value_changed)),
        "distinct_count": int(distinct.shape[0]),
        "distinct_values": capped([fnum(x) for x in distinct[:EVIDENCE_LIMIT]]),
    }
    return OpResult(samples=SampleSet.from_points(idx, bit), evidence=evidence, status=STATUS_PASS)
