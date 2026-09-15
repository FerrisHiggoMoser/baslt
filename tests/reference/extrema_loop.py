"""Naive per-sample reference loops for global_extrema, window_extrema and state_transitions.

Written straight from docs/contracts.md with plain Python loops and the `math` module. They do not call into
baslt and are only meant for tests.

Conventions taken from the contract text:

- A sample is finite when every component is finite. Extrema are taken over finite samples only.
- state_transitions compares values bitwise (docs/verification.md: "Values: bitwise equality"), except that any two
  NaNs are equal. Hold reconstruction must reproduce the source bits, so a sample whose bits differ from the
  previous sample is retained even when it is not a transition (a NaN-payload change).
"""

from __future__ import annotations

import math


def _is_finite(x) -> bool:
    if isinstance(x, (bool, int)):
        return True
    return math.isfinite(float(x))


def _columns(v) -> list[list]:
    """List of components, each a Python list of per-sample values."""
    rows = v.tolist() if hasattr(v, "tolist") else list(v)
    if not rows or not isinstance(rows[0], list):
        if getattr(v, "ndim", 1) == 2:  # (0, k) array: k empty components
            return [[] for _ in range(v.shape[1])]
        return [rows]
    k = len(rows[0])
    return [[row[c] for row in rows] for c in range(k)]


def _sample_finite(cols: list[list]) -> list[bool]:
    """Per sample: True when every component of that sample is finite."""
    n = len(cols[0]) if cols else 0
    out = []
    for i in range(n):
        ok = True
        for col in cols:
            if not _is_finite(col[i]):
                ok = False
        out.append(ok)
    return out


def global_extrema_loop(v) -> tuple[list[tuple[int | None, int | None]], set[int]]:
    """Per component (max index, min index) over finite samples, lowest index on ties; and the retained set."""
    cols = _columns(v)
    finite = _sample_finite(cols)
    result = []
    retained: set[int] = set()
    for col in cols:
        best_hi = None
        best_lo = None
        for i, x in enumerate(col):
            if not finite[i]:
                continue
            if best_hi is None or x > col[best_hi]:
                best_hi = i
            if best_lo is None or x < col[best_lo]:
                best_lo = i
        result.append((best_hi, best_lo))
        if best_hi is not None:
            retained.add(best_hi)
            retained.add(best_lo)
    return result, retained


def window_buckets_loop(t, interval: float, origin: float) -> list[list[int]]:
    """For every sample, the list of buckets it belongs to (its own bucket first)."""
    t = [float(x) for x in t]
    n = len(t)
    if n == 0:
        return []
    delta = 8.0 * math.ulp(max(abs(t[0]), abs(t[n - 1])) / interval)
    out = []
    for i in range(n):
        u = (t[i] - origin) / interval
        m = math.floor(u + delta)
        buckets = [m]
        boundary = math.floor(u + 0.5)
        if abs(u - boundary) <= delta:
            # The two buckets touching integer boundary K are K - 1 and K; add the one m is not.
            buckets.append(boundary - 1 if m >= boundary else boundary)
        out.append(buckets)
    return out


def window_extrema_loop(t, v, interval: float, origin: float) -> dict:
    """Retained set and evidence counts for window_extrema."""
    per_sample = window_buckets_loop(t, interval, origin)
    # Only primary buckets exist; borrowing widens membership but never creates a bucket.
    exists = {buckets[0] for buckets in per_sample}
    members: dict[int, list[int]] = {b: [] for b in exists}
    for i, buckets in enumerate(per_sample):
        for b in buckets:
            if b in exists:
                members[b].append(i)
    cols = _columns(v)
    finite = _sample_finite(cols)
    retained: set[int] = set()
    with_finite = 0
    per_bucket: dict[int, list[tuple[int | None, int | None]]] = {}
    for b in sorted(members):
        rows = sorted(members[b])
        per_bucket[b] = []
        if any(finite[i] for i in rows):
            with_finite += 1
        for col in cols:
            hi = None
            lo = None
            for i in rows:
                if not finite[i]:
                    continue
                x = col[i]
                if hi is None or x > col[hi]:
                    hi = i
                if lo is None or x < col[lo]:
                    lo = i
            per_bucket[b].append((hi, lo))
            if hi is not None:
                retained.add(hi)
                retained.add(lo)
    applicable = with_finite > 0
    return {
        "retained": retained,
        "buckets": len(members),
        "buckets_with_finite": with_finite,
        "per_bucket": per_bucket,
        "applicable": applicable,
    }


def _is_nan_scalar(x) -> bool:
    """x is a numpy scalar."""
    py = x.item()
    return isinstance(py, float) and math.isnan(py)


def same_value(a, b) -> bool:
    """Contract equality of two numpy scalars: identical bits, or both NaN."""
    if _is_nan_scalar(a) and _is_nan_scalar(b):
        return True
    return a.tobytes() == b.tobytes()


def _sort_key(x):
    py = x.item()
    if isinstance(py, float):
        if math.isnan(py):
            return (1, 0.0, 0)
        return (0, py, 0 if math.copysign(1.0, py) < 0 else 1)
    return (0, py, 0)


def state_transitions_loop(v) -> dict:
    """Retained set, transition count and sorted distinct values for state_transitions."""
    n = v.shape[0]
    retained: set[int] = set()
    transitions = 0
    distinct: list = []
    for i in range(n):
        x = v[i]
        if i == 0 or i == n - 1:
            retained.add(i)
        if i > 0:
            prev = v[i - 1]
            if not same_value(x, prev):
                retained.add(i)
                transitions += 1
            elif x.tobytes() != prev.tobytes():
                retained.add(i)  # equal NaNs with different payloads: needed for a bitwise hold
        seen = False
        for d in distinct:
            if same_value(x, d):
                seen = True
        if not seen:
            distinct.append(x)
    ordered = sorted(distinct, key=_sort_key)
    return {
        "retained": retained,
        "transitions": transitions,
        "distinct_count": len(distinct),
        "distinct_values": [d.item() for d in ordered],
    }


def json_values_equal(got: list, expected: list) -> bool:
    """Element-wise comparison of evidence value lists: floats bitwise (any NaN equals any NaN), others by type."""
    import struct

    if len(got) != len(expected):
        return False
    for a, b in zip(got, expected):
        if type(a) is not type(b):
            return False
        if isinstance(a, float):
            if math.isnan(a) and math.isnan(b):
                continue
            if struct.pack("<d", a) != struct.pack("<d", b):
                return False
        elif a != b:
            return False
    return True
