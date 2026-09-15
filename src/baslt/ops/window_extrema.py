"""window_extrema: retain the finite max and min of every component in every time bucket.

Contract (docs/contracts.md, "window_extrema"):

    m(i) = floor((t[i] - o) / interval + delta),   delta = 8 * ulp(max(|t[0]|, |t[n-1]|) / interval)

A sample whose `(t[i] - o) / interval` lies within `delta` of an integer boundary also belongs to the neighbouring
bucket. For every bucket containing a finite sample and for every component, the lowest-index max and min over the
bucket's finite samples are retained.

Finiteness follows the contract's convention: a sample is finite when every component is finite
(`_common.finite_mask`).

Existence and membership differ. The buckets of a signal are exactly the primary buckets `m(i)` of its samples; a
borrowed boundary sample widens the membership of an existing bucket but never creates one (in particular it never
creates a bucket below the first sample's bucket). Evidence `buckets` counts the distinct primary buckets and
`buckets_with_finite` those that contain a finite sample, borrowed boundary samples included.

`bucket_ids` is the single implementation of the bucket formula and is meant to be reused. For a sample flagged by
its boundary mask the neighbouring bucket is always `ids - 1`. Let `u = (t - o) / interval` and `K = rint(u)` with
`|u - K| <= delta`. `delta` is a power of two, so `delta < 0.5` means `delta <= 0.25`; then `K` is the unique nearest
integer and the exact sum `u + delta` lies in `[K, K + 0.5]`. With `|u| < 2**52`, `K` and `K + 1` are representable
and rounding is monotone, so `floor(u + delta) == K` and the other bucket touching boundary `K` is `K - 1`.

Outside those limits (`delta >= 0.5` or `|u| >= 2**52`) the neighbouring bucket is not well defined in float64.
`bucket_ids` raises ValueError and `evaluate` reports `not_applicable` with the reason. At epoch timestamps
(`|t|` about 1.7e9 s), `delta < 0.5` requires `|t| / interval < 2**48`, so the interval must exceed about 6.04 us
(1e-5 s gives delta 0.25 and is accepted; 5e-6 s gives delta 0.5 and is rejected). Near `t = 0` the `|u| < 2**52`
limit applies instead.

The status is `not_applicable` for an empty signal, a signal without any finite sample, or a clock outside the
precision limits above; in the last case `buckets` and `buckets_with_finite` are None because they were not
computed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..sampleset import SampleSet
from ._common import STATUS_NOT_APPLICABLE, STATUS_PASS, OpResult, components, finite_mask

if TYPE_CHECKING:
    from ..signals import Signal

_MAX_BUCKET = float(2**52)
_MAX_DELTA = 0.5


def _check_params(interval: float, origin: float) -> tuple[float, float]:
    interval = float(interval)
    origin = float(origin)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError(f"window_extrema interval must be a positive finite duration, got {interval!r}")
    if not np.isfinite(origin):
        raise ValueError(f"window_extrema origin must be finite, got {origin!r}")
    return interval, origin


def bucket_delta(t: np.ndarray, interval: float) -> float:
    """`delta = 8 * ulp(max(|t[0]|, |t[n-1]|) / interval)`; 0.0 for an empty clock, NaN if the ratio overflows."""
    if t.shape[0] == 0:
        return 0.0
    scale = max(abs(float(t[0])), abs(float(t[-1]))) / float(interval)
    with np.errstate(invalid="ignore", over="ignore"):
        return 8.0 * float(np.spacing(np.float64(scale)))


def _bucket_ids_or_reason(t: np.ndarray, interval: float, origin: float) -> tuple[np.ndarray, np.ndarray] | str:
    """`(ids, boundary)` as documented in `bucket_ids`, or a message explaining which precision limit is exceeded."""
    n = t.shape[0]
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, dtype=bool)
    top = max(abs(float(t[0])), abs(float(t[-1])))
    delta = bucket_delta(t, interval)
    if not (delta < _MAX_DELTA):
        return (
            f"window_extrema interval {interval!r} s is too small for timestamps up to {top!r} s: "
            f"delta = 8*ulp({top!r} / {interval!r}) = {delta!r}, which must be below 0.5"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        u = (t - origin) / interval
    au = np.abs(u)
    if not bool((au < _MAX_BUCKET).all()):
        return (
            f"window_extrema bucket numbers reach {float(au.max())!r} with interval {interval!r} s and origin "
            f"{origin!r} s; they must stay below 2**52"
        )
    ids = np.floor(u + delta).astype(np.int64)
    boundary = np.abs(u - np.rint(u)) <= delta
    return ids, boundary


def bucket_ids(t, interval: float, origin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Bucket of every sample and the boundary-neighbour mask.

    Returns `(ids, boundary)`: `ids[i] = floor((t[i] - origin) / interval + delta)` as int64, and `boundary[i]` is
    True when `(t[i] - origin) / interval` lies within `delta` of an integer, in which case sample i also belongs to
    bucket `ids[i] - 1`. `ids` is non-decreasing for a non-decreasing `t`.

    Raises ValueError for a non-positive or non-finite interval, a non-finite origin, `delta >= 0.5`, or bucket
    numbers of magnitude 2**52 or more. The message names the limit that was hit.
    """
    t = np.asarray(t, dtype=np.float64)
    interval, origin = _check_params(interval, origin)
    out = _bucket_ids_or_reason(t, interval, origin)
    if isinstance(out, str):
        raise ValueError(out)
    return out


def _grouped_first(hit: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Position of the first True in every contiguous group; `hit.size` where a group has none."""
    m = hit.shape[0]
    pos = np.where(hit, np.arange(m, dtype=np.int64), np.int64(m))
    return np.minimum.reduceat(pos, starts)


def evaluate(sig: Signal, params: Mapping[str, object], bits: Mapping[str, int]) -> OpResult:
    bit = int(bits["main"])
    interval, origin = _check_params(params["interval"], params.get("origin", 0.0) or 0.0)
    t = sig.t
    v = sig.v
    n = int(t.shape[0])
    evidence: dict = {"buckets": 0, "buckets_with_finite": 0, "interval": interval, "origin": origin}
    if n == 0:
        return OpResult(SampleSet.empty(), evidence, STATUS_NOT_APPLICABLE, ["empty signal"])

    out = _bucket_ids_or_reason(np.asarray(t, dtype=np.float64), interval, origin)
    if isinstance(out, str):
        evidence["buckets"] = None
        evidence["buckets_with_finite"] = None
        return OpResult(SampleSet.empty(), evidence, STATUS_NOT_APPLICABLE, [out])
    ids, boundary = out

    extra = np.flatnonzero(boundary)
    key = np.concatenate([ids, ids[extra] - 1])
    src = np.concatenate([np.arange(n, dtype=np.int64), extra])
    # Stable sort by bucket keeps index order inside a bucket: a bucket's own samples precede the boundary samples
    # it borrows from the next bucket, because ids is non-decreasing in index.
    order = np.argsort(key, kind="stable")
    key = key[order]
    src = src[order]
    m = key.shape[0]
    head = np.empty(m, dtype=bool)
    head[0] = True
    np.not_equal(key[1:], key[:-1], out=head[1:])
    starts = np.flatnonzero(head)
    sizes = np.diff(np.append(starts, m))

    fin = finite_mask(v)[src]  # one mask per sample, shared by every component
    has = _grouped_first(fin, starts) < m
    # Only primary buckets exist; a borrowed boundary sample never creates a bucket of its own.
    primary = np.unique(ids)
    exists = np.isin(key[starts], primary)
    keep = has & exists
    evidence["buckets"] = int(primary.shape[0])
    evidence["buckets_with_finite"] = int(keep.sum())
    if not keep.any():
        return OpResult(SampleSet.empty(), evidence, STATUS_NOT_APPLICABLE, ["no finite sample"])

    all_finite = bool(fin.all())
    picks: list[np.ndarray] = []
    for _, comp in components(v):
        col = comp[src]
        if all_finite:
            lo_fill = hi_fill = col
        else:
            lo_fill = np.where(fin, col, -np.inf)
            hi_fill = np.where(fin, col, np.inf)
        gmax = np.repeat(np.maximum.reduceat(lo_fill, starts), sizes)
        gmin = np.repeat(np.minimum.reduceat(hi_fill, starts), sizes)
        first_max = _grouped_first(fin & (col == gmax), starts)
        first_min = _grouped_first(fin & (col == gmin), starts)
        picks.append(src[first_max[keep]])
        picks.append(src[first_min[keep]])

    samples = SampleSet.from_points(np.concatenate(picks), bit)
    return OpResult(samples=samples, evidence=evidence, status=STATUS_PASS)
