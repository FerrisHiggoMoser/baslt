"""Nested preview ranking for the soft layer.

`preview_order(v, cap)` orders up to `cap` source indices so that every prefix is a sensible preview of the signal:

1. the global minimum and maximum of every component;
2. the boundaries of non-finite runs, so a preview shows its gaps;
3. level by level, the minimum and maximum of every bin of an equal-count split into 2, 4, 8, ... bins, new samples
   only, bins with the widest value range first;
4. if that still leaves room, the remaining samples in bit-reversed index order, which refines evenly.

The order is fixed, so the first k entries for a larger k only ever add samples, which is what lets the budget
search treat artifact size as growing with k. Nothing here is guaranteed: hard requirements carry the
guarantees, the preview only decides how to spend what is left of the budget.

Cost: one reduction over the signal at the finest level needed for `cap`, then work proportional to `cap`. A signal
too flat to fill `cap` from that level is ranked again from the finest level, a second pass over it.
"""

from __future__ import annotations

import numpy as np

from ..ops._common import finite_mask, gap_samples


def _finest_bins(values: np.ndarray, finite: np.ndarray, levels: int) -> tuple[np.ndarray, ...]:
    """Per-bin (min, argmin, max, argmax) of one component at 2**levels equal-count bins."""
    n = values.shape[0]
    bins = 1 << levels
    edges = (np.arange(bins + 1, dtype=np.int64) * n) // bins
    starts = edges[:-1]
    sizes = np.diff(edges)
    low = np.where(finite, values, np.inf)
    high = np.where(finite, values, -np.inf)
    bmin = np.minimum.reduceat(low, starts)
    bmax = np.maximum.reduceat(high, starts)
    index = np.arange(n, dtype=np.int64)
    at_min = np.where(finite & (low == np.repeat(bmin, sizes)), index, n)
    at_max = np.where(finite & (high == np.repeat(bmax, sizes)), index, n)
    return bmin, np.minimum.reduceat(at_min, starts), bmax, np.minimum.reduceat(at_max, starts)


def _coarsen(bmin, amin, bmax, amax):
    """Merge neighbouring bins pairwise; the left bin wins ties, so the lowest index stays the argmin/argmax."""
    lmin, rmin = bmin[0::2], bmin[1::2]
    lmax, rmax = bmax[0::2], bmax[1::2]
    take_right_min = rmin < lmin
    take_right_max = rmax > lmax
    return (
        np.where(take_right_min, rmin, lmin),
        np.where(take_right_min, amin[1::2], amin[0::2]),
        np.where(take_right_max, rmax, lmax),
        np.where(take_right_max, amax[1::2], amax[0::2]),
    )


def _bit_reversed(n: int) -> np.ndarray:
    """0..n-1 in the order of their bit-reversed values: an even refinement of the index range."""
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    width = int(n - 1).bit_length()
    index = np.arange(n, dtype=np.int64)
    reversed_ = np.zeros(n, dtype=np.int64)
    for bit in range(width):
        reversed_ |= ((index >> bit) & 1) << (width - 1 - bit)
    return index[np.argsort(reversed_, kind="stable")]


def _pyramid_order(v: np.ndarray, columns: list[np.ndarray], finite: np.ndarray, levels: int, cap: int,
                   fill: bool) -> np.ndarray:
    """Steps 1-3 over levels 0..`levels`, stopping once `cap` samples are ranked; step 4 only when `fill`."""
    n = int(v.shape[0])
    pyramids = []
    for column in columns:
        level = [_finest_bins(column, finite, levels)]
        for _ in range(levels):
            level.append(_coarsen(*level[-1]))
        pyramids.append(level[::-1])  # coarsest first

    taken = np.zeros(n + 1, dtype=bool)  # slot n marks "no finite sample in this bin"
    taken[n] = True
    order: list[np.ndarray] = []
    count = 0

    def add(candidates: np.ndarray) -> None:
        nonlocal count
        fresh = candidates[~taken[candidates]]
        if fresh.size == 0:
            return
        _, first = np.unique(fresh, return_index=True)
        fresh = fresh[np.sort(first)]
        taken[fresh] = True
        order.append(fresh)
        count += int(fresh.size)

    for depth in range(levels + 1):
        spans = []
        picks = []
        for pyramid in pyramids:
            bmin, amin, bmax, amax = pyramid[depth]
            with np.errstate(invalid="ignore"):
                spans.append(np.where(np.isfinite(bmax - bmin), bmax - bmin, -1.0))
            picks.append(np.stack([amin, amax], axis=1))
        span = np.max(np.stack(spans, axis=0), axis=0)
        by_range = np.lexsort((np.arange(span.shape[0]), -span))
        stacked = np.concatenate([p[by_range] for p in picks], axis=1).reshape(-1)
        add(stacked)
        if depth == 0:
            add(gap_samples(v, 0).materialize()[0])
        if count >= cap:
            break

    if fill and count < cap:
        add(_bit_reversed(n))
    return np.concatenate(order) if order else np.empty(0, dtype=np.int64)


def preview_order(v: np.ndarray, cap: int) -> np.ndarray:
    """Up to `cap` distinct source indices of `v` ((n,) or (n, k)), most useful first.

    The result is the first `cap` entries of one fixed order of all n indices, whatever `cap` is.
    """
    n = int(v.shape[0])
    cap = min(int(cap), n)
    if cap <= 0:
        return np.empty(0, dtype=np.int64)
    finite = finite_mask(v)
    columns = [np.asarray(v, dtype=np.float64)] if v.ndim == 1 else [
        np.asarray(v[:, k], dtype=np.float64) for k in range(v.shape[1])
    ]

    # 2**levels bins give up to 2 * 2**levels picks per component. Equal-count bins nest, so a coarser pyramid
    # is a prefix of a finer one; only when the pyramid runs out (flat stretches pick one sample per bin) is the
    # finest possible level needed before the bit-reversed fill, or a small cap would not be a prefix of a large one.
    finest = int(n).bit_length() - 1
    wanted = max(1, (cap + 1) // 2)
    levels = min(int(wanted - 1).bit_length(), finest)
    result = _pyramid_order(v, columns, finite, levels, cap, fill=levels == finest)
    if result.shape[0] < cap:
        result = _pyramid_order(v, columns, finite, finest, cap, fill=True)
    return result[:cap]
