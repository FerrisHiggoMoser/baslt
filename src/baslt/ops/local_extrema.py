"""local_extrema: retain every sufficiently prominent peak (and valley) together with its two bases.

Contract (docs/contracts.md, "local_extrema"), applied to `x` for maxima and to `-x` for minima:

1. Equal-value runs are merged; a run is a candidate peak when both neighbouring runs are strictly lower. The first
   and last run are never peaks, and non-finite samples are +inf walls. The peak index is the middle of its run.
2. The left base is the minimum between the peak and the nearest strictly higher sample (or wall, or the signal
   start), nearest the peak on ties; the right base likewise. Prominence = height - max(left base, right base).
3. Peaks with prominence >= p are kept.
4. Separation: kept peaks are visited by value descending (lowest index first); a peak is dropped when an
   already-kept peak lies closer than `separation` seconds.

Every surviving peak is retained with both bases, so a peak's prominence measured on the retained samples alone is
at least its source prominence.

Technique (the verifier uses a different one). Both searches work on the compressed runs, vectorized over all
candidates, with a block size of 16:

- the nearest strictly higher sample is always a candidate peak or a wall: a higher sample that is not itself a
  candidate would have a candidate between it and the peak that is higher still. So the search runs over the short
  list of candidates and walls: within the own block, then a binary descent over a sparse table of block maxima,
  then within the block found;
- the base is a range-minimum query over the runs between that stopper and the peak: block suffix and prefix
  minima for the two partial blocks plus a sparse table over block minima for the whole blocks in between.

Right-hand searches run the same code on the reversed arrays. Memory stays linear in the number of runs.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..sampleset import SampleSet
from ._common import (
    STATUS_NOT_APPLICABLE,
    STATUS_PASS,
    OpResult,
    capped,
    components,
    finite_mask,
    fnum,
)

if TYPE_CHECKING:
    from ..signals import Signal

BLOCK = 16
KINDS = ("max", "min", "both")


# --------------------------------------------------------------------------------------------------------------
# Range queries


def _sparse_table(values: np.ndarray, better) -> list[np.ndarray]:
    """Level j holds, for each start i, the position of the best value in [i, i + 2**j) (positions into `values`)."""
    table = [np.arange(values.shape[0], dtype=np.int64)]
    span = 1
    while 2 * span <= values.shape[0]:
        prev = table[-1]
        left = prev[: prev.shape[0] - span]
        right = prev[span:]
        table.append(np.where(better(values[right], values[left]), right, left))
        span *= 2
    return table


def _right_wins_min(candidate: np.ndarray, incumbent: np.ndarray) -> np.ndarray:
    return candidate <= incumbent


class _RangeMin:
    """Minimum over runs[a..b] (inclusive), the rightmost position on ties."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        n = values.shape[0]
        blocks = -(-n // BLOCK)
        padded = np.full(blocks * BLOCK, np.inf)
        padded[:n] = values
        grid = padded.reshape(blocks, BLOCK)
        base = (np.arange(blocks, dtype=np.int64) * BLOCK)[:, None]
        offsets = np.arange(BLOCK, dtype=np.int64)

        # Prefix: position of the rightmost minimum of block[0..i].
        running = np.minimum.accumulate(grid, axis=1)
        marks = np.where(grid == running, offsets, -1)
        self.prefix = (np.maximum.accumulate(marks, axis=1) + base).reshape(-1)[:n]

        # Suffix: position of the rightmost minimum of block[i..end]: a leftmost-tie prefix on the reversed block.
        rev = grid[:, ::-1]
        rev_running = np.minimum.accumulate(rev, axis=1)
        before = np.concatenate([np.full((blocks, 1), np.inf), rev_running[:, :-1]], axis=1)
        first = np.where(rev < before, offsets, -1)
        rev_arg = np.maximum.accumulate(first, axis=1)
        # A suffix made only of padding or walls has no finite minimum; such positions are never queried, but
        # keep them inside the array.
        self.suffix = np.minimum(((BLOCK - 1 - rev_arg)[:, ::-1] + base).reshape(-1)[:n], n - 1)

        block_arg = self.prefix[np.minimum(base[:, 0] + BLOCK - 1, n - 1)]
        self.block_arg = block_arg
        self.table = _sparse_table(values[block_arg], _right_wins_min)

    def _pick(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """The better of two positions, `right` winning ties (it lies further right)."""
        return np.where(self.values[right] <= self.values[left], right, left)

    def query(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.empty(a.shape[0], dtype=np.int64)
        ba = a // BLOCK
        bb = b // BLOCK
        same = ba == bb
        if same.any():
            lo, hi = a[same], b[same]
            best = lo.copy()
            for step in range(1, BLOCK):
                pos = lo + step
                ok = pos <= hi
                if not ok.any():
                    break
                pos = np.where(ok, pos, lo)
                best = np.where(ok & (self.values[pos] <= self.values[best]), pos, best)
            out[same] = best
        diff = ~same
        if diff.any():
            lo, hi, bl, bh = a[diff], b[diff], ba[diff], bb[diff]
            best = self._pick(self.suffix[lo], self.prefix[hi])
            first, last = bl + 1, bh - 1
            middle = first <= last
            if middle.any():
                f, l = first[middle], last[middle]
                width = l - f + 1
                level = np.floor(np.log2(width)).astype(np.int64)
                mid = np.empty(f.shape[0], dtype=np.int64)
                for j in np.unique(level):
                    sel = level == j
                    table = self.table[int(j)]
                    left_block = table[f[sel]]
                    right_block = table[l[sel] - (1 << int(j)) + 1]
                    left = self.block_arg[left_block]
                    right = self.block_arg[right_block]
                    mid[sel] = self._pick(left, right)
                # suffix part < middle blocks < prefix part, from left to right
                suffix_part = self.suffix[lo[middle]]
                prefix_part = self.prefix[hi[middle]]
                best_mid = self._pick(self._pick(suffix_part, mid), prefix_part)
                best[middle] = best_mid
            out[diff] = best
        return out


def _previous_greater(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """For each target position e, the largest s < e with values[s] > values[e], or -1."""
    n = values.shape[0]
    heights = values[targets]
    found = np.full(targets.shape[0], -1, dtype=np.int64)
    pending = np.ones(targets.shape[0], dtype=bool)

    # 1. inside the target's own block
    for step in range(1, BLOCK):
        pos = targets - step
        live = pending & (pos >= 0) & ((pos // BLOCK) == (targets // BLOCK))
        if not live.any():
            break
        hit = live & (values[np.maximum(pos, 0)] > heights)
        found[hit] = pos[hit]
        pending &= ~hit

    if not pending.any():
        return found

    # 2. the nearest earlier block whose maximum is higher: binary descent over block maxima
    blocks = -(-n // BLOCK)
    padded = np.full(blocks * BLOCK, -np.inf)
    padded[:n] = values
    block_max = padded.reshape(blocks, BLOCK).max(axis=1)
    levels = [block_max]
    span = 1
    while 2 * span <= blocks:
        prev = levels[-1]
        levels.append(np.maximum(prev[: prev.shape[0] - span], prev[span:]))
        span *= 2

    idx = np.flatnonzero(pending)
    h = heights[idx]
    pos = targets[idx] // BLOCK  # exclusive bound, in blocks
    for j in range(len(levels) - 1, -1, -1):
        width = 1 << j
        table = levels[j]
        can = pos >= width
        start = np.where(can, pos - width, 0)
        clear = can & (table[np.minimum(start, table.shape[0] - 1)] <= h)
        pos = np.where(clear, pos - width, pos)
    has_block = pos > 0
    block = pos - 1

    # 3. inside that block, the last position higher than the target
    result = np.full(idx.shape[0], -1, dtype=np.int64)
    look = np.flatnonzero(has_block)
    if look.size:
        end = np.minimum((block[look] + 1) * BLOCK, n) - 1
        hh = h[look]
        answer = np.full(look.shape[0], -1, dtype=np.int64)
        open_ = np.ones(look.shape[0], dtype=bool)
        for step in range(BLOCK):
            p = end - step
            live = open_ & (p >= block[look] * BLOCK)
            if not live.any():
                break
            hit = live & (values[np.maximum(p, 0)] > hh)
            answer[hit] = p[hit]
            open_ &= ~hit
        result[look] = answer
    found[idx] = result
    return found


# --------------------------------------------------------------------------------------------------------------
# Peaks


def _left_bases(run_value: np.ndarray, entry_runs: np.ndarray, candidate_entries: np.ndarray) -> np.ndarray:
    """Run position of the left base of each candidate (given as positions in the entry list)."""
    stoppers = _previous_greater(run_value[entry_runs], candidate_entries)
    first = np.where(stoppers >= 0, entry_runs[np.maximum(stoppers, 0)] + 1, 0)
    last = entry_runs[candidate_entries] - 1
    return _RangeMin(run_value).query(first, last)


def peaks(t: np.ndarray, y: np.ndarray, prominence: float, separation: float) -> dict[str, np.ndarray]:
    """Peaks of `y` per the contract. Returns arrays index, left_base, right_base, prominence, value (sorted by index)."""
    n = y.shape[0]
    empty = np.empty(0, dtype=np.int64)
    none = {"index": empty, "left_base": empty, "right_base": empty,
            "prominence": np.empty(0), "value": np.empty(0)}
    if n < 3:
        return none
    walled = np.where(np.isfinite(y), y, np.inf)
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(walled[1:], walled[:-1], out=change[1:])
    run_start = np.flatnonzero(change)
    run_end = np.append(run_start[1:], n) - 1
    run_value = walled[run_start]
    runs = run_value.shape[0]
    if runs < 3:
        return none

    middle = run_value[1:-1]
    is_peak = np.zeros(runs, dtype=bool)
    is_peak[1:-1] = np.isfinite(middle) & (run_value[:-2] < middle) & (run_value[2:] < middle)
    peak_runs = np.flatnonzero(is_peak)
    if peak_runs.size == 0:
        return none

    # Entries: candidate peaks and walls, in run order.
    entry_mask = is_peak | np.isinf(run_value)
    entry_runs = np.flatnonzero(entry_mask)
    candidate_entries = np.flatnonzero(is_peak[entry_runs])

    left_run = _left_bases(run_value, entry_runs, candidate_entries)
    rev_value = run_value[::-1].copy()
    rev_entries = (runs - 1 - entry_runs)[::-1].copy()
    rev_candidates = (entry_runs.shape[0] - 1 - candidate_entries)[::-1].copy()
    right_run = (runs - 1 - _left_bases(rev_value, rev_entries, rev_candidates))[::-1]

    height = run_value[peak_runs]
    prom = height - np.maximum(run_value[left_run], run_value[right_run])
    keep = prom >= prominence
    peak_runs, left_run, right_run, prom, height = (
        peak_runs[keep], left_run[keep], right_run[keep], prom[keep], height[keep])
    index = (run_start[peak_runs] + run_end[peak_runs]) // 2

    if separation > 0 and index.size > 1:
        order = np.lexsort((index, -height))
        by_time = np.argsort(t[index], kind="stable")
        times = t[index][by_time].tolist()
        rank_of = np.empty(index.size, dtype=np.int64)
        rank_of[by_time] = np.arange(index.size)
        removed = np.zeros(index.size, dtype=bool)
        kept = np.zeros(index.size, dtype=bool)
        for i in order.tolist():
            rank = int(rank_of[i])
            if removed[rank]:
                continue
            kept[i] = True
            tt = times[rank]
            lo = bisect_right(times, tt - separation)
            hi = bisect_left(times, tt + separation)
            removed[lo:hi] = True
        sel = np.flatnonzero(kept)
        index, left_run, right_run, prom, height = (
            index[sel], left_run[sel], right_run[sel], prom[sel], height[sel])

    return {
        "index": index,
        "left_base": run_end[left_run],
        "right_base": run_start[right_run],
        "prominence": prom,
        "value": height,
    }


def evaluate(sig: Signal, params: Mapping[str, object], bits: Mapping[str, int]) -> OpResult:
    prominence = float(params["prominence"])
    separation = float(params.get("separation", 0.0) or 0.0)
    kind = str(params.get("kind", "both") or "both")
    if kind not in KINDS:
        raise ValueError(f"local_extrema kind must be one of {', '.join(KINDS)}, got {kind!r}")
    if not np.isfinite(prominence) or prominence < 0:
        raise ValueError(f"local_extrema prominence must be a non-negative number, got {prominence!r}")
    if not np.isfinite(separation) or separation < 0:
        raise ValueError(f"local_extrema separation must be a non-negative duration, got {separation!r}")

    peak_bit, base_bit = int(bits["peak"]), int(bits["base"])
    t = sig.t
    fin = finite_mask(sig.v)
    evidence: dict = {"maxima": 0, "minima": 0, "peaks": [], "prominence": prominence, "separation": separation}
    if not fin.any():
        return OpResult(SampleSet.empty(), evidence, STATUS_NOT_APPLICABLE, ["no finite sample"])

    labels = ("max", "min") if kind == "both" else (kind,)
    peak_idx: list[np.ndarray] = []
    base_idx: list[np.ndarray] = []
    records: list[tuple[int, int, str, float]] = []
    for component, column in components(sig.v):
        y = np.asarray(column, dtype=np.float64)
        if sig.v.ndim == 2:
            y = np.where(fin, y, np.nan)  # a sample is finite only when every component is
        for label in labels:
            found = peaks(t, y if label == "max" else -y, prominence, separation)
            peak_idx.append(found["index"])
            base_idx.append(found["left_base"])
            base_idx.append(found["right_base"])
            evidence["maxima" if label == "max" else "minima"] += int(found["index"].shape[0])
            records.extend(
                (int(i), component, label, float(p)) for i, p in zip(found["index"].tolist(), found["prominence"])
            )

    records.sort(key=lambda r: (r[0], r[1], r[2]))
    evidence["peaks"] = capped([
        {"index": i, "t": fnum(t[i]),
         "value": fnum(sig.v[i] if sig.v.ndim == 1 else sig.v[i, component]),
         "prominence": p, "kind": label, "component": component}
        for i, component, label, p in records
    ])
    samples = SampleSet.from_points(np.concatenate(peak_idx), peak_bit).union(
        SampleSet.from_points(np.concatenate(base_idx), base_bit))
    return OpResult(samples=samples, evidence=evidence, status=STATUS_PASS)
