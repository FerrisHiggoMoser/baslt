"""Reference detectors for the verifier, written directly from docs/contracts.md.

These functions recompute, from plain arrays, the facts that the fidelity contracts protect: global and window
extrema, local peaks, threshold crossings, violation runs, state transitions, event triggers and windows, and
trajectory SED. They are the verifier's own implementation and import nothing from the compiler.

Style: numpy is used for element-wise arithmetic and masks and to find candidate positions (level crossings,
state flips, run boundaries, peak runs). The contract's rules are then applied to those candidates in the order
the contract text describes them: vectorized where a rule is element-wise, and as a single left-to-right sweep
where it is sequential (debounce, peak bases). No rule is applied by re-scanning the signal once per candidate,
so cost stays linear in the number of samples even for the shapes that make candidates as dense as samples -- a
monotone trend with dither, a quantized signal toggling between two codes, noise sitting on the threshold. What
is left of the per-candidate cost is one dict per *reported* record.

Conventions (docs/contracts.md): timestamps are float seconds, non-decreasing and finite; a sample is finite when
every component is finite; `ulp(x)` is the gap between `|x|` and the next larger float64.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

__all__ = [
    "bucket_delta",
    "crossing_time",
    "crossings",
    "event_triggers",
    "event_window_clipped",
    "event_window_indices",
    "finite",
    "global_extrema",
    "local_peaks",
    "prominences_at",
    "reconstruct_hold",
    "reconstruct_linear",
    "sed_max",
    "state_transitions",
    "ulp",
    "violation_runs",
    "window_buckets",
    "window_evidence",
    "window_extrema",
]

EDGES = ("rising", "falling", "both")
INTERPOLATIONS = ("linear", "none")
PEAK_KINDS = ("max", "min", "both")
OCCURRENCES = ("first", "last", "all")


# ---------------------------------------------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------------------------------------------


def ulp(value: float) -> float:
    """The gap between `|value|` and the next larger float64."""
    magnitude = abs(float(value))
    return math.nextafter(magnitude, math.inf) - magnitude


def finite(x) -> np.ndarray:
    """Boolean mask of finite samples; a 2-D sample is finite when every component is finite."""
    ok = np.isfinite(np.asarray(x))
    if ok.ndim == 2:
        ok = ok.all(axis=1)
    return ok


def crossing_time(t_before: float, t_after: float, x_before: float, x_after: float, level: float) -> float:
    """`tc = t[i] + (t[i+1] - t[i]) * (V - x[i]) / (x[i+1] - x[i])`, the linear crossing time of `level`, clamped
    to `[t[i], t[i+1]]` as the contract says (rounding can otherwise land one ulp outside the bracket).

    Callers only use it on a pair that straddles `level`, so `x_after != x_before`.
    """
    t0, t1, x0, x1, v = float(t_before), float(t_after), float(x_before), float(x_after), float(level)
    return min(max(t0 + (t1 - t0) * (v - x0) / (x1 - x0), t0), t1)


def _crossing_times(t_before, t_after, x_before, x_after, level: float) -> np.ndarray:
    """`crossing_time` for whole arrays of straddling pairs, in the same operation order (so, bit-identical).

    Only called with pairs that straddle `level`, where `x_after != x_before`; no lane divides by zero.
    """
    tc = t_before + (t_after - t_before) * (float(level) - x_before) / (x_after - x_before)
    return np.minimum(np.maximum(tc, t_before), t_after)


# ---------------------------------------------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------------------------------------------


def reconstruct_linear(t_ret, x_ret, t_query) -> np.ndarray:
    """Linear reconstruction from retained samples, evaluated at `t_query`.

    - At a retained timestamp the value is that retained sample, bit-identical. When several retained samples share
      the timestamp, the later one is used.
    - Strictly between two consecutive retained timestamps `ta < tq < tb` the value is `x[a] + w * (x[b] - x[a])`
      with `w = (tq - ta) / (tb - ta)` -- the weight-first form of the trajectories contract, which differs in the
      last ulp from `x[a] + (x[b] - x[a]) * (tq - ta) / (tb - ta)`; the denominator is positive, so repeated
      timestamps never divide by zero.
    - The interval between a finite and a non-finite retained sample is a gap: the value is NaN.
    - Before the first or after the last retained timestamp there is no reconstruction: the value is NaN.

    `x_ret` may be 1-D or 2-D `(m, k)`; the result has shape `t_query.shape + x_ret.shape[1:]`, so a 2-D query
    keeps its shape and a scalar query of a scalar signal returns a 0-d array.
    """
    tr = np.asarray(t_ret, dtype=np.float64)
    xr = np.asarray(x_ret, dtype=np.float64)
    query = np.asarray(t_query, dtype=np.float64)
    shape = query.shape + xr.shape[1:]
    tq = query.reshape(-1)
    out = np.full(tq.shape + xr.shape[1:], np.nan)
    m = tr.shape[0]
    if m == 0 or tq.size == 0:
        return out.reshape(shape)

    # k = the last retained sample with t_ret[k] <= query (the later one when timestamps repeat), or -1.
    k = np.searchsorted(tr, tq, side="right") - 1
    has_left = k >= 0
    at_knot = has_left & (tr[np.maximum(k, 0)] == tq)
    out[at_knot] = xr[k[at_knot]]

    between = np.flatnonzero(has_left & ~at_knot & (k < m - 1))
    a = k[between]
    b = a + 1
    not_gap = finite(xr[a]) & finite(xr[b])
    between, a, b = between[not_gap], a[not_gap], b[not_gap]
    w = (tq[between] - tr[a]) / (tr[b] - tr[a])
    if xr.ndim == 2:
        w = w[:, None]
    out[between] = xr[a] + w * (xr[b] - xr[a])
    return out.reshape(shape)


def reconstruct_hold(t_ret, x_ret, t_query) -> np.ndarray:
    """Sample-and-hold reconstruction: the previous retained value (the later one when timestamps repeat).

    Queries before the first retained timestamp have no held value. They are NaN; a non-float result is promoted to
    float64 only when such a query exists, so an integer state signal keeps its dtype in the normal case.

    As in `reconstruct_linear`, the result has shape `t_query.shape + x_ret.shape[1:]`.
    """
    tr = np.asarray(t_ret, dtype=np.float64)
    xr = np.asarray(x_ret)
    query = np.asarray(t_query, dtype=np.float64)
    shape = query.shape + xr.shape[1:]
    tq = query.reshape(-1)
    if tr.shape[0] == 0:
        return np.full(shape, np.nan)
    k = np.searchsorted(tr, tq, side="right") - 1
    held = xr[np.maximum(k, 0)]
    before_first = k < 0
    if before_first.any():
        if held.dtype.kind != "f":
            held = held.astype(np.float64)
        held[before_first] = np.nan
    return held.reshape(shape)


# ---------------------------------------------------------------------------------------------------------------
# global_extrema and window_extrema
# ---------------------------------------------------------------------------------------------------------------


def global_extrema(x) -> dict[str, int | None] | list[dict[str, int | None]]:
    """Index of the maximum and of the minimum finite value, lowest index on ties.

    A 1-D signal gives one record `{max_index, min_index}`. An `(n, k)` signal gives one record per component,
    because the contract states the guarantee per component. Eligibility is always the Conventions rule -- a
    sample is finite when *every* component is finite -- so a row that is non-finite in another component is
    never a component's extremum. Slicing a vector signal and calling this per column would instead use
    per-column finiteness, which is why the vector form exists.

    Both indices are None when there is no finite sample (not applicable).
    """
    values = np.asarray(x)
    ok = finite(values)
    if values.ndim == 1:
        return _extremum_record(values, ok)
    return [_extremum_record(values[:, c], ok) for c in range(values.shape[1])]


def _extremum_record(column: np.ndarray, ok: np.ndarray) -> dict[str, int | None]:
    if not ok.any():
        return {"max_index": None, "min_index": None}
    eligible = column[ok]
    largest = eligible.max()
    smallest = eligible.min()
    return {
        "max_index": int(np.flatnonzero(ok & (column == largest))[0]),
        "min_index": int(np.flatnonzero(ok & (column == smallest))[0]),
    }


def bucket_delta(t, interval: float) -> float:
    """`δ = 8 · ulp(max(|t[0]|, |t[n-1]|) / Δ)`; 0 for an empty signal, which has no bucket to widen."""
    times = np.asarray(t, dtype=np.float64)
    if times.shape[0] == 0:
        return 0.0
    return 8.0 * ulp(max(abs(float(times[0])), abs(float(times[-1]))) / float(interval))


def window_buckets(t, interval: float, origin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Bucket of every sample and whether it also belongs to the neighbouring bucket.

    Primary bucket: `m(i) = floor((t[i] - o) / Δ + δ)`. A sample whose `u = (t[i] - o) / Δ` lies within `δ` of an
    integer boundary `r` (`|u - r| <= δ`) also belongs to the neighbouring bucket. Such a sample always has primary
    bucket `r` (the `+ δ` lifts `u` to at least `r`), so its neighbouring bucket is `m(i) - 1`.

    Returns `(primary, near_boundary)`: int64 bucket ids and a boolean mask.
    """
    times = np.asarray(t, dtype=np.float64)
    if interval <= 0:
        raise ValueError("interval must be positive")
    if times.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    delta = bucket_delta(times, interval)
    u = (times - float(origin)) / float(interval)
    primary = np.floor(u + delta).astype(np.int64)
    near_boundary = np.abs(u - np.rint(u)) <= delta
    return primary, near_boundary


def window_extrema(t, x, interval: float, origin: float = 0.0):
    """For every bucket of the signal containing a finite sample: `(max_index, min_index)`, lowest index on ties.

    A 1-D signal gives one `{bucket: (max_index, min_index)}` map; an `(n, k)` signal gives one map per component,
    because the contract states the guarantee per component. Eligibility is the Conventions rule -- a sample is
    finite when *every* component is finite -- so every component's map draws on the same eligible rows.

    Two readings of the contract meet here, and are resolved like this:

    - *Which buckets exist.* `m(i) = floor((t[i] - o) / Δ + δ)` gives each sample exactly one bucket, so the
      buckets of a signal are the primary buckets of its samples. A bucket below the first sample's bucket is not
      one of them, even though the first sample sits on its boundary and would otherwise be borrowed into it.
    - *Which samples a bucket contains.* Wider: a boundary sample belongs to its own bucket and to the one before
      it (`window_buckets`) and may be that bucket's extremum. A bucket whose own samples are all non-finite is
      therefore still reported when a borrowed boundary sample is finite: it is a bucket, and it contains a finite
      sample.
    """
    values = np.asarray(x)
    primary, near_boundary = window_buckets(t, interval, origin)
    ok = finite(values)

    # Every (bucket, sample) membership pair: each finite sample in its own bucket, and each finite boundary sample
    # also in the bucket before it when that bucket is one of the signal's own.
    own = np.flatnonzero(ok)
    borrowed = np.flatnonzero(ok & near_boundary)
    if borrowed.size:
        borrowed = borrowed[np.isin(primary[borrowed] - 1, primary)]
    member = np.concatenate((own, borrowed))
    bucket = np.concatenate((primary[own], primary[borrowed] - 1))
    if values.ndim == 1:
        return _bucket_extrema(values, member, bucket)
    return [_bucket_extrema(values[:, c], member, bucket) for c in range(values.shape[1])]


def _bucket_extrema(column: np.ndarray, member: np.ndarray, bucket: np.ndarray) -> dict[int, tuple[int, int]]:
    if member.size == 0:
        return {}
    value = column[member]

    # Sort memberships by bucket, then value, then index. The first entry of a bucket is its minimum with the lowest
    # index. Sorting the index descending instead makes the last entry of a bucket its maximum with the lowest index.
    by_min = np.lexsort((member, value, bucket))
    by_max = np.lexsort((-member, value, bucket))
    sorted_bucket = bucket[by_min]
    first = np.flatnonzero(np.concatenate(([True], sorted_bucket[1:] != sorted_bucket[:-1])))
    last = np.concatenate((first[1:], [sorted_bucket.size])) - 1

    ids = sorted_bucket[first].tolist()
    min_index = member[by_min][first].tolist()
    max_index = member[by_max][last].tolist()
    return {b: (hi, lo) for b, hi, lo in zip(ids, max_index, min_index)}


def window_evidence(t, x, interval: float, origin: float = 0.0) -> dict[str, int]:
    """The contract's evidence for window_extrema: `{buckets, buckets_with_finite}`.

    `buckets` counts the distinct primary buckets of the signal's samples -- buckets that no sample falls in, for
    example inside a long time gap, are not counted. `buckets_with_finite` counts those that contain a finite
    sample, borrowed boundary samples included, and is the same for every component because eligibility is the
    all-components rule. Returning both turns a disagreement with the planner's bucketing into two numbers to
    compare instead of a retention failure found later.
    """
    primary, _ = window_buckets(t, interval, origin)
    per_component = window_extrema(t, x, interval, origin)
    if isinstance(per_component, list):
        per_component = per_component[0] if per_component else {}
    return {"buckets": int(np.unique(primary).size), "buckets_with_finite": len(per_component)}


# ---------------------------------------------------------------------------------------------------------------
# local_extrema
# ---------------------------------------------------------------------------------------------------------------


def local_peaks(t, x, prominence: float = 0.0, separation: float = 0.0, kind: str = "max") -> list[dict]:
    """Peaks surviving steps 1-4 of the local_extrema contract.

    Returns `{index, prominence, left_base, right_base, kind}` records sorted by index. Minima are the peaks of `-x`
    (their prominence is measured on `-x`, so it is positive); `kind="both"` runs maxima and minima independently.
    """
    if kind not in PEAK_KINDS:
        raise ValueError(f"kind must be one of {PEAK_KINDS}")
    times = np.asarray(t, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    found: list[dict] = []
    if kind in ("max", "both"):
        found += _peaks_of(times, values, float(prominence), float(separation), "max")
    if kind in ("min", "both"):
        found += _peaks_of(times, -values, float(prominence), float(separation), "min")
    found.sort(key=lambda peak: peak["index"])
    return found


def _peaks_of(t: np.ndarray, y: np.ndarray, prominence: float, separation: float, label: str) -> list[dict]:
    n = y.shape[0]
    if n < 3:
        return []
    # Non-finite samples act like +inf walls.
    walled = np.where(np.isfinite(y), y, np.inf)

    # Step 1. Merge equal-value runs. A run is a candidate when both neighbouring runs are strictly lower; the first
    # and last run are never peaks, and a run of walls is never a peak. The peak index is the middle of the run,
    # rounded down.
    run_start = np.flatnonzero(np.concatenate(([True], walled[1:] != walled[:-1])))
    run_end = np.concatenate((run_start[1:], [n])) - 1
    run_value = walled[run_start]
    middle = run_value[1:-1]
    is_candidate = np.isfinite(middle) & (run_value[:-2] < middle) & (run_value[2:] < middle)
    candidate_runs = np.flatnonzero(is_candidate) + 1
    peak_indices = (run_start[candidate_runs] + run_end[candidate_runs]) // 2

    if candidate_runs.size == 0:
        return []

    # Step 2. The base interval of a peak reaches to the nearest strictly higher sample (a wall counts as higher)
    # or to the signal end, and the base is that interval's minimum, nearest the peak on ties. One sweep per
    # direction over the *runs* answers that for every candidate at once (`_base_runs`); scanning the interval per
    # peak instead is quadratic whenever peaks are not dominated by a nearby higher sample, which is the normal
    # case for a dithering or quantized signal. Inside the base run, the sample nearest the peak is the run's last
    # sample on the left and its first sample on the right.
    run_count = run_value.shape[0]
    left_base_run = _base_runs(run_value, candidate_runs)
    mirrored = _base_runs(run_value[::-1], run_count - 1 - candidate_runs[::-1])
    right_base_run = (run_count - 1 - mirrored)[::-1]
    left_base = run_end[left_base_run]
    right_base = run_start[right_base_run]
    height = run_value[candidate_runs]
    prominences = height - np.maximum(run_value[left_base_run], run_value[right_base_run])

    # Step 3. Keep peaks with prominence >= p.
    kept = np.flatnonzero(prominences >= prominence)

    # Step 4. Visit kept peaks by value descending (lowest index first on ties); keep a peak when no already-kept
    # peak is closer than the separation. The closest kept peaks in time are the neighbours in a sorted time list.
    if separation > 0 and kept.size:
        peak_times = t[peak_indices[kept]]
        kept_times: list[float] = []
        separated: list[int] = []
        for position in np.lexsort((peak_indices[kept], -height[kept])).tolist():
            tp = float(peak_times[position])
            pos = bisect.bisect_left(kept_times, tp)
            too_close_after = pos < len(kept_times) and kept_times[pos] - tp < separation
            too_close_before = pos > 0 and tp - kept_times[pos - 1] < separation
            if not (too_close_after or too_close_before):
                kept_times.insert(pos, tp)
                separated.append(position)
        separated.sort()
        kept = kept[separated]

    return [
        {
            "index": index,
            "prominence": value,
            "left_base": left,
            "right_base": right,
            "kind": label,
        }
        for index, value, left, right in zip(
            peak_indices[kept].tolist(),
            prominences[kept].tolist(),
            left_base[kept].tolist(),
            right_base[kept].tolist(),
        )
    ]


def prominences_at(x, positions) -> np.ndarray:
    """Prominence of `x` treated as a peak at each given position; NaN where that position is not a peak.

    Used on retained samples: a claimed peak need not be the middle of its run there (retained neighbours may
    merge into a plateau), so only run membership matters. The bases follow step 2 of the contract.
    """
    values = np.asarray(x, dtype=np.float64)
    wanted = np.asarray(positions, dtype=np.int64)
    out = np.full(wanted.shape[0], np.nan)
    n = values.shape[0]
    if n < 3 or wanted.size == 0:
        return out
    walled = np.where(np.isfinite(values), values, np.inf)
    run_start = np.flatnonzero(np.concatenate(([True], walled[1:] != walled[:-1])))
    run_value = walled[run_start]
    runs = run_value.shape[0]
    run_of = np.searchsorted(run_start, wanted, side="right") - 1
    interior = (run_of > 0) & (run_of < runs - 1)
    safe = np.clip(run_of, 1, max(runs - 2, 1))
    is_peak = (interior & np.isfinite(run_value[safe])
               & (run_value[safe - 1] < run_value[safe]) & (run_value[np.minimum(safe + 1, runs - 1)] < run_value[safe]))
    if runs < 3 or not is_peak.any():
        return out
    candidates = np.unique(run_of[is_peak])
    left = _base_runs(run_value, candidates)
    mirrored = _base_runs(run_value[::-1], runs - 1 - candidates[::-1])
    right = (runs - 1 - mirrored)[::-1]
    prominence = run_value[candidates] - np.maximum(run_value[left], run_value[right])
    lookup = dict(zip(candidates.tolist(), prominence.tolist()))
    out[is_peak] = [lookup[r] for r in run_of[is_peak].tolist()]
    return out


def _base_runs(run_value: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """For each candidate run, the run holding its left base: the minimum of the runs between it and the nearest
    strictly higher run on its left, taking the run nearest the candidate on ties.

    One left-to-right sweep. The stack holds the runs that are still higher than every run after them, each
    carrying the minimum of the runs it owns -- those from the entry below it up to itself. Reaching a candidate
    pops exactly the runs of its base interval, already reduced to one minimum per popped entry, so every run is
    pushed and popped once and the whole sweep is linear. Right bases are this same sweep over the reversed runs.

    `candidates` must be increasing. A wall (a `+inf` run) is never popped by a finite candidate, so it stops the
    interval exactly as the contract's wall does, and the base is always a finite run: the run just before a
    candidate is lower than it, so it always lies in the interval.
    """
    values = run_value.tolist()
    is_candidate = np.zeros(len(values), dtype=bool)
    is_candidate[candidates] = True
    wanted = is_candidate.tolist()
    bases: list[int] = []
    stack: list[tuple[float, float, int]] = []
    for run, height in enumerate(values):
        lowest, at = math.inf, -1
        while stack and stack[-1][0] <= height:
            _, owned_low, owned_at = stack.pop()
            if owned_low < lowest:  # popped right to left, so `<` keeps the run nearest the candidate
                lowest, at = owned_low, owned_at
        if wanted[run]:
            bases.append(at)
        if height <= lowest:  # this run is the rightmost minimum of the region it now owns
            lowest, at = height, run
        stack.append((height, lowest, at))
    return np.array(bases, dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------
# threshold_crossing
# ---------------------------------------------------------------------------------------------------------------


def crossings(
    t,
    x,
    value: float,
    edge: str = "both",
    hysteresis: float = 0.0,
    debounce: float = 0.0,
    interpolate: str = "linear",
) -> dict:
    """Steps 1-5 of the threshold_crossing contract.

    Returns a dict:

    - `accepted`: records `{t, edge, index_before, index_after, confirm_index, gap}` after debounce and the edge
      filter. `index_before`/`index_after` are the bracketing finite samples of the level crossing (adjacent unless
      the crossing spans a non-finite gap), `confirm_index` is the sample where the state changed.
    - `pending_at_end`: the final run (after the last flip, before the edge filter) is shorter than the debounce.
    - `raw_level_crossings`: number of level crossings (step 1), `flips`: number of state changes (step 3),
      `gap_flips`: flips whose crossing spans a non-finite gap.
    - `pending_at_end_edge`, `gap_flips_accepted`: the same two caveats counted only where they concern the
      requested edge -- a pending final run started by a flip the edge filter keeps, and gap flips among the
      accepted crossings. The contract defines `pending_at_end` in step 4 and the edge filter in step 5, so the
      first pair is what the machine saw; but the evidence list is per requirement, and a `falling` requirement
      that never claimed a rising flip should not inherit its caveat. Both pairs are returned so the caller
      raising WARN can choose, rather than this module choosing for it.
    """
    if edge not in EDGES:
        raise ValueError(f"edge must be one of {EDGES}")
    if interpolate not in INTERPOLATIONS:
        raise ValueError(f"interpolate must be one of {INTERPOLATIONS}")
    times = np.asarray(t, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    level = float(value)
    hyst = float(hysteresis)
    min_run = float(debounce)
    result: dict = {
        "accepted": [],
        "pending_at_end": False,
        "pending_at_end_edge": False,
        "raw_level_crossings": 0,
        "flips": 0,
        "gap_flips": 0,
        "gap_flips_accepted": 0,
    }

    # Work on the finite samples only: position p in `xf` is source sample `finite_index[p]`.
    finite_index = np.flatnonzero(np.isfinite(values))
    if finite_index.size == 0:
        return result
    xf = values[finite_index]

    # Step 1. Level crossings between consecutive finite samples (pair p joins positions p and p + 1):
    # rising if x[i] < V <= x[i+1], falling if x[i] >= V > x[i+1].
    before, after = xf[:-1], xf[1:]
    rising_pairs = np.flatnonzero((before < level) & (level <= after))
    falling_pairs = np.flatnonzero((before >= level) & (level > after))
    result["raw_level_crossings"] = int(rising_pairs.size + falling_pairs.size)

    # Step 2. State. With H = 0 every sample asserts its own state (x >= V). With H > 0 a sample asserts high at
    # x >= V + H/2, low at x <= V - H/2, and otherwise holds. Non-finite samples hold (they are not in `xf`).
    # The initial state is x[first finite] >= V.
    if hyst > 0:
        asserts_high = xf >= level + hyst / 2
        asserts_low = xf <= level - hyst / 2
    else:
        asserts_high = xf >= level
        asserts_low = ~asserts_high
    initial_high = bool(xf[0] >= level)
    asserting = np.flatnonzero(asserts_high | asserts_low)
    asserted_high = asserts_high[asserting]
    state_before = np.concatenate(([initial_high], asserted_high[:-1]))
    changes = asserted_high != state_before
    flip_positions = asserting[changes]
    flip_to_high = asserted_high[changes]

    # Step 3. A flip at position j takes the last level crossing in its direction with i + 1 <= j.
    # searchsorted(..., side="right") - 1 finds exactly that crossing among the sorted pair ends p + 1.
    flip_count = int(flip_positions.size)
    if flip_count == 0:
        return result
    last_rising = np.searchsorted(rising_pairs + 1, flip_positions, side="right") - 1
    last_falling = np.searchsorted(falling_pairs + 1, flip_positions, side="right") - 1
    pair = np.empty(flip_count, dtype=np.int64)
    for rising, direction_pairs, last in ((True, rising_pairs, last_rising), (False, falling_pairs, last_falling)):
        lane = np.flatnonzero(flip_to_high if rising else ~flip_to_high)
        if lane.size == 0:
            continue
        # A low state was asserted by a sample below V and a high state by one at or above V, so a crossing in the
        # flip's direction always exists between the previous assertion and this one.
        if (last[lane] < 0).any():
            raise AssertionError("state flip without a level crossing in its direction")
        pair[lane] = direction_pairs[last[lane]]

    index_before = finite_index[pair]
    index_after = finite_index[pair + 1]
    if interpolate == "linear":
        tc = _crossing_times(
            times[index_before], times[index_after], values[index_before], values[index_after], level
        )
    else:
        tc = times[index_after]
    gap = (index_after - index_before) > 1
    result["flips"] = flip_count
    result["gap_flips"] = int(np.count_nonzero(gap))

    # Step 4. Debounce. Order flips by time (stable, so equal times keep sample order). The run started by flip k
    # lasts until flip k + 1, the last run until the last sample time. A flip is accepted when its run lasts >= D.
    # Accepted transitions are the changes of the accepted state, and that state is always the direction of the
    # last accepted flip -- so an accepted flip is a transition exactly when it differs from the one before it.
    order = np.argsort(tc, kind="stable")
    in_time_order = tc[order]
    run_end = np.empty(flip_count, dtype=np.float64)
    run_end[:-1] = in_time_order[1:]
    run_end[-1] = float(times[-1])
    debounced = np.ones(flip_count, dtype=bool) if min_run <= 0 else (run_end - in_time_order) >= min_run
    result["pending_at_end"] = not bool(debounced[-1])
    passed = order[debounced]
    passed_high = flip_to_high[passed]
    is_transition = np.empty(passed.size, dtype=bool)
    if passed.size:
        is_transition[0] = passed_high[0] != initial_high
        is_transition[1:] = passed_high[1:] != passed_high[:-1]
    transitions = passed[is_transition]

    # Step 5. Edge filter. Only the crossings that survive it become records.
    if edge == "rising":
        selected = transitions[flip_to_high[transitions]]
    elif edge == "falling":
        selected = transitions[~flip_to_high[transitions]]
    else:
        selected = transitions
    result["accepted"] = [
        {
            "t": when,
            "edge": "rising" if rising else "falling",
            "index_before": i,
            "index_after": i_next,
            "confirm_index": confirm,
            "gap": spans_gap,
        }
        for when, rising, i, i_next, confirm, spans_gap in zip(
            tc[selected].tolist(),
            flip_to_high[selected].tolist(),
            index_before[selected].tolist(),
            index_after[selected].tolist(),
            finite_index[flip_positions[selected]].tolist(),
            gap[selected].tolist(),
        )
    ]

    # The two caveats of step 4, restricted to the requested edge (see the docstring).
    result["gap_flips_accepted"] = sum(1 for flip in result["accepted"] if flip["gap"])
    result["pending_at_end_edge"] = result["pending_at_end"] and (
        edge == "both" or (edge == "rising") == bool(flip_to_high[order[-1]])
    )
    return result


# ---------------------------------------------------------------------------------------------------------------
# violation
# ---------------------------------------------------------------------------------------------------------------


def violation_runs(
    t, x, above: float | None = None, below: float | None = None, min_duration: float = 0.0
) -> list[dict]:
    """Violating runs of the violation contract (give exactly one of `above` and `below`).

    Returns records `{start, end, worst_index, open_start, open_end, gap_start, gap_end, first_index, last_index}`
    for runs with `end - start >= min_duration`. The worst sample is the largest (above) or smallest (below) value
    of the run, lowest index on ties.
    """
    if (above is None) == (below is None):
        raise ValueError("give exactly one of above or below")
    times = np.asarray(t, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    n = values.shape[0]
    ok = np.isfinite(values)
    if above is not None:
        limit = float(above)
        violating = ok & (values > limit)
    else:
        limit = float(below)
        violating = ok & (values < limit)

    # A maximal run of consecutive violating samples starts where the mask switches on and ends where it switches
    # off; non-finite samples are not violating, so they end a run.
    padded = np.concatenate(([False], violating, [False]))
    switches = np.flatnonzero(padded[1:] != padded[:-1])
    first_indices = switches[0::2]
    last_indices = switches[1::2] - 1
    if first_indices.size == 0:
        return []

    # Boundary times: the crossing of the limit against the neighbouring sample, except where the run opens the
    # signal or sits against a gap, which take the run's own edge sample. Only the interpolating runs are computed,
    # so no lane of the division is a run that has no neighbour to interpolate against.
    open_start = first_indices == 0
    gap_start = ~open_start & ~ok[np.maximum(first_indices - 1, 0)]
    start = times[first_indices]
    inner = first_indices[~(open_start | gap_start)]
    start[~(open_start | gap_start)] = _crossing_times(
        times[inner - 1], times[inner], values[inner - 1], values[inner], limit
    )

    open_end = last_indices == n - 1
    gap_end = ~open_end & ~ok[np.minimum(last_indices + 1, n - 1)]
    end = times[last_indices]
    inner = last_indices[~(open_end | gap_end)]
    end[~(open_end | gap_end)] = _crossing_times(
        times[inner], times[inner + 1], values[inner], values[inner + 1], limit
    )

    kept = np.flatnonzero((end - start) >= min_duration)
    if kept.size == 0:
        return []

    # The most extreme sample of each kept run, lowest index on ties. The kept runs' samples are laid end to end
    # and reduced per run, so the cost is one pass over the violating samples rather than an argmax call per run.
    first_kept, last_kept = first_indices[kept], last_indices[kept]
    lengths = last_kept - first_kept + 1
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    member = np.arange(int(lengths.sum())) - np.repeat(offsets, lengths) + np.repeat(first_kept, lengths)
    inside = values[member]
    reduction = np.maximum if above is not None else np.minimum
    extreme = np.repeat(reduction.reduceat(inside, offsets), lengths)
    at_extreme = np.where(inside == extreme, np.arange(member.size), member.size)
    worst = member[np.minimum.reduceat(at_extreme, offsets)]

    return [
        {
            "start": begins,
            "end": ends,
            "worst_index": extreme_index,
            "open_start": opens,
            "open_end": closes,
            "gap_start": after_gap,
            "gap_end": before_gap,
            "first_index": first,
            "last_index": last,
        }
        for begins, ends, extreme_index, opens, closes, after_gap, before_gap, first, last in zip(
            start[kept].tolist(),
            end[kept].tolist(),
            worst.tolist(),
            open_start[kept].tolist(),
            open_end[kept].tolist(),
            gap_start[kept].tolist(),
            gap_end[kept].tolist(),
            first_kept.tolist(),
            last_kept.tolist(),
        )
    ]


# ---------------------------------------------------------------------------------------------------------------
# state_transitions
# ---------------------------------------------------------------------------------------------------------------


def state_transitions(x) -> list[int]:
    """The first sample, every sample differing from the previous one (NaN equals NaN), and the last sample."""
    values = np.asarray(x)
    n = values.shape[0]
    if n == 0:
        return []
    differs = values[1:] != values[:-1]
    if values.dtype.kind == "f":
        differs &= ~(np.isnan(values[1:]) & np.isnan(values[:-1]))
    if differs.ndim == 2:
        differs = differs.any(axis=1)
    indices = [0] + (np.flatnonzero(differs) + 1).tolist()
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return indices


# ---------------------------------------------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------------------------------------------


def event_triggers(
    t,
    x,
    condition: str,
    value: float,
    hysteresis: float = 0.0,
    debounce: float = 0.0,
    occurrence: str = "all",
    interpolate: str = "linear",
) -> dict:
    """Event triggers: `{found, triggers, pending_at_end, pending_at_end_edge}`.

    `falls_below` / `rises_above` run the crossing machine with a fixed edge; `equals` takes the first sample of
    each run with `x == V` (hysteresis and debounce do not apply). Each trigger is
    `{t, index_before, index_after, at_start, gap}`: the bracketing samples of the crossing, or for `equals` the
    sample before the run (None at the start) and the run's first sample. `found` counts all triggers before
    `occurrence` selects the first, last or all of them.

    An event triggers on one fixed edge, so `pending_at_end_edge` -- pending only when the final short run was
    started by a flip in the event's own direction -- is usually the caveat an event wants; `pending_at_end` is
    the raw step 4 flag, which a flip in the other direction can also set. `equals` has no debounce and reports
    both as False.
    """
    if occurrence not in OCCURRENCES:
        raise ValueError(f"occurrence must be one of {OCCURRENCES}")
    pending = False
    pending_edge = False
    if condition in ("falls_below", "rises_above"):
        detected = crossings(
            t,
            x,
            value,
            edge="falling" if condition == "falls_below" else "rising",
            hysteresis=hysteresis,
            debounce=debounce,
            interpolate=interpolate,
        )
        pending = detected["pending_at_end"]
        pending_edge = detected["pending_at_end_edge"]
        triggers = [
            {
                "t": c["t"],
                "index_before": c["index_before"],
                "index_after": c["index_after"],
                "at_start": False,
                "gap": c["gap"],
            }
            for c in detected["accepted"]
        ]
    elif condition == "equals":
        times = np.asarray(t, dtype=np.float64)
        hit = np.asarray(x) == value
        run_starts = np.flatnonzero(hit & np.concatenate(([True], ~hit[:-1]))) if hit.size else np.zeros(0, int)
        triggers = [
            {
                "t": float(times[s]),
                "index_before": s - 1 if s > 0 else None,
                "index_after": s,
                "at_start": s == 0,
                "gap": False,
            }
            for s in run_starts.tolist()
        ]
    else:
        raise ValueError("condition must be falls_below, rises_above or equals")

    if occurrence == "first":
        selected = triggers[:1]
    elif occurrence == "last":
        selected = triggers[-1:]
    else:
        selected = triggers
    return {
        "found": len(triggers),
        "triggers": selected,
        "pending_at_end": pending,
        "pending_at_end_edge": pending_edge,
    }


def event_window_indices(t, tau: float, before: float, after: float) -> tuple[int, int] | None:
    """Inclusive source index range retained for an event window, or None for an empty signal.

    `lo` = first index with `t >= tau - before`, `hi` = last index with `t <= tau + after`; the retained range is
    `[max(lo - 1, 0), min(hi + 1, n - 1)]`.
    """
    times = np.asarray(t, dtype=np.float64)
    n = times.shape[0]
    if n == 0:
        return None
    lo = int(np.searchsorted(times, float(tau) - float(before), side="left"))  # n when no sample qualifies
    hi = int(np.searchsorted(times, float(tau) + float(after), side="right")) - 1  # -1 when no sample qualifies
    return max(lo - 1, 0), min(hi + 1, n - 1)


def event_window_clipped(t, tau: float, before: float, after: float) -> bool:
    """True when the window `[tau - before, tau + after]` extends past the signal's first or last timestamp."""
    times = np.asarray(t, dtype=np.float64)
    if times.shape[0] == 0:
        return True
    return float(tau) - float(before) < float(times[0]) or float(tau) + float(after) > float(times[-1])


# ---------------------------------------------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------------------------------------------


def sed_max(t, P, t_ret, P_ret) -> tuple[float, int | None]:
    """Largest synchronized Euclidean distance `‖P[i] - recon(t[i])‖` over the finite source samples, and its index.

    - Strictly between two distinct retained timestamps, `recon` is the linear interpolation of each component
      (`np.interp`).
    - At a timestamp that has retained samples, the path is at those retained samples; equal retained timestamps
      are a discontinuity, so the distance is to the nearest retained sample with that exact timestamp.
    - A finite source sample outside the retained time span, or inside a gap (a non-finite bracketing retained
      sample), has no reconstruction: its distance is inf.
    - Non-finite source samples are not evaluated. Returns `(0.0, None)` when no source sample is finite.

    Ties return the lowest index.
    """
    times = np.asarray(t, dtype=np.float64)
    points = np.asarray(P, dtype=np.float64)
    tr = np.asarray(t_ret, dtype=np.float64)
    pr = np.asarray(P_ret, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    if pr.ndim == 1:
        pr = pr[:, None]
    n = times.shape[0]
    m = tr.shape[0]
    source_ok = finite(points)
    sed = np.full(n, np.nan)
    if not source_ok.any():
        return 0.0, None
    if m == 0:
        sed[source_ok] = np.inf
    else:
        retained_ok = finite(pr)
        first_at_or_after = np.searchsorted(tr, times, side="left")
        first_after = np.searchsorted(tr, times, side="right")
        at_knot = source_ok & (first_after > first_at_or_after)
        strictly_inside = source_ok & ~at_knot & (first_at_or_after > 0) & (first_at_or_after < m)
        outside = source_ok & ~at_knot & ~strictly_inside
        sed[outside] = np.inf

        # Strictly between retained samples a = first_at_or_after - 1 and b = first_at_or_after.
        inside = np.flatnonzero(strictly_inside)
        b = first_at_or_after[inside]
        a = b - 1
        bracket_ok = retained_ok[a] & retained_ok[b]
        recon = np.column_stack([np.interp(times[inside], tr, pr[:, c]) for c in range(pr.shape[1])])
        distance = np.sqrt(np.sum((points[inside] - recon) ** 2, axis=1))
        sed[inside] = np.where(bracket_ok, distance, np.inf)

        # On a retained timestamp: nearest of the retained samples [first_at_or_after, first_after) at that time.
        knots = np.flatnonzero(at_knot)
        lo = first_at_or_after[knots]
        hi = first_after[knots] - 1
        d_lo = np.sqrt(np.sum((points[knots] - pr[lo]) ** 2, axis=1))
        d_hi = np.sqrt(np.sum((points[knots] - pr[hi]) ** 2, axis=1))
        nearest = np.fmin(d_lo, d_hi)
        for q in np.flatnonzero(hi - lo > 1).tolist():  # three or more retained samples share the timestamp
            group = pr[lo[q] : hi[q] + 1]
            nearest[q] = np.fmin.reduce(np.sqrt(np.sum((points[knots[q]] - group) ** 2, axis=1)))
        sed[knots] = np.where(np.isnan(nearest), np.inf, nearest)

    k = int(np.nanargmax(sed))
    return float(sed[k]), k
