"""threshold_crossing: level crossings with hysteresis, debounce and an edge filter.

Implements steps 1-5 of the threshold_crossing contract in docs/contracts.md:

1. level crossings between consecutive finite samples (a crossing whose bracket spans non-finite samples uses
   the bracketing finite samples and is marked `gap`);
2. a Schmitt state (`hi = V + H/2`, `lo = V - H/2`, initial state `x[first finite] >= V`, non-finite samples hold);
3. each state flip takes the last same-direction level crossing whose after-sample is not later than the flip;
4. debounce on run durations between flip times, the last run ending at the last sample time;
5. the edge filter.

The retained set is the bracket of *every* flip of step 3, not only of the crossings that survive steps 4 and 5:
the contract's second guarantee ("running steps 1-5 on the reconstruction yields exactly the same accepted
crossings", checked by the verifier with basis `artifact`) needs the rejected flips too, because a flip that is
missing from the reconstruction moves the run boundaries that debounce measures, and a flip whose bracket is only
half retained is detected at a different time. With H > 0 the confirming sample of every flip is retained as well.

Two floating-point details are fixed here because the formulas alone leave them open:

- `hi` and `lo` are computed in float64. In exact arithmetic `lo = V - H/2 < V` for every `H > 0`, so a sample that
  marks the state low lies strictly below `V` and step 3 always finds a falling level crossing to attach the flip
  to. Rounding collapses `V - H/2` onto `V` for a band narrower than one ulp of `V`, so the low mark is
  `x <= lo and x < V`: without the second condition a sample at exactly `V` would mark the state low while step 1
  reports no falling crossing at all, and step 3 would be undefined (see
  tests/unit/test_op_threshold_crossing.py::test_band_rounding_onto_level_keeps_step_3_well_defined).
- The crossing time lies inside its bracket. Rounding can put the formula one ulp past `t[i+1]`, so the result is
  clamped to `[t[i], t[i+1]]`; step 4 measures run lengths between crossing times, which a time outside its own
  bracket would make negative. When `x[i+1] - x[i]` overflows the fraction is computed on halved values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..sampleset import SampleSet
from ._common import (
    EVIDENCE_LIMIT,
    STATUS_NOT_APPLICABLE,
    STATUS_PASS,
    STATUS_WARN,
    OpResult,
    capped,
    components,
    finite_mask,
    fnum,
)

if TYPE_CHECKING:
    from ..signals import Signal

EDGES = ("rising", "falling", "both")
INTERPOLATIONS = ("linear", "none")
RISING = 1
FALLING = -1
EDGE_NAMES = {RISING: "rising", FALLING: "falling"}


@dataclass(slots=True)
class Flips:
    """Every state flip before debounce, in time order (the input for closure repair)."""

    t: np.ndarray  # float64, crossing time of the flip's level crossing
    edge: np.ndarray  # int8, +1 rising / -1 falling
    index_before: np.ndarray  # int64, finite sample before the level crossing
    index_after: np.ndarray  # int64, finite sample after it (index_before + 1 unless gap)
    confirm: np.ndarray  # int64, sample j at which the state changed
    gap: np.ndarray  # bool, the level crossing spans non-finite samples
    duration: np.ndarray  # float64, length of the run the flip starts
    accepted: np.ndarray  # bool, duration >= debounce
    transition: np.ndarray  # bool, accepted and changes the accepted state (before the edge filter)


@dataclass(slots=True)
class Detection:
    """Accepted crossings of one 1-D series after steps 1-5."""

    t: np.ndarray  # float64 crossing times
    edge: np.ndarray  # int8, +1 rising / -1 falling
    index_before: np.ndarray  # int64, bracket start i
    index_after: np.ndarray  # int64, bracket end (i + 1, or the next finite sample for a gap crossing)
    confirm: np.ndarray  # int64, confirming sample j
    gap: np.ndarray  # bool
    discretization_error: np.ndarray  # float64, t[after] - t[before] with interpolate "none", else 0
    pending_at_end: bool
    initial_state: bool | None  # x[first finite] >= V, None without finite samples
    n_level_crossings: int
    n_flips: int
    flips: Flips

    @property
    def count(self) -> int:
        return int(self.t.shape[0])

    @property
    def gap_flips(self) -> int:
        """The contract's `count of gap flips` evidence: flips of step 3 whose level crossing spans a gap.

        Counted before debounce and before the edge filter, because "flip" is the step-3 term.
        """
        return int(np.count_nonzero(self.flips.gap))


def _empty_flips() -> Flips:
    i64 = np.empty(0, np.int64)
    return Flips(
        t=np.empty(0, np.float64),
        edge=np.empty(0, np.int8),
        index_before=i64,
        index_after=i64.copy(),
        confirm=i64.copy(),
        gap=np.empty(0, bool),
        duration=np.empty(0, np.float64),
        accepted=np.empty(0, bool),
        transition=np.empty(0, bool),
    )


def _float_param(value, name: str, *, minimum: float | None = None, allow_inf: bool = False) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if np.isnan(out) or (not allow_inf and np.isinf(out)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return out


def level_time(t: np.ndarray, x: np.ndarray, before: np.ndarray, after: np.ndarray, level: float) -> np.ndarray:
    """Linear crossing time of `level` between samples `before` and `after` (vectorized).

    `tc = t[i] + (t[j] - t[i]) * (level - x[i]) / (x[j] - x[i])`, clamped to `[t[i], t[j]]`. The caller guarantees
    that `level` lies between `x[i]` and `x[j]` and that the two values differ.
    """
    ta = t[before]
    tb = t[after]
    xa = x[before]
    xb = x[after]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        num = level - xa
        den = xb - xa
        # Evaluated in the contract's order so an independent implementation of the formula agrees bitwise.
        tc = ta + (tb - ta) * num / den
        bad = ~(np.isfinite(num) & np.isfinite(den) & np.isfinite(tc))
        if bad.any():
            frac = (0.5 * level - 0.5 * xa[bad]) / (0.5 * xb[bad] - 0.5 * xa[bad])
            np.clip(frac, 0.0, 1.0, out=frac)
            tc[bad] = ta[bad] + (tb[bad] - ta[bad]) * frac
    return np.fmin(np.fmax(tc, ta), tb)


def detect(
    t,
    x,
    value: float,
    *,
    edge: str = "both",
    hysteresis: float = 0.0,
    debounce: float = 0.0,
    interpolate: str = "linear",
) -> Detection:
    """Run contract steps 1-5 on one 1-D series (source samples or a reconstruction's retained samples)."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.ndim != 1 or x.shape != t.shape:
        raise ValueError(f"t and x must be 1-D arrays of equal length, got {t.shape} and {x.shape}")
    level = _float_param(value, "value")
    hyst = _float_param(hysteresis, "hysteresis", minimum=0.0, allow_inf=True)
    bounce = _float_param(debounce, "debounce", minimum=0.0, allow_inf=True)
    if edge not in EDGES:
        raise ValueError(f"edge must be one of {', '.join(EDGES)}, got {edge!r}")
    if interpolate not in INTERPOLATIONS:
        raise ValueError(f"interpolate must be one of {', '.join(INTERPOLATIONS)}, got {interpolate!r}")

    fin_idx = np.flatnonzero(np.isfinite(x))
    initial_state = bool(x[fin_idx[0]] >= level) if fin_idx.shape[0] else None
    if fin_idx.shape[0] < 2:
        return _detection_from(t, _empty_flips(), np.empty(0, bool), initial_state, 0, False, interpolate)

    xf = x[fin_idx]

    # Step 1: level crossings between consecutive finite samples.
    below_a = xf[:-1] < level
    above_b = xf[1:] >= level
    rising = below_a & above_b
    falling = ~below_a & ~above_b
    pos = np.flatnonzero(rising | falling)
    lc_before = fin_idx[pos]
    lc_after = fin_idx[pos + 1]
    lc_rising = rising[pos]

    # Step 2: Schmitt state on finite samples; only marked samples can change it.
    hi = level + hyst / 2
    lo = level - hyst / 2
    high = xf >= hi
    low = (xf <= lo) & (xf < level)
    marked = high | low
    marked[0] = True
    state = high
    state[0] = xf[0] >= level
    mark_pos = np.flatnonzero(marked)
    mark_state = state[mark_pos]
    change = np.flatnonzero(mark_state[1:] != mark_state[:-1]) + 1
    confirm = fin_idx[mark_pos[change]]
    flip_rising = mark_state[change]

    # Step 3: last same-direction level crossing with after-index <= j.
    n_flips = int(change.shape[0])
    cross = np.empty(n_flips, np.int64)
    for want in (True, False):
        sel = flip_rising == want
        if not sel.any():
            continue
        cand = np.flatnonzero(lc_rising == want)
        q = np.searchsorted(lc_after[cand], confirm[sel], side="right") - 1
        if q.shape[0] and int(q.min()) < 0:
            raise AssertionError("state flip without a preceding level crossing")
        cross[sel] = cand[q]
    before = lc_before[cross]
    after = lc_after[cross]
    if interpolate == "linear":
        tc = level_time(t, x, before, after, level)
    else:
        tc = t[after]

    # Step 4: debounce. Flips are in sample order, which is time order because every crossing time lies inside
    # its bracket and brackets of successive flips do not overlap.
    duration = np.empty(n_flips, np.float64)
    if n_flips:
        duration[:-1] = tc[1:] - tc[:-1]
        duration[-1] = t[-1] - tc[-1]
    accepted = duration >= bounce
    acc = np.flatnonzero(accepted)
    new_state = flip_rising[acc]
    prev_state = np.concatenate([[initial_state], new_state[:-1]]).astype(bool)
    transition = np.zeros(n_flips, bool)
    transition[acc[new_state != prev_state]] = True
    pending = bool(n_flips and duration[-1] < bounce)

    flips = Flips(
        t=tc,
        edge=np.where(flip_rising, RISING, FALLING).astype(np.int8),
        index_before=before,
        index_after=after,
        confirm=confirm,
        gap=after - before > 1,
        duration=duration,
        accepted=accepted,
        transition=transition,
    )

    # Step 5: edge filter.
    keep = transition.copy()
    if edge == "rising":
        keep &= flip_rising
    elif edge == "falling":
        keep &= ~flip_rising
    return _detection_from(t, flips, keep, initial_state, int(pos.shape[0]), pending, interpolate)


def _detection_from(t, flips: Flips, keep, initial_state, n_level, pending, interpolate) -> Detection:
    before = flips.index_before[keep]
    after = flips.index_after[keep]
    if interpolate == "none":
        disc = t[after] - t[before]
    else:
        disc = np.zeros(before.shape[0], np.float64)
    return Detection(
        t=flips.t[keep],
        edge=flips.edge[keep],
        index_before=before,
        index_after=after,
        confirm=flips.confirm[keep],
        gap=flips.gap[keep],
        discretization_error=disc,
        pending_at_end=bool(pending),
        initial_state=initial_state,
        n_level_crossings=int(n_level),
        n_flips=int(flips.t.shape[0]),
        flips=flips,
    )


def evaluate(sig: Signal, params: Mapping[str, object], bits: Mapping[str, int]) -> OpResult:
    """Accepted crossings of every component.

    Retains the bracket of every flip of step 3 -- including the flips that debounce or the edge filter drops --
    and, with H > 0, their confirming samples. Only the accepted crossings are reported as evidence.
    """
    value = _float_param(params["value"], "value")
    edge = str(params.get("edge", "both"))
    hysteresis = _float_param(params.get("hysteresis", 0.0) or 0.0, "hysteresis", minimum=0.0, allow_inf=True)
    debounce = _float_param(params.get("debounce", 0.0) or 0.0, "debounce", minimum=0.0, allow_inf=True)
    interpolate = str(params.get("interpolate", "linear"))
    bit = int(bits["main"])

    v = sig.v
    t = sig.t
    fin = finite_mask(v)
    n_finite = int(np.count_nonzero(fin))
    none_mode = interpolate == "none"
    if n_finite < 2:
        # Validate the enums even when nothing can be evaluated.
        detect(t[:0], np.empty(0), value, edge=edge, hysteresis=hysteresis, debounce=debounce,
               interpolate=interpolate)
        evidence = _evidence([], 0, 0, False, 0, none_mode, 0.0)
        note = f"{sig.name}: fewer than 2 finite samples"
        return OpResult(SampleSet.empty(), evidence, STATUS_NOT_APPLICABLE, [note], raw=[])

    all_finite = n_finite == v.shape[0]
    detections: list[Detection] = []
    points: list[np.ndarray] = []
    for k, comp in components(v):
        xk = comp if all_finite else np.where(fin, comp, np.nan)
        det = detect(t, xk, value, edge=edge, hysteresis=hysteresis, debounce=debounce, interpolate=interpolate)
        detections.append(det)
        points.extend([det.flips.index_before, det.flips.index_after])
        if hysteresis > 0:
            points.append(det.flips.confirm)

    samples = SampleSet.from_points(np.concatenate(points), bit) if points else SampleSet.empty()

    n_rising = sum(int(np.count_nonzero(d.edge == RISING)) for d in detections)
    n_falling = sum(int(np.count_nonzero(d.edge == FALLING)) for d in detections)
    pending = any(d.pending_at_end for d in detections)
    gap_flips = sum(d.gap_flips for d in detections)
    max_disc = max((float(d.discretization_error.max()) for d in detections if d.count), default=0.0)

    comp_ids = np.concatenate([np.full(d.count, k, np.int64) for k, d in enumerate(detections)])
    pos_ids = np.concatenate([np.arange(d.count, dtype=np.int64) for d in detections])
    times = np.concatenate([d.t for d in detections])
    order = np.lexsort((comp_ids, times))[:EVIDENCE_LIMIT]
    items = []
    for o in order:
        d = detections[int(comp_ids[o])]
        p = int(pos_ids[o])
        item = {
            "t": fnum(d.t[p]),
            "edge": EDGE_NAMES[int(d.edge[p])],
            "index_before": fnum(d.index_before[p]),
            "component": int(comp_ids[o]),
        }
        if none_mode:
            item["discretization_error"] = fnum(d.discretization_error[p])
        items.append(item)

    evidence = _evidence(items, n_rising, n_falling, pending, gap_flips, none_mode, max_disc)
    notes = []
    if pending:
        notes.append(f"{sig.name}: final run shorter than debounce (pending_at_end)")
    if gap_flips:
        notes.append(f"{sig.name}: {gap_flips} state flip(s) span non-finite samples")
    status = STATUS_WARN if notes else STATUS_PASS
    return OpResult(samples, evidence, status, notes, raw=detections)


def _evidence(items, n_rising, n_falling, pending, gap_flips, none_mode, max_disc) -> dict:
    evidence = {
        "count": int(n_rising + n_falling),
        "rising": int(n_rising),
        "falling": int(n_falling),
        "crossings": capped(items),
        "pending_at_end": bool(pending),
        "gap_flips": int(gap_flips),
    }
    if none_mode:
        evidence["max_discretization_error"] = float(max_disc)
    return evidence
