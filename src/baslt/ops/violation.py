"""violation: runs of samples beyond a limit (docs/contracts.md, violation).

A violating run is a maximal run of consecutive finite samples with `x > above` (or `x < below`); a non-finite
sample ends a run. The start time is the linear crossing time of the limit between the sample before the run and
its first sample, or the first sample's time when the run opens the signal (`open_start`) or follows a non-finite
sample (`gap_start`). The end time is the crossing time between the last sample of the run and the next sample
(`t[e] + (t[e+1] - t[e]) * (L - x[e]) / (x[e+1] - x[e])`), or the last sample's time (`open_end`, `gap_end`).
Crossing times are clamped to their bracket (see `threshold_crossing.level_time`). Runs with
`end - start >= min_duration` are kept.

The boundary pairs of *every* candidate run are retained, not only those of the kept runs: the contract's second
guarantee ("detection on the reconstruction yields the same runs", checked by the verifier with basis `artifact`)
fails otherwise, because a sample retained inside a run that `min_duration` rejected -- the mandatory `gap`
retention alone is enough -- stretches that run back to the previous retained sample and makes it long enough to
keep. The `worst` role stays on the kept runs, which are the only ones the evidence claims.
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
    OpResult,
    capped,
    components,
    finite_mask,
    fnum,
)
from .threshold_crossing import level_time

if TYPE_CHECKING:
    from ..signals import Signal

OPEN_START = 1
OPEN_END = 2
GAP_START = 4
GAP_END = 8
FLAG_NAMES = ((OPEN_START, "open_start"), (OPEN_END, "open_end"), (GAP_START, "gap_start"), (GAP_END, "gap_end"))


@dataclass(slots=True)
class Runs:
    """Violating runs of one 1-D series, in sample order."""

    start: np.ndarray  # float64 start times
    end: np.ndarray  # float64 end times
    index_start: np.ndarray  # int64, first sample of the run
    index_end: np.ndarray  # int64, last sample of the run
    worst: np.ndarray  # int64, most extreme sample (lowest index on ties)
    flags: np.ndarray  # uint8, OPEN_START | OPEN_END | GAP_START | GAP_END
    n_candidates: int = 0  # runs before the min_duration filter

    @property
    def count(self) -> int:
        return int(self.start.shape[0])

    def select(self, mask) -> Runs:
        return Runs(
            start=self.start[mask],
            end=self.end[mask],
            index_start=self.index_start[mask],
            index_end=self.index_end[mask],
            worst=self.worst[mask],
            flags=self.flags[mask],
            n_candidates=self.n_candidates,
        )

    def edge_indices(self, n: int) -> np.ndarray:
        """Both boundary pairs of every run: (start - 1, start) and (end, end + 1), clipped to the signal."""
        pts = np.concatenate([self.index_start - 1, self.index_start, self.index_end, self.index_end + 1])
        return pts[(pts >= 0) & (pts < n)]


def flag_names(flags: int) -> list[str]:
    return [name for bit, name in FLAG_NAMES if int(flags) & bit]


def _empty_runs() -> Runs:
    f = np.empty(0, np.float64)
    i = np.empty(0, np.int64)
    return Runs(f, f.copy(), i, i.copy(), i.copy(), np.empty(0, np.uint8), 0)


def _limit(above, below) -> tuple[float, bool]:
    if (above is None) == (below is None):
        raise ValueError("exactly one of above and below must be given")
    raw = above if above is not None else below
    try:
        limit = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"limit must be a number, got {raw!r}") from None
    if np.isnan(limit):
        raise ValueError("limit must not be NaN")
    return limit, above is not None


def all_runs(t, x, *, above: float | None = None, below: float | None = None) -> Runs:
    """Every violating run, before the min_duration filter."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.ndim != 1 or x.shape != t.shape:
        raise ValueError(f"t and x must be 1-D arrays of equal length, got {t.shape} and {x.shape}")
    limit, is_above = _limit(above, below)
    n = x.shape[0]
    fin = np.isfinite(x)
    viol = fin & ((x > limit) if is_above else (x < limit))
    if not viol.any():
        return _empty_runs()

    d = np.diff(viol.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1) - 1

    open_start = s == 0
    gap_start = ~open_start & ~fin[np.maximum(s - 1, 0)]
    start = t[s].copy()
    interp = ~open_start & ~gap_start
    if interp.any():
        start[interp] = level_time(t, x, s[interp] - 1, s[interp], limit)

    open_end = e == n - 1
    gap_end = ~open_end & ~fin[np.minimum(e + 1, n - 1)]
    end = t[e].copy()
    interp = ~open_end & ~gap_end
    if interp.any():
        end[interp] = level_time(t, x, e[interp], e[interp] + 1, limit)

    # Most extreme sample per run, lowest index on ties.
    score = np.where(viol, x if is_above else -x, -np.inf)
    run_best = np.maximum.reduceat(score, s)
    vi = np.flatnonzero(viol)
    rid = np.searchsorted(s, vi, side="right") - 1
    best = score[vi] == run_best[rid]
    _, first = np.unique(rid[best], return_index=True)
    worst = vi[best][first]

    flags = (
        open_start.astype(np.uint8) * OPEN_START
        | open_end.astype(np.uint8) * OPEN_END
        | gap_start.astype(np.uint8) * GAP_START
        | gap_end.astype(np.uint8) * GAP_END
    ).astype(np.uint8)
    return Runs(start, end, s.astype(np.int64), e.astype(np.int64), worst.astype(np.int64), flags, int(s.shape[0]))


def detect_runs(t, x, *, above: float | None = None, below: float | None = None, min_duration: float = 0.0) -> Runs:
    """Violating runs with `end - start >= min_duration`."""
    m = _duration(min_duration)
    runs = all_runs(t, x, above=above, below=below)
    return runs.select((runs.end - runs.start) >= m)


def _duration(value) -> float:
    try:
        m = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"min_duration must be a number, got {value!r}") from None
    if np.isnan(m) or m < 0:
        raise ValueError(f"min_duration must be >= 0, got {value!r}")
    return m


def evaluate(sig: Signal, params: Mapping[str, object], bits: Mapping[str, int]) -> OpResult:
    """Kept runs of every component; retains both boundary pairs (`edge`) and the worst sample (`worst`)."""
    above = params.get("above")
    below = params.get("below")
    limit, is_above = _limit(above, below)
    m = _duration(params.get("min_duration", 0.0) or 0.0)
    edge_bit = int(bits["edge"])
    worst_bit = int(bits["worst"])

    v = sig.v
    t = sig.t
    n = t.shape[0]
    fin = finite_mask(v)
    n_finite = int(np.count_nonzero(fin))
    kw = {"above": limit} if is_above else {"below": limit}
    if n_finite == 0:
        evidence = {"count": 0, "total_duration": 0.0, "runs": []}
        return OpResult(SampleSet.empty(), evidence, STATUS_NOT_APPLICABLE, [f"{sig.name}: no finite sample"], raw=[])

    all_finite = n_finite == n
    raw = []
    kept_runs: list[Runs] = []
    comp_values: list[np.ndarray] = []
    samples = SampleSet.empty()
    for k, comp in components(v):
        xk = comp if all_finite else np.where(fin, comp, np.nan)
        runs = all_runs(t, xk, **kw)
        keep = (runs.end - runs.start) >= m
        kept = runs.select(keep)
        raw.append({"component": k, "runs": runs, "kept": keep})
        kept_runs.append(kept)
        comp_values.append(xk)
        if runs.count:
            samples = samples.union(SampleSet.from_points(runs.edge_indices(n), edge_bit))
        if kept.count:
            samples = samples.union(SampleSet.from_points(kept.worst, worst_bit))

    count = sum(r.count for r in kept_runs)
    total = float(sum(float(np.sum(r.end - r.start)) for r in kept_runs))

    comp_ids = np.concatenate([np.full(r.count, k, np.int64) for k, r in enumerate(kept_runs)])
    pos_ids = np.concatenate([np.arange(r.count, dtype=np.int64) for r in kept_runs])
    starts = np.concatenate([r.start for r in kept_runs])
    order = np.lexsort((comp_ids, starts))[:EVIDENCE_LIMIT]
    items = []
    for o in order:
        k = int(comp_ids[o])
        r = kept_runs[k]
        p = int(pos_ids[o])
        w = int(r.worst[p])
        items.append(
            {
                "start": fnum(r.start[p]),
                "end": fnum(r.end[p]),
                "worst_t": fnum(t[w]),
                "worst_value": fnum(comp_values[k][w]),
                "component": k,
                "flags": flag_names(int(r.flags[p])),
            }
        )
    evidence = {"count": int(count), "total_duration": total, "runs": capped(items)}
    return OpResult(samples, evidence, STATUS_PASS, [], raw=raw)
