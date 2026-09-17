"""Plot data for the run report: small, deduplicated traces with the limits, windows and violations of each check.

Traces keep every sample up to 1024 samples. Longer ones keep the lowest and highest sample of 512 equal-count
bins, so the global extremes always survive, plus the first sample of each stretch of missing data so lines
break there. Failing and warning checks also get detail windows at full resolution around their worst point and
their first violations. Times are stored relative to `base` (time zero of the run, or 0).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..reduce.rank import minmax_bins
from ..report.html import Packer
from .check import RequirementResult, Trace, _intervals
from .evaluate import EvalError, RunContext
from .expr import ExprError
from .units_ext import UnitsError

__all__ = ["build_plots", "overview_indices"]

OVERVIEW_BINS = 512
RAW_LIMIT = 2 * OVERVIEW_BINS
GAP_MARKS = 256
STEP_LIMIT = 512
DETAIL_HALF = 64
DETAIL_MAX = 512
DETAIL_WINDOWS = 4
PHASES = 16
PHASE_SPANS = 64
PLOT_ERRORS = (EvalError, ExprError, UnitsError, LookupError, ValueError)


def overview_indices(values: np.ndarray) -> np.ndarray:
    """Sample indices that keep the shape of a 1-D series: all of them, or bin extremes and gap starts."""
    n = int(values.shape[0])
    if n <= RAW_LIMIT:
        return np.arange(n, dtype=np.int64)
    finite = np.isfinite(values)
    _, amin, _, amax = minmax_bins(values, finite, OVERVIEW_BINS)
    keep = [amin[amin < n], amax[amax < n], np.array([0, n - 1], dtype=np.int64)]
    bad = ~finite
    if bad.any():
        starts = np.flatnonzero(bad & ~np.concatenate([[False], bad[:-1]]))
        if starts.shape[0] > GAP_MARKS:
            ends = np.flatnonzero(bad & ~np.concatenate([bad[1:], [False]]))
            longest = np.argsort(ends - starts, kind="stable")[::-1][:GAP_MARKS]
            starts = starts[np.sort(longest)]
        keep.append(starts)
    return np.unique(np.concatenate(keep))


def _step_indices(values: np.ndarray) -> np.ndarray | None:
    """Where a piecewise-constant series changes (NaN counts as a value), or None when it changes too often."""
    n = values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    same = (values[1:] == values[:-1]) | (np.isnan(values[1:]) & np.isnan(values[:-1]))
    changes = np.flatnonzero(~same) + 1
    if changes.shape[0] > STEP_LIMIT:
        return None
    return np.concatenate([[0], changes, [n - 1]]).astype(np.int64)


def _rel(value, base: float) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value - base if math.isfinite(value) else None


def _num(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class PlotBuilder:
    def __init__(self, ctx: RunContext, base: float, packer: Packer) -> None:
        self.ctx = ctx
        self.base = base
        self.packer = packer
        self.series: list[dict] = []
        self._series_index: dict[tuple, int] = {}

    # ----- arrays -----------------------------------------------------------------------------------------

    def times(self, t: np.ndarray) -> dict:
        rel = np.asarray(t, dtype=np.float64) - self.base
        if rel.shape[0] == 0:
            return self.packer.uint16(rel, 0.0, 0.0)
        return self.packer.uint16(rel, float(rel[0]), float(rel[-1]))

    def add_series(self, key: tuple, name: str, t: np.ndarray, values: np.ndarray, *, unit, discrete: bool,
                   labels) -> int:
        found = self._series_index.get(key)
        if found is not None:
            return found
        values = np.asarray(values, dtype=np.float64)
        keep = overview_indices(values)
        entry = {"name": name, "unit": unit if unit not in (None, "1", "?") else None, "discrete": bool(discrete),
                 "labels": labels, "t": self.times(t[keep]), "v": self.packer.uint16(values[keep]),
                 "samples": int(values.shape[0])}
        self.series.append(entry)
        self._series_index[key] = len(self.series) - 1
        return len(self.series) - 1

    def limit_line(self, side: str, t: np.ndarray, values: np.ndarray) -> dict | None:
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).any():
            return None
        steps = _step_indices(values)
        if steps is not None:
            return {"side": side, "step": True, "t": self.times(t[steps]), "v": self.packer.uint16(values[steps])}
        keep = overview_indices(values)
        return {"side": side, "step": False, "t": self.times(t[keep]), "v": self.packer.uint16(values[keep])}

    # ----- one requirement --------------------------------------------------------------------------------

    def labels(self, node) -> list | None:
        try:
            labels = self.ctx.labels_of(node)
        except PLOT_ERRORS:
            return None
        if not labels:
            return None
        if isinstance(labels, Mapping):
            return [[int(code), str(name)] for code, name in sorted(labels.items())]
        return [[code, str(name)] for code, name in enumerate(labels)]

    def plot(self, result: RequirementResult) -> dict | None:
        trace: Trace | None = result.trace
        if trace is None or trace.grid is None:
            return None
        grid = trace.grid
        t = grid.t
        spec: dict = {"series": [], "limits": [], "lines": [], "spans": [], "details": [], "unit": None,
                      "windows": [[_rel(a, self.base), _rel(b, self.base)] for a, b in trace.windows]}
        subject = trace.subject
        if subject is None:
            return None
        if trace.x is not None:
            values = trace.x
        else:
            try:
                values = np.asarray(self.ctx.series(subject, grid), dtype=np.float64)
            except PLOT_ERRORS:
                return None
        x = values if values.ndim == 2 else values[:, None]
        labels = self.labels(subject) if subject.type.dtype == "enum" else None
        label = _describe(subject)
        unit = result.unit if trace.x is not None else subject.type.unit
        for c in range(x.shape[1]):
            key = (subject.key, grid.key, c)
            name = label if x.shape[1] == 1 else f"{label} [{'xyz'[c] if x.shape[1] == 3 else c}]"
            spec["series"].append(self.add_series(key, name, t, x[:, c], unit=unit,
                                                  discrete=trace.discrete or subject.type.discrete, labels=labels))
        spec["unit"] = unit
        for side, values in (("upper", trace.upper), ("lower", trace.lower)):
            if values is not None:
                line = self.limit_line(side, t, values)
                if line is not None:
                    spec["limits"].append(line)
        spec["lines"] = [{"kind": kind, "v": _num(value)} for kind, value in trace.lines if _num(value) is not None]
        if trace.spans:
            spec["spans"] = [{"s": _rel(a, self.base), "e": _rel(b, self.base), "held": True}
                             for a, b in trace.spans]
        for run in result.runs:
            spec["spans"].append({"s": _rel(run["start"], self.base), "e": _rel(run["end"], self.base),
                                  "tol": bool(run["tolerated"]), "c": run.get("component", 0)})
        if trace.band is not None:
            lo, hi = trace.band
            spec["band"] = [_rel(lo, self.base), _rel(hi, self.base)]
        if trace.marker is not None:
            spec["marker"] = {"t": _rel(trace.marker[0], self.base), "v": _num(trace.marker[1])}
        elif trace.x is not None and result.at is not None and isinstance(result.value, float):
            spec["marker"] = {"t": _rel(result.at, self.base), "v": _num(result.value),
                              "c": int(result.totals.get("component", 0))}
        if result.verdict in ("fail", "warn") and grid.n > RAW_LIMIT:
            spec["details"] = self.details(result, trace, x)
        return spec

    def details(self, result: RequirementResult, trace: Trace, x: np.ndarray) -> list[dict]:
        t = trace.grid.t
        n = t.shape[0]
        wanted: list[tuple[int, int]] = []
        if result.at is not None:
            k = int(np.searchsorted(t, result.at))
            wanted.append((k - DETAIL_HALF, k + DETAIL_HALF + 1))
        for run in [r for r in result.runs if not r["tolerated"]][:DETAIL_WINDOWS - 1]:
            first = int(np.searchsorted(t, run["start"]))
            last = int(np.searchsorted(t, run["end"]))
            wanted.append((first - DETAIL_HALF, min(last, first + DETAIL_MAX - 2 * DETAIL_HALF) + DETAIL_HALF + 1))
        spans: list[list[int]] = []
        for lo, hi in sorted((max(0, lo), min(n, hi)) for lo, hi in wanted):
            if spans and lo <= spans[-1][1] and hi - spans[-1][0] <= DETAIL_MAX:
                spans[-1][1] = max(spans[-1][1], hi)
            else:
                spans.append([lo, hi])
        out = []
        for lo, hi in spans[:DETAIL_WINDOWS]:
            piece = {"t": self.packer.float32(t[lo:hi] - self.base),
                     "v": [self.packer.uint16(x[lo:hi, c]) for c in range(x.shape[1])], "limits": []}
            for side, values in (("upper", trace.upper), ("lower", trace.lower)):
                if values is not None and np.isfinite(values[lo:hi]).any():
                    piece["limits"].append({"side": side, "step": bool(trace.discrete),
                                            "v": self.packer.uint16(values[lo:hi])})
            out.append(piece)
        return out

    # ----- the run ----------------------------------------------------------------------------------------

    def phases(self) -> list[dict]:
        out = []
        for name in list(self.ctx.config.conditions)[:PHASES]:
            try:
                bound = self.ctx.bind(name)
                grid = self.ctx.grid_for(bound.signals)
                truth = self.ctx.truth(bound.root, grid)
            except PLOT_ERRORS:
                continue
            spans = _intervals(grid.t, truth.active(), limit=PHASE_SPANS)
            out.append({"name": name, "spans": [[_rel(a, self.base), _rel(b, self.base)] for a, b in spans]})
        return out


def _describe(node) -> str:
    """A short name for a plotted node: the alias or signal it reads."""
    current = node
    while True:
        if current.op == "alias":
            return str(current.value)
        if current.op == "sig":
            name, alias = current.value
            return alias or name
        if current.op == "stack":
            return str(current.value)
        if current.op in ("unit", "scale", "cond") and current.args:
            current = current.args[0]
            continue
        break
    inner = [arg for arg in node.args if hasattr(arg, "op")]
    if node.op == "comp" and inner:
        return f"{_describe(inner[0])}.{'xyz'[node.value] if node.value < 3 else node.value}"
    if node.op in ("agg", "movmean", "deriv", "abs", "norm", "neg") and len(inner) == 1:
        name = node.value if node.op == "agg" else node.op
        return f"{name}({_describe(inner[0])})" if node.op != "neg" else f"-{_describe(inner[0])}"
    return node.op if not inner else f"{node.op}({', '.join(_describe(arg) for arg in inner)})"


def build_plots(run, ctx: RunContext, packer: Packer) -> dict:
    """The plot part of the report manifest for one run."""
    base = float(run.t0) if run.t0 is not None else 0.0
    builder = PlotBuilder(ctx, base, packer)
    plots = {}
    for result in run.results:
        try:
            spec = builder.plot(result)
        except PLOT_ERRORS:
            spec = None
        if spec is not None:
            plots[result.id] = spec
    starts, ends = [], []
    for sig in ctx.run.signals.values():
        if sig.n:
            starts.append(float(sig.t[0]))
            ends.append(float(sig.t[-1]))
    events = [{"name": name, "times": [_rel(v, base) for v in info.get("times", [])]}
              for name, info in run.events.items()]
    return {
        "base": base,
        "span": [_rel(min(starts), base), _rel(max(ends), base)] if starts else [0.0, 0.0],
        "series": builder.series,
        "plots": plots,
        "events": events,
        "phases": builder.phases(),
    }
