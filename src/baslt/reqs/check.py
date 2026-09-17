"""Checking bound requirements on one run: the five check kinds and their verdicts.

    bound     every sample while a case's condition holds stays within that case's limit
    assert    a condition holds while a case's condition holds
    duration  the time a condition holds (inside the case's condition) meets a limit
    value     one number (an aggregate, an event time, a parameter...) meets a limit
    event     an event happens the expected number of times, at a time within the limit

Series checks (bound, assert) evaluate all cases on one clock: each sample belongs to the first case whose
conditions hold, and a violation is a run of consecutive samples beyond that sample's limit. A run no longer than
its tolerance is tolerated. Single-number checks evaluate each case in its own window. docs/requirements.md has
the exact rules; the verdict precedence is FAIL > ERROR > WARN > PASS > N/A.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..errors import Issue
from ..ops.violation import FLAG_NAMES, all_runs
from .bind import BoundCase, BoundLimit, BoundRequirement
from .evaluate import EvalError, RunContext, Tri, measure
from .expr import ExprError
from .model import worst
from .units_ext import UnitsError

__all__ = ["CaseResult", "RequirementResult", "evaluate_requirement"]

TINY = 5e-324  # the smallest positive float: a value exactly at a strict limit counts as beyond it
EVIDENCE_LIMIT = 256


@dataclass
class CaseResult:
    label: str
    verdict: str
    reason: str | None = None
    value: float | None = None
    limit: float | None = None
    margin: float | None = None
    margin_pct: float | None = None
    at: float | None = None
    runs: int = 0
    active_time: float | None = None
    applies: bool | None = True
    severity: str | None = None
    row: int | None = None

    def to_json(self) -> dict:
        return {key: _clean(getattr(self, key)) for key in (
            "label", "verdict", "reason", "value", "limit", "margin", "margin_pct", "at", "runs", "active_time",
            "applies", "severity", "row")}


@dataclass
class RequirementResult:
    id: str
    title: str
    kind: str
    check: str
    verdict: str = "pass"
    reason: str | None = None
    unit: str | None = None
    value: float | str | None = None
    limit: str | None = None  # the limit as it applies at the worst point, for display
    limit_value: float | None = None
    margin: float | None = None
    margin_pct: float | None = None
    at: float | None = None
    first_violation: float | None = None
    case: str | None = None
    severity: str | None = None
    runs: list[dict] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    loc: str = ""
    rows: list[int] = field(default_factory=list)
    passthrough: dict = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    trace: object = None  # what the report plots; not written to JSON

    def to_json(self) -> dict:
        out = {key: _clean(getattr(self, key)) for key in (
            "id", "title", "kind", "check", "verdict", "reason", "unit", "value", "limit", "limit_value", "margin",
            "margin_pct", "at", "first_violation", "case", "severity", "runs", "totals", "context", "notes", "loc",
            "rows", "passthrough", "events")}
        out["cases"] = [case.to_json() for case in self.cases]
        out["issues"] = [{"path": i.path, "message": i.message, "location": i.location} for i in self.issues]
        return out


def _clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        return _clean(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


@dataclass
class Trace:
    """Everything the report needs to draw a series check."""

    grid: object
    x: np.ndarray
    upper: np.ndarray | None
    lower: np.ndarray | None
    claims: np.ndarray
    windows: list[tuple[float, float]]
    discrete: bool


# ------------------------------------------------------------------------------------------------------------
# helpers


class _Outcome:
    """Verdict bookkeeping for one requirement."""

    def __init__(self) -> None:
        self.fail: list[str] = []
        self.warn: list[str] = []
        self.na: list[str] = []
        self.error: list[str] = []

    def verdict(self) -> tuple[str, str | None]:
        for name, reasons in (("fail", self.fail), ("error", self.error), ("warn", self.warn)):
            if reasons:
                return name, "; ".join(dict.fromkeys(reasons))
        if self.na:
            return "not_applicable", "; ".join(dict.fromkeys(self.na))
        return "pass", None


def _applies(case: BoundCase, ctx: RunContext) -> tuple[bool | None, str | None]:
    if case.applies_to is None:
        return True, None
    try:
        value = ctx.value(case.applies_to.root, None)
    except (EvalError, ExprError) as exc:
        return None, str(exc)
    if not isinstance(value, Tri):
        return None, "Applies to is not a condition"
    if not bool(np.all(value.known)):
        return None, "Applies to could not be decided (a parameter is not a number)"
    return bool(np.all(value.val)), None


def _grid(breq: BoundRequirement, ctx: RunContext):
    mode = breq.grid if breq.grid not in (None, "", "first") else None
    if mode is None:
        mode = ctx.config.defaults.grid if ctx.config.defaults.grid not in ("", "first") else None
    return ctx.grid_for(breq.grid_names(), mode)


def _run_bounds(runs, t: np.ndarray, discrete: bool) -> tuple[np.ndarray, np.ndarray]:
    """Run start and end times; discrete runs last until the next sample."""
    if not discrete:
        return runs.start, runs.end
    n = t.shape[0]
    start = t[runs.index_start]
    after = np.minimum(runs.index_end + 1, n - 1)
    end = np.where(runs.index_end + 1 < n, t[after], t[runs.index_end])
    return start, end


def _per_run_min(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """The minimum of `values` over each run [start, end]."""
    if starts.shape[0] == 0:
        return np.empty(0)
    padded = np.concatenate([values, [np.inf]])
    bounds = np.empty(2 * starts.shape[0], dtype=np.int64)
    bounds[0::2] = starts
    bounds[1::2] = ends + 1
    return np.minimum.reduceat(padded, bounds)[0::2]


def _flag_names(flags: int) -> list[str]:
    return [name for bit, name in FLAG_NAMES if int(flags) & bit]


def _intervals(t: np.ndarray, mask: np.ndarray, limit: int = 512) -> list[tuple[float, float]]:
    """[start, end] times of the runs of `mask` (a sample's interval reaches to the next sample)."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    n = t.shape[0]
    out = [(float(t[s]), float(t[min(e + 1, n - 1)])) for s, e in zip(starts, ends)]
    if len(out) > limit:
        gaps = sorted(range(len(out) - 1), key=lambda k: out[k + 1][0] - out[k][1])
        merge = set(gaps[:len(out) - limit])
        merged: list[tuple[float, float]] = []
        for k, span in enumerate(out):
            if merged and (k - 1) in merge:
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)
        out = merged
    return out


def _gap_time(t: np.ndarray, region: np.ndarray, bad: np.ndarray) -> float:
    """Time of the intervals inside `region` that touch a `bad` sample."""
    if t.shape[0] < 2:
        return 0.0
    dt = np.diff(t)
    touch = (bad[:-1] | bad[1:]) & region[:-1] & region[1:]
    return float(np.sum(dt[touch]))


def _scalar_side(side, ctx: RunContext, grid, window) -> float:
    value = ctx.scalar(side.node, grid, window)
    if isinstance(value, Tri) or isinstance(value, str):
        raise EvalError(f"the limit {side.text!r} is not a number")
    return float(value)


def _side_values(side, ctx: RunContext, grid) -> np.ndarray:
    value = ctx.value(side.node, grid)
    if isinstance(value, Tri) or isinstance(value, str):
        raise EvalError(f"the limit {side.text!r} is not a number")
    return np.broadcast_to(np.asarray(value, dtype=np.float64), (grid.n,))


def _context(ctx: RunContext, t_star: float | None) -> dict:
    """Named conditions holding and the last event before a moment, to say where something went wrong."""
    if t_star is None or not math.isfinite(t_star):
        return {}
    conditions = []
    for name in ctx.config.conditions:
        try:
            bound = ctx.bind(name)
            grid = ctx.grid_for(bound.signals)
            truth = ctx.truth(bound.root, grid)
            k = int(np.searchsorted(grid.t, t_star, side="right")) - 1
            if 0 <= k < grid.n and truth.known[k] and truth.val[k]:
                conditions.append(name)
        except (EvalError, ExprError, UnitsError, LookupError):
            continue
    last = None
    for name in ctx.config.events:
        try:
            times = ctx.event(name).times
        except (EvalError, ExprError, UnitsError, LookupError):
            continue
        before = times[times <= t_star]
        if before.shape[0] and (last is None or before[-1] > last[1]):
            last = (name, float(before[-1]))
    out: dict = {"conditions": conditions}
    if last is not None:
        out["event"] = last[0]
        out["since"] = t_star - last[1]
    return out


# ------------------------------------------------------------------------------------------------------------
# series checks


def _series(breq: BoundRequirement, ctx: RunContext, result: RequirementResult, outcome: _Outcome) -> None:
    defaults = ctx.config.defaults
    grid = _grid(breq, ctx)
    t = grid.t
    n = grid.n
    is_assert = breq.kind == "assert"
    root = breq.check.root
    discrete = grid.discrete or (root.type.discrete and not is_assert)

    claims = np.full(n, -1, dtype=np.int64)
    unknown = np.zeros(n, dtype=bool)
    overlap = np.zeros(n, dtype=np.int64)
    usable: list[int] = []
    for k, case in enumerate(breq.cases):
        applies, problem = _applies(case, ctx)
        record = result.cases[k]
        if problem is not None:
            record.verdict, record.reason = "error", problem
            outcome.error.append(f"case {record.label or k + 1}: {problem}")
            continue
        record.applies = applies
        if not applies:
            record.verdict, record.reason = "not_applicable", "Applies to does not match this run"
            continue
        if case.when is None:
            active = np.ones(n, dtype=bool)
        else:
            truth = ctx.truth(case.when.root, grid)
            active = truth.active()
            unknown |= ~np.asarray(truth.known, dtype=bool) & (claims < 0)
        overlap += active
        claims[(claims < 0) & active] = k
        usable.append(k)
    if not usable:
        outcome.na.append("no case applies to this run")
        return
    overlap_time = measure(t, overlap > 1, discrete=discrete) if len(usable) > 1 else 0.0
    if defaults.on_case_overlap == "error" and overlap_time > 0:
        outcome.error.append(f"cases overlap for {overlap_time:.3g} s")

    claimed = claims >= 0
    unknown &= ~claimed
    if is_assert:
        truth = ctx.truth(root, grid)
        known = np.asarray(truth.known, dtype=bool)
        x = None
        finite_row = known
        excess_rows = np.where(claimed & known, np.where(truth.val, -1.0, 1.0), np.nan)[:, None]
        upper = lower = None
        warn_rows = None
        side_rows = None
    else:
        values = np.asarray(ctx.series(root, grid), dtype=np.float64)
        x = values if values.ndim == 2 else values[:, None]
        finite_row = np.isfinite(x).all(axis=1)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        strict_u = np.zeros(n, dtype=bool)
        strict_l = np.zeros(n, dtype=bool)
        margin_u = np.full(n, np.nan)
        margin_l = np.full(n, np.nan)
        for k in usable:
            case = breq.cases[k]
            sel = claims == k
            if not sel.any():
                continue
            limit = case.limit
            if limit.upper is not None:
                upper[sel] = _side_values(limit.upper, ctx, grid)[sel]
                strict_u[sel] = not limit.upper.inclusive
            if limit.lower is not None:
                lower[sel] = _side_values(limit.lower, ctx, grid)[sel]
                strict_l[sel] = not limit.lower.inclusive
            if case.margin_abs is not None:
                margin_u[sel] = case.margin_abs
                margin_l[sel] = case.margin_abs
            elif case.margin_rel is not None:
                margin_u[sel] = case.margin_rel * np.abs(upper[sel])
                margin_l[sel] = case.margin_rel * np.abs(lower[sel])
        with np.errstate(invalid="ignore"):
            eu = x - upper[:, None]
            el = lower[:, None] - x
            eu = np.where(strict_u[:, None] & (eu == 0), TINY, eu)
            el = np.where(strict_l[:, None] & (el == 0), TINY, el)
            excess_rows = np.fmax(eu, el)
            warn_rows = np.fmax(eu + margin_u[:, None], el + margin_l[:, None])
            side_rows = np.where(np.isnan(el) | (eu >= el), 1, -1)  # which side the excess comes from
        excess_rows[~claimed] = np.nan
        warn_rows[~claimed] = np.nan

    tol_s = np.zeros(n)
    tol_n = np.zeros(n)  # samples a run may last and still be tolerated; 0 when no sample tolerance is given
    for k in usable:
        sel = claims == k
        tol_s[sel] = breq.cases[k].tol_seconds
        if breq.cases[k].tol_samples is not None:
            tol_n[sel] = breq.cases[k].tol_samples

    gap_time = _gap_time(t, claimed, claimed & ~finite_row) if not is_assert else 0.0
    unknown_time = _gap_time(t, claimed | unknown, unknown | (claimed & ~finite_row)) if is_assert else \
        measure(t, unknown, discrete=discrete)
    active_time = measure(t, claimed, discrete=discrete)
    usable_samples = claimed & finite_row
    result.totals.update({"active_time": active_time, "gap_time": gap_time, "unknown_time": unknown_time,
                          "overlap_time": overlap_time, "samples": int(np.count_nonzero(usable_samples))})
    for k in usable:
        result.cases[k].active_time = measure(t, claims == k, discrete=discrete)
    if not usable_samples.any():
        reason = "the condition never holds" if not claimed.any() else "no data while the condition holds"
        outcome.na.append(reason)
        for k in usable:
            result.cases[k].verdict, result.cases[k].reason = "not_applicable", reason
        result.trace = Trace(grid, None if x is None else x, None, None, claims, _intervals(t, claimed), discrete)
        return

    all_runs_list: list[dict] = []
    untolerated_total = 0.0
    tolerated_count = 0
    first_violation = math.inf
    fail_cases: set[int] = set()
    warn_cases: set[int] = set()
    worst_value = -math.inf
    worst_index = -1
    worst_component = 0
    for c in range(excess_rows.shape[1]):
        e = excess_rows[:, c]
        finite = np.isfinite(e)
        if finite.any():
            candidate = int(np.argmax(np.where(finite, e, -np.inf)))
            if e[candidate] > worst_value:
                worst_value, worst_index, worst_component = float(e[candidate]), candidate, c
        runs = all_runs(t, e, above=0.0)
        if runs.count:
            start, end = _run_bounds(runs, t, discrete)
            duration = end - start
            samples = runs.index_end - runs.index_start + 1
            run_tol_s = _per_run_min(tol_s, runs.index_start, runs.index_end)
            run_tol_n = _per_run_min(tol_n, runs.index_start, runs.index_end)
            tolerated = ((run_tol_s > 0) & (duration <= run_tol_s)) | (samples <= run_tol_n)
            for r in range(runs.count):
                owner = int(claims[runs.worst[r]])
                w = int(runs.worst[r])
                if not tolerated[r]:
                    fail_cases.add(owner)
                    untolerated_total += float(duration[r])
                    first_violation = min(first_violation, float(start[r]))
                else:
                    tolerated_count += 1
                record = {
                    "start": float(start[r]), "end": float(end[r]), "duration": float(duration[r]),
                    "samples": int(samples[r]), "worst_t": float(t[w]), "tolerated": bool(tolerated[r]),
                    "case": result.cases[owner].label if owner >= 0 else None, "component": c,
                    "flags": _flag_names(int(runs.flags[r])),
                }
                if x is not None:
                    record["worst_value"] = float(x[w, c])
                    side = side_rows[w, c]
                    record["limit"] = float(upper[w] if side > 0 else lower[w])
                    record["excess"] = float(e[w])
                all_runs_list.append(record)
        if warn_rows is not None:
            w_series = warn_rows[:, c]
            warns = all_runs(t, w_series, above=0.0)
            if warns.count:
                start, end = _run_bounds(warns, t, discrete)
                duration = end - start
                samples = warns.index_end - warns.index_start + 1
                run_tol_s = _per_run_min(tol_s, warns.index_start, warns.index_end)
                run_tol_n = _per_run_min(tol_n, warns.index_start, warns.index_end)
                tolerated = ((run_tol_s > 0) & (duration <= run_tol_s)) | (samples <= run_tol_n)
                for r in np.flatnonzero(~tolerated):
                    warn_cases.add(int(claims[warns.worst[r]]))

    all_runs_list.sort(key=lambda item: (item["start"], item["component"]))
    result.runs = all_runs_list[:EVIDENCE_LIMIT]
    result.totals.update({"runs": len(all_runs_list), "violating_time": untolerated_total,
                          "tolerated": tolerated_count})
    if math.isfinite(first_violation):
        result.first_violation = first_violation

    if worst_index >= 0:
        owner = int(claims[worst_index])
        result.case = result.cases[owner].label if owner >= 0 else None
        if is_assert:
            failing = [run for run in all_runs_list if not run["tolerated"]]
            result.at = failing[0]["start"] if failing else None
        else:
            side = side_rows[worst_index, worst_component]
            limit_value = float(upper[worst_index] if side > 0 else lower[worst_index])
            result.value = float(x[worst_index, worst_component])
            result.limit_value = limit_value
            result.at = float(t[worst_index])
            result.margin = -worst_value if worst_value != TINY else 0.0
            result.margin_pct = result.margin / abs(limit_value) if limit_value not in (0.0,) else None
            if x.shape[1] > 1:
                result.totals["component"] = worst_component
            chosen = breq.cases[owner].limit if owner >= 0 else None
            if chosen is not None:
                result.limit = _limit_text(chosen, side)
        # per case: its own closest approach
        for k in usable:
            sel = (claims == k)
            record = result.cases[k]
            if is_assert:
                record.verdict = "fail" if k in fail_cases else ("warn" if k in warn_cases else "pass")
                record.runs = sum(1 for run in all_runs_list if run["case"] == record.label and not run["tolerated"])
                continue
            e_case = np.where(sel[:, None], excess_rows, np.nan)
            if not np.isfinite(e_case).any():
                record.verdict, record.reason = "not_applicable", "no data while its condition holds"
                continue
            flat = int(np.nanargmax(e_case))
            i, c = divmod(flat, e_case.shape[1])
            side = side_rows[i, c]
            record.value = float(x[i, c])
            record.limit = float(upper[i] if side > 0 else lower[i])
            record.margin = -float(e_case[i, c]) if e_case[i, c] != TINY else 0.0
            record.margin_pct = record.margin / abs(record.limit) if record.limit != 0 else None
            record.at = float(t[i])
            record.runs = sum(1 for run in all_runs_list if run["case"] == record.label and not run["tolerated"])
            record.verdict = "fail" if k in fail_cases else ("warn" if k in warn_cases else "pass")

    if fail_cases:
        count = sum(1 for run in all_runs_list if not run["tolerated"])
        kind = "condition failed" if is_assert else "limit exceeded"
        outcome.fail.append(f"{kind} {count} time{'s' if count != 1 else ''} for {untolerated_total:.3g} s")
    if warn_cases and not fail_cases:
        outcome.warn.append("inside the warning margin")
    if tolerated_count:
        note = f"{tolerated_count} short violation{'s' if tolerated_count != 1 else ''} within the tolerance"
        if defaults.on_tolerated == "warn":
            outcome.warn.append(note)
        else:
            result.notes.append(note)
    for label, amount in (("data gap", gap_time), ("unknown condition", unknown_time)):
        if amount > 0:
            message = f"{label} of {amount:.3g} s inside the checked window"
            if defaults.on_gap == "warn":
                outcome.warn.append(message)
            elif defaults.on_gap == "fail":
                outcome.fail.append(message)
            else:
                result.notes.append(message)
    windows = _intervals(t, claimed)
    result.trace = Trace(grid, None if x is None else x, None if is_assert else upper,
                         None if is_assert else lower, claims, windows, discrete)


def _limit_text(limit: BoundLimit, side: int) -> str:
    """The side of the limit that applies at the worst point, with its comparison: "<= 70 kPa"."""
    if (side > 0 and limit.upper is not None) or limit.lower is None:
        chosen, op = limit.upper, "<"
    else:
        chosen, op = limit.lower, ">"
    if chosen is None:
        return limit.text
    return f"{op}{'=' if chosen.inclusive else ''} {_side_text(chosen)}"


def _limit_display(limit: BoundLimit) -> str:
    """A whole limit with its units: "<= 5.2 m", "[99 s, 102 s]", "== heavy"."""
    lower, upper = limit.lower, limit.upper
    if limit.kind in ("equal", "not_equal") and lower is not None:
        return f"{'==' if limit.kind == 'equal' else '!='} {_side_text(lower)}"
    if lower is not None and upper is not None:
        return (f"{'[' if lower.inclusive else '('}{_side_text(lower)}, {_side_text(upper)}"
                f"{']' if upper.inclusive else ')'}")
    if upper is not None:
        return f"<{'=' if upper.inclusive else ''} {_side_text(upper)}"
    if lower is not None:
        return f">{'=' if lower.inclusive else ''} {_side_text(lower)}"
    return limit.text


def _side_text(side) -> str:
    return side.text[2:] if side.text.startswith("= ") else side.text


# ------------------------------------------------------------------------------------------------------------
# single-number checks


def _compare(value: float, limit: BoundLimit, lower: float | None, upper: float | None,
             margin_abs: float | None, margin_rel: float | None) -> tuple[str, str | None, float | None,
                                                                          float | None, float | None]:
    """(verdict, reason, limit value, margin, margin fraction) of one number against one limit."""
    if limit.kind in ("equal", "not_equal"):
        target = lower
        equal = value == target
        ok = equal if limit.kind == "equal" else not equal
        shown = f"{_fmt(value)} instead of {_fmt(target)}" if limit.kind == "equal" else f"equal to {_fmt(target)}"
        return ("pass", None, target, None, None) if ok else ("fail", shown, target, None, None)
    candidates = []
    if upper is not None:
        e = value - upper
        if e == 0 and not limit.upper.inclusive:
            e = TINY
        m = margin_abs if margin_abs is not None else (margin_rel * abs(upper) if margin_rel is not None else None)
        candidates.append((e, upper, m, "above"))
    if lower is not None:
        e = lower - value
        if e == 0 and not limit.lower.inclusive:
            e = TINY
        m = margin_abs if margin_abs is not None else (margin_rel * abs(lower) if margin_rel is not None else None)
        candidates.append((e, lower, m, "below"))
    e, bound, m, side = max(candidates, key=lambda item: item[0])
    margin = -e if e != TINY else 0.0
    fraction = margin / abs(bound) if bound != 0 else None
    if e > 0:
        return "fail", f"{_fmt(value)} is {side} {_fmt(bound)}", bound, margin, fraction
    if m is not None and e + m > 0:
        return "warn", f"{_fmt(value)} is within the warning margin of {_fmt(bound)}", bound, margin, fraction
    return "pass", None, bound, margin, fraction


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _window(case: BoundCase, ctx: RunContext, grid) -> Tri | None:
    if case.when is None or grid is None:
        return None
    return ctx.truth(case.when.root, grid)


def _single(breq: BoundRequirement, ctx: RunContext, result: RequirementResult, outcome: _Outcome) -> None:
    names = breq.grid_names()
    grid = _grid(breq, ctx) if names else None
    for k, case in enumerate(breq.cases):
        record = result.cases[k]
        applies, problem = _applies(case, ctx)
        if problem is not None:
            record.verdict, record.reason = "error", problem
            outcome.error.append(problem)
            continue
        record.applies = applies
        if not applies:
            record.verdict, record.reason = "not_applicable", "Applies to does not match this run"
            continue
        window = _window(case, ctx, grid)
        if window is not None:
            record.active_time = measure(grid.t, window.active(), discrete=grid.discrete)
        if breq.kind == "duration":
            value, runs = _duration(breq, ctx, grid, window)
            record.runs = runs["count"]
            result.totals.update(runs)
        else:
            value = ctx.scalar(breq.check.root, grid, window)
            if isinstance(value, Tri):
                raise EvalError("the check is a condition, not a number")
        if isinstance(value, str):
            lower = _scalar_value_or_text(case.limit.lower, ctx, grid, window)
            verdict, reason, bound, margin, fraction = _compare_text(value, case.limit, lower)
        else:
            value = float(value)
            if math.isnan(value):
                record.verdict = "not_applicable"
                record.reason = "no data" if window is None or window.active().any() else "the condition never holds"
                continue
            lower = _scalar_side(case.limit.lower, ctx, grid, window) if case.limit.lower is not None else None
            upper = _scalar_side(case.limit.upper, ctx, grid, window) if case.limit.upper is not None else None
            verdict, reason, bound, margin, fraction = _compare(value, case.limit, lower, upper,
                                                                case.margin_abs, case.margin_rel)
        record.verdict, record.reason = verdict, reason
        record.value, record.limit, record.margin, record.margin_pct = value, bound, margin, fraction
        if verdict == "fail":
            outcome.fail.append(reason)
        elif verdict == "warn":
            outcome.warn.append(reason)
    _summarize_cases(result, outcome, breq)


def _scalar_value_or_text(side, ctx, grid, window):
    if side is None:
        return None
    return ctx.scalar(side.node, grid, window)


def _compare_text(value: str, limit: BoundLimit, target) -> tuple:
    if limit.kind not in ("equal", "not_equal"):
        raise EvalError(f"the value is the text {value!r}; compare it with == or !=")
    equal = str(target) == value
    ok = equal if limit.kind == "equal" else not equal
    return ("pass" if ok else "fail", None if ok else f"{value!r} {'is not' if limit.kind == 'equal' else 'is'} "
            f"{target!r}", None, None, None)


def _duration(breq: BoundRequirement, ctx: RunContext, grid, window: Tri | None) -> tuple[float, dict]:
    """Total time the check's condition holds inside the window, with exact crossing times for comparisons."""
    root = breq.check.root
    t = grid.t
    active = np.ones(grid.n, dtype=bool) if window is None else window.active()
    discrete = grid.discrete
    if root.op == "cmp" and root.value in ("<", "<=", ">", ">=") and root.args[0].type.dtype == "num" \
            and root.args[1].type.dtype == "num" and not discrete:
        left = ctx.series(root.args[0], grid)
        right = ctx.series(root.args[1], grid)
        with np.errstate(invalid="ignore"):
            e = left - right if root.value in (">", ">=") else right - left
            if root.value in (">=", "<="):
                e = np.where(e == 0, TINY, e)
    else:
        truth = ctx.truth(root, grid)
        e = np.where(truth.known, np.where(truth.val, 1.0, -1.0), np.nan)
    e = np.where(active, np.asarray(e, dtype=np.float64), np.nan)
    runs = all_runs(t, e, above=0.0)
    if runs.count == 0:
        return 0.0, {"count": 0, "longest": 0.0, "total": 0.0}
    start, end = _run_bounds(runs, t, discrete)
    durations = end - start
    total = float(np.sum(durations))
    return total, {"count": runs.count, "longest": float(durations.max()), "total": total}


def _event(breq: BoundRequirement, ctx: RunContext, result: RequirementResult, outcome: _Outcome) -> None:
    from .events import split_reference

    name, occurrence = split_reference(breq.event)
    event = ctx.event(name)
    result.events = [name]
    times = event.times
    result.totals.update({"count": event.count, "times": [float(v) for v in times[:EVIDENCE_LIMIT]],
                          "pending_at_end": event.pending})
    chosen = event.selected(occurrence)
    for k, case in enumerate(breq.cases):
        record = result.cases[k]
        applies, problem = _applies(case, ctx)
        if problem is not None:
            record.verdict, record.reason = "error", problem
            outcome.error.append(problem)
            continue
        record.applies = applies
        if not applies:
            record.verdict, record.reason = "not_applicable", "Applies to does not match this run"
            continue
        reasons: list[str] = []
        expected = case.count if case.count is not None else 1
        if event.count != expected:
            reasons.append(f"{name} happened {event.count} time{'s' if event.count != 1 else ''}, "
                           f"expected {expected}")
        record.runs = event.count
        verdict = "fail" if reasons else "pass"
        if chosen.shape[0]:
            record.value = float(chosen[0])
            record.at = float(chosen[0])
            if case.limit is not None:
                lower = _scalar_side(case.limit.lower, ctx, None, None) if case.limit.lower is not None else None
                upper = _scalar_side(case.limit.upper, ctx, None, None) if case.limit.upper is not None else None
                limit_verdict, reason, bound, margin, fraction = _compare(record.value, case.limit, lower, upper,
                                                                          case.margin_abs, case.margin_rel)
                record.limit, record.margin, record.margin_pct = bound, margin, fraction
                if limit_verdict == "fail":
                    reasons.append(f"{name} at {reason}")
                    verdict = "fail"
                elif limit_verdict == "warn" and verdict == "pass":
                    verdict = "warn"
                    reasons.append(f"{name} at {reason}")
        elif case.limit is not None:
            reasons.append(f"{name} did not happen")
            verdict = "fail"
        if verdict == "pass" and event.pending:
            verdict = "warn"
            reasons.append(f"{name} was still pending (not confirmed by its debounce) when the run ended")
        record.verdict = verdict
        record.reason = "; ".join(reasons) or None
        if verdict == "fail":
            outcome.fail.extend(reasons)
        elif verdict == "warn":
            outcome.warn.extend(reasons)
    _summarize_cases(result, outcome, breq)


def _summarize_cases(result: RequirementResult, outcome: _Outcome, breq: BoundRequirement) -> None:
    """The requirement's headline value: the case with the least margin (failing ones first)."""
    records = [c for c in result.cases if c.verdict in ("fail", "warn", "pass") and c.value is not None]
    if not records:
        if all(c.verdict == "not_applicable" for c in result.cases):
            outcome.na.append(result.cases[0].reason or "not applicable")
        return
    order = {"fail": 0, "warn": 1, "pass": 2}
    chosen = min(records, key=lambda c: (order[c.verdict], c.margin if c.margin is not None else math.inf))
    result.value = chosen.value
    result.limit_value = chosen.limit
    result.margin, result.margin_pct = chosen.margin, chosen.margin_pct
    result.at = chosen.at
    result.case = chosen.label or None
    case = breq.cases[result.cases.index(chosen)]
    if case.limit is not None:
        result.limit = _limit_display(case.limit)


# ------------------------------------------------------------------------------------------------------------
# entry point


def evaluate_requirement(breq: BoundRequirement, ctx: RunContext, *, with_context: bool = True
                         ) -> RequirementResult:
    req = breq.req
    result = RequirementResult(
        id=req.id, title=req.title, kind=breq.kind or (req.kind or ""), check=req.check, unit=breq.unit,
        loc=req.loc.text, rows=list(req.rows), passthrough=dict(req.passthrough),
        notes=list(dict.fromkeys(breq.notes)),
    )
    result.cases = [CaseResult(label=bc.case.label, verdict="pass", severity=bc.case.severity, row=bc.case.row)
                     for bc in breq.cases]
    severities = [bc.case.severity for bc in breq.cases if bc.case.severity]
    result.severity = severities[0] if severities else None
    if not breq.ok:
        result.verdict = "error"
        result.issues = breq.all_issues
        result.reason = "; ".join(dict.fromkeys(issue.message for issue in result.issues))
        for record, bc in zip(result.cases, breq.cases):
            if bc.issues:
                record.verdict, record.reason = "error", "; ".join(i.message for i in bc.issues)
        return result
    outcome = _Outcome()
    ctx.missing.clear()
    ctx.curve_outside.clear()
    try:
        if breq.kind in ("bound", "assert"):
            _series(breq, ctx, result, outcome)
        elif breq.kind == "event":
            _event(breq, ctx, result, outcome)
        else:
            _single(breq, ctx, result, outcome)
    except (EvalError, ExprError, UnitsError, ValueError) as exc:
        result.verdict = "error"
        result.reason = str(exc)
        result.issues = [Issue(path=req.id, message=str(exc), location=req.loc.text)]
        return result
    if ctx.missing:
        names = ", ".join(sorted(ctx.missing))
        message = f"event {names} did not happen" if len(ctx.missing) == 1 else f"events {names} did not happen"
        rule = ctx.config.defaults.on_missing_event
        if rule == "fail":
            outcome.fail.append(message)
        elif rule == "warn":
            if not outcome.fail:
                outcome.warn.append(message)
        else:
            result.notes.append(message)
        result.events = sorted(set(result.events) | ctx.missing)
    for name, count in ctx.curve_outside.items():
        result.notes.append(f"curve {name} was used outside its range at {count} samples")
    result.verdict, result.reason = outcome.verdict()
    if breq.kind in ("bound", "assert"):
        case_verdicts = [c.verdict for c in result.cases]
        if result.verdict == "pass" and case_verdicts and worst(case_verdicts) == "warn":
            result.verdict = "warn"
    if with_context and result.verdict in ("fail", "warn") and result.at is not None:
        result.context = _context(ctx, result.at)
    return result


def evaluate_all(bound: Sequence[BoundRequirement], ctx: RunContext) -> list[RequirementResult]:
    return [evaluate_requirement(item, ctx) for item in bound]
