"""Evaluating bound expressions on one run: series on a time grid, three-valued conditions, and single numbers.

Values are float64 arrays (states as their codes), conditions are `Tri` (true / false / unknown per sample; a
sample is unknown where a value it depends on is not finite), and single numbers are floats. Scalars broadcast
against series. Event times are found once per run; resampled signals and evaluated sub-expressions are cached by
grid within a byte budget.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..policy import SignalIndex
from .align import Grid, Signals
from .events import EventTimes, detect_event, split_reference
from .expr import B, Binder, BoundExpr, ExprError

__all__ = ["EvalError", "RunContext", "Tri", "measure", "time_weighted"]


class EvalError(ExprError):
    """An expression that cannot be evaluated on this run (a missing parameter, a curve left, ...)."""


class Tri:
    """A condition per sample: `val` where `known`, unknown elsewhere."""

    __slots__ = ("known", "val")

    def __init__(self, val, known) -> None:
        self.val = val
        self.known = known

    @property
    def nbytes(self) -> int:
        return int(getattr(self.val, "nbytes", 0)) + int(getattr(self.known, "nbytes", 0))

    def active(self) -> np.ndarray:
        return np.asarray(self.val & self.known, dtype=bool)

    def broadcast(self, n: int) -> Tri:
        return Tri(np.broadcast_to(np.asarray(self.val, dtype=bool), (n,)),
                   np.broadcast_to(np.asarray(self.known, dtype=bool), (n,)))


def measure(t: np.ndarray, mask: np.ndarray, *, discrete: bool) -> float:
    """Time covered by `mask`: intervals between consecutive samples that are both in the mask (continuous), or
    each sample's interval up to the next sample (discrete, held values)."""
    if t.shape[0] < 2:
        return 0.0
    dt = np.diff(t)
    mask = np.asarray(mask, dtype=bool)
    covered = mask[:-1] if discrete else (mask[:-1] & mask[1:])
    return float(np.sum(dt[covered]))


def time_weighted(t: np.ndarray, x: np.ndarray, mask: np.ndarray, *, discrete: bool, what: str) -> tuple[float, float]:
    """(integral, time) of `x` over the samples in `mask`: trapezoids (continuous) or held values (discrete)."""
    if t.shape[0] < 2:
        return 0.0, 0.0
    dt = np.diff(t)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(x)
    if discrete:
        pair = ok[:-1]
        heights = x[:-1]
    else:
        pair = ok[:-1] & ok[1:]
        heights = 0.5 * (x[:-1] + x[1:]) if what != "square" else (x[:-1] ** 2 + x[1:] ** 2 + x[:-1] * x[1:]) / 3
    if discrete and what == "square":
        heights = x[:-1] ** 2
    with np.errstate(invalid="ignore", over="ignore"):
        area = float(np.sum(dt[pair] * heights[pair]))
    return area, float(np.sum(dt[pair]))


def _tri_not(a: Tri) -> Tri:
    return Tri(~a.val, a.known)


def _tri_and(a: Tri, b: Tri) -> Tri:
    val = a.val & b.val
    known = (a.known & b.known) | (a.known & ~a.val) | (b.known & ~b.val)
    return Tri(val & known, known)


def _tri_or(a: Tri, b: Tri) -> Tri:
    val = (a.val & a.known) | (b.val & b.known)
    known = (a.known & b.known) | (a.known & a.val) | (b.known & b.val)
    return Tri(val, known)


class RunContext:
    """One run, ready for requirement checks."""

    def __init__(self, run, config, *, params: Mapping[str, object] | None = None,
                 cache_bytes: int = 512 << 20) -> None:
        self.run = run
        self.config = config
        self.params = dict(params or {})
        infos = {name: sig.info() for name, sig in run.signals.items()}
        self.index = SignalIndex(list(infos.values()))
        discrete = {}
        for alias in config.signals.values():
            if alias.path is not None and alias.kind:
                try:
                    discrete[self.index.lookup(alias.path)] = alias.kind == "discrete"
                except LookupError:
                    pass
        self.signals = Signals(run.signals, discrete=discrete, cache_bytes=cache_bytes)
        labels = {name: list(sig.labels) for name, sig in run.signals.items() if sig.labels}
        self.binder = Binder(self.index, infos, config, param_names=None, param_units=config.params.units,
                             labels=labels)
        self._events: dict[str, EventTimes] = {}
        self.missing: set[str] = set()  # events a condition needed but that did not happen
        self.curve_outside: dict[str, int] = {}

    # ----- grids ------------------------------------------------------------------------------------------

    def grid_for(self, names: Sequence[str], mode: str | None = None) -> Grid:
        if not names:
            return self.signals.default_grid()
        if mode == "union":
            return self.signals.union(list(names))
        if mode and mode not in ("first", "union"):
            bound = self.binder.bind(mode, role="check")
            if not bound.signals:
                raise EvalError(f"grid {mode!r} names no signal")
            return self.signals.grid(bound.signals[0])
        return self.signals.grid(names[0])

    # ----- events -----------------------------------------------------------------------------------------

    def event(self, name: str) -> EventTimes:
        found = self._events.get(name)
        if found is None:
            defn = self.config.events.get(name)
            if defn is None:
                raise EvalError(f"unknown event {name!r}")
            try:
                found = detect_event(defn, self)
            except ExprError as exc:
                raise EvalError(f"event {name!r}: {exc}") from None
            self._events[name] = found
        return found

    def event_times(self, reference: str) -> np.ndarray:
        name, occurrence = split_reference(reference)
        times = self.event(name).selected(occurrence)
        if times.shape[0] == 0:
            self.missing.add(name)
        return times

    # ----- labels -----------------------------------------------------------------------------------------

    def labels_of(self, node: B) -> list[str] | dict[int, str] | None:
        while node.op in ("alias", "unit", "scale", "comp", "at", "cond") and node.args:
            node = node.args[0]
        if node.op != "sig":
            return None
        name, alias_name = node.value
        alias = self.config.signals.get(alias_name) if alias_name else None
        if alias is not None and alias.labels:
            return alias.labels
        labels = self.run.signals[name].labels
        return list(labels) if labels else None

    def code_of(self, node: B, label) -> float:
        """The numeric code of a state label (or a number given as a label)."""
        labels = self.labels_of(node)
        text = str(label).strip()
        if isinstance(labels, dict):
            for code, name in labels.items():
                if name == text:
                    return float(code)
        elif labels:
            if text in labels:
                return float(labels.index(text))
        try:
            return float(text)
        except ValueError:
            known = sorted(labels.values()) if isinstance(labels, dict) else list(labels or [])
            hint = f"; labels are {', '.join(map(str, known))}" if known else "; the signal has no labels"
            raise EvalError(f"{text!r} is not a state of this signal{hint}") from None

    # ----- evaluation -------------------------------------------------------------------------------------

    def series(self, node: B, grid: Grid) -> np.ndarray:
        value = self.value(node, grid)
        if isinstance(value, Tri):
            raise EvalError("a condition was used where a number is needed")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return np.broadcast_to(array, (grid.n,))
        return array

    def truth(self, node: B, grid: Grid) -> Tri:
        value = self.value(node, grid)
        if isinstance(value, Tri):
            return value.broadcast(grid.n) if np.ndim(value.val) == 0 else value
        raise EvalError("a number was used where a condition is needed")

    def scalar(self, node: B, grid: Grid | None, window: Tri | None = None) -> object:
        """A single value. Aggregates without a grid use the clock of the signal they aggregate."""
        return self.value(node, grid, window)

    def value(self, node: B, grid: Grid | None, window: Tri | None = None):
        cacheable = node.type.series and grid is not None and window is None and node.op not in (
            "num", "qty", "str", "label", "bool", "sig", "tvar")
        if cacheable:
            key = (node.key, grid.key)
            found = self.signals.cached(key)
            if found is not None:
                return found
        result = self._eval(node, grid, window)
        if cacheable:
            self.signals.remember(key, result)
        return result

    def _eval(self, node: B, grid: Grid | None, window: Tri | None):
        op = node.op
        if op == "num":
            return float(node.value)
        if op == "qty":
            return float(node.value[0])
        if op in ("str", "label"):
            return node.value
        if op == "bool":
            return Tri(np.bool_(node.value), np.bool_(True))
        if op == "tvar":
            return grid.t
        if op == "sig":
            return self.signals.on(node.value[0], grid)
        if op == "stack":
            return np.column_stack([self._num(arg, grid, window) for arg in node.args])
        if op == "param":
            if node.value not in self.params or self.params[node.value] in (None, ""):
                raise EvalError(f"run parameter {node.value!r} is not given for this run")
            value = self.params[node.value]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            try:
                return float(str(value).strip())
            except ValueError:
                return str(value).strip()
        if op in ("alias", "unit", "cond"):
            return self.value(node.args[0], grid, window)
        if op == "scale":
            return self._num(node.args[0], grid, window) * node.value
        if op == "neg":
            return -self._num(node.args[0], grid, window)
        if op in ("add", "sub", "mul", "div", "pow"):
            a = self._num(node.args[0], grid, window)
            b = self._num(node.args[1], grid, window)
            if np.ndim(a) == 2 and np.ndim(b) == 1:
                b = b[:, None]
            elif np.ndim(b) == 2 and np.ndim(a) == 1:
                a = a[:, None]
            with np.errstate(all="ignore"):
                if op == "add":
                    return a + b
                if op == "sub":
                    return a - b
                if op == "mul":
                    return a * b
                if op == "div":
                    return np.divide(a, b)
                return np.power(a, b)
        if op == "not":
            return _tri_not(self._tri(node.args[0], grid, window))
        if op in ("and", "or"):
            parts = [self._tri(arg, grid, window) for arg in node.args]
            result = parts[0]
            for part in parts[1:]:
                result = _tri_and(result, part) if op == "and" else _tri_or(result, part)
            return result
        if op == "cmp":
            return self._compare(node, grid, window)
        if op in ("in", "notin"):
            subject = node.args[0]
            values = self._num(subject, grid, window) if subject.type.dtype != "any" else \
                self.value(subject, grid, window)
            wanted = []
            for item in node.args[1:]:
                if item.op == "label":
                    wanted.append(self.code_of(subject, item.value) if subject.type.dtype != "any" else item.value)
                else:
                    wanted.append(self.value(item, grid, window))
            if isinstance(values, str):
                hit = values in [str(w) for w in wanted]
                return Tri(np.bool_(hit if node.op == "in" else not hit), np.bool_(True))
            arr = np.asarray(values, dtype=np.float64)
            hit = np.isin(arr, np.asarray(wanted, dtype=np.float64))
            known = np.isfinite(arr)
            return Tri(hit if node.op == "in" else ~hit, known)
        if op == "where":
            test = self._tri(node.args[0], grid, window)
            if node.type.dtype == "bool":
                a = self._tri(node.args[1], grid, window)
                b = self._tri(node.args[2], grid, window)
                return Tri(np.where(test.val, a.val, b.val), test.known & np.where(test.val, a.known, b.known))
            a = self._num(node.args[1], grid, window)
            b = self._num(node.args[2], grid, window)
            cond = test.val if np.ndim(a) < 2 and np.ndim(b) < 2 else np.asarray(test.val)[:, None]
            out = np.where(cond, a, b)
            return np.where(test.known if np.ndim(out) < 2 else np.asarray(test.known)[:, None], out, np.nan)
        if op == "comp":
            values = self._num(node.args[0], grid, window)
            return values[:, node.value] if np.ndim(values) == 2 else values
        if op in ("abs", "sign", "sqrt", "exp", "log", "sin", "cos", "tan"):
            x = self._num(node.args[0], grid, window)
            with np.errstate(all="ignore"):
                return getattr(np, op)(x)
        if op in ("min", "max"):
            values = [self._num(arg, grid, window) for arg in node.args]
            fn = np.fmin if op == "min" else np.fmax
            result = values[0]
            for other in values[1:]:
                result = fn(result, other)
            return result
        if op == "clip":
            x, lo, hi = (self._num(arg, grid, window) for arg in node.args)
            return np.clip(x, lo, hi)
        if op == "hypot":
            return np.hypot(self._num(node.args[0], grid, window), self._num(node.args[1], grid, window))
        if op == "norm":
            x = self._num(node.args[0], grid, window)
            return np.sqrt(np.sum(np.square(x), axis=1)) if np.ndim(x) == 2 else np.abs(x)
        if op == "deriv":
            return self._deriv(self._num(node.args[0], grid, window), grid)
        if op == "movmean":
            return self._movmean(self._num(node.args[0], grid, window), grid, float(node.value))
        if op == "curve":
            return self._curve(node, grid, window)
        if op in ("after", "before", "during"):
            name, delay = node.value
            times = self.event_times(name)
            t = grid.t
            if times.shape[0] == 0:
                val = np.full(t.shape, op == "before")
            elif op == "after":
                val = t >= times[0] + delay
            elif op == "before":
                val = t < times[0] + delay
            else:
                val = (t >= times[0]) & (t < times[0] + delay)
            if op != "before" and times.shape[0] > 1:  # an "E#all" reference: any of its triggers
                for tau in times[1:]:
                    val = val | ((t >= tau + delay) if op == "after" else ((t >= tau) & (t < tau + delay)))
            return Tri(val, np.ones(t.shape, dtype=bool))
        if op == "between":
            first, second = node.value
            starts = self.event_times(first)
            name, _ = split_reference(second)
            ends = self.event(name).times
            t = grid.t
            val = np.zeros(t.shape, dtype=bool)
            for tau in starts:
                later = ends[ends >= tau]
                if later.shape[0] == 0:
                    self.missing.add(name)
                    val |= t >= tau
                else:
                    val |= (t >= tau) & (t < later[0])
            return Tri(val, np.ones(t.shape, dtype=bool))
        if op == "since":
            times = self.event_times(node.value)
            if times.shape[0] == 0:
                return np.full(grid.n, np.nan)
            out = grid.t - times[0]
            return np.where(out >= 0, out, np.nan)
        if op == "time":
            times = self.event_times(node.value)
            return float(times[0]) if times.shape[0] else math.nan
        if op == "count":
            name, occurrence = split_reference(node.value)
            return float(self.event(name).count)
        if op == "at":
            return self._at(node)
        if op == "agg":
            return self._aggregate(node, grid, window)
        raise EvalError(f"cannot evaluate {op!r}")

    def _num(self, node: B, grid: Grid | None, window: Tri | None):
        value = self.value(node, grid, window)
        if isinstance(value, Tri):
            raise EvalError("a condition was used where a number is needed")
        if isinstance(value, str):
            raise EvalError(f"{value!r} is a text where a number is needed")
        return value

    def _tri(self, node: B, grid: Grid | None, window: Tri | None) -> Tri:
        value = self.value(node, grid, window)
        if not isinstance(value, Tri):
            raise EvalError("a number was used where a condition is needed")
        return value

    def _compare(self, node: B, grid: Grid | None, window: Tri | None) -> Tri:
        left_node, right_node = node.args
        symbol = node.value
        if left_node.type.dtype == "bool" and right_node.type.dtype == "bool":
            a = self._tri(left_node, grid, window)
            b = self._tri(right_node, grid, window)
            equal = a.val == b.val
            return Tri(equal if symbol == "==" else ~equal, a.known & b.known)
        a = self._side(left_node, right_node, grid, window)
        b = self._side(right_node, left_node, grid, window)
        if isinstance(a, str) or isinstance(b, str):
            if not (isinstance(a, str) and isinstance(b, str)):
                if isinstance(a, str) and _is_number(a):
                    a = float(a)
                elif isinstance(b, str) and _is_number(b):
                    b = float(b)
                else:
                    raise EvalError(f"cannot compare {a!r} with {b!r}")
            if isinstance(a, str):
                equal = a == b
                result = equal if symbol == "==" else (not equal if symbol == "!=" else None)
                if result is None:
                    raise EvalError(f"texts can only be compared with == or !=, not {symbol}")
                return Tri(np.bool_(result), np.bool_(True))
        with np.errstate(invalid="ignore"):
            a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            val = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
                   "==": np.equal, "!=": np.not_equal}[symbol](a_arr, b_arr)
        known = np.isfinite(a_arr) & np.isfinite(b_arr)
        return Tri(val & known, known)

    def _side(self, node: B, other: B, grid, window):
        if node.op == "label":
            if other.type.dtype == "any":
                return node.value
            return self.code_of(other, node.value)
        return self.value(node, grid, window)

    def _deriv(self, x: np.ndarray, grid: Grid) -> np.ndarray:
        t = grid.t
        if t.shape[0] < 2:
            return np.full(np.shape(x), np.nan)
        x = np.broadcast_to(x, (t.shape[0],) + np.shape(x)[1:])
        with np.errstate(all="ignore"):
            out = np.gradient(x, t, axis=0)
        out = np.asarray(out, dtype=np.float64)
        out[~np.isfinite(out)] = np.nan
        return out

    def _movmean(self, x: np.ndarray, grid: Grid, window: float) -> np.ndarray:
        t = grid.t
        x = np.asarray(np.broadcast_to(x, (t.shape[0],) + np.shape(x)[1:]), dtype=np.float64)
        columns = x.reshape(t.shape[0], -1)
        out = np.empty_like(columns)
        dt = np.diff(t)
        start = t - window
        for k in range(columns.shape[1]):
            col = columns[:, k]
            ok = np.isfinite(col[:-1]) & np.isfinite(col[1:])
            with np.errstate(invalid="ignore"):
                area = np.where(ok, 0.5 * (col[:-1] + col[1:]) * dt, 0.0)
            span = np.where(ok, dt, 0.0)
            integral = np.concatenate(([0.0], np.cumsum(area)))
            covered = np.concatenate(([0.0], np.cumsum(span)))
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = (integral - np.interp(start, t, integral)) / (covered - np.interp(start, t, covered))
            out[:, k] = np.where(np.isfinite(mean), mean, col)
        return out.reshape(x.shape)

    def _curve(self, node: B, grid: Grid, window: Tri | None) -> np.ndarray:
        curve = self.config.curves[node.value]
        arg = self._num(node.args[0], grid, window)
        xs = np.array([p[0] for p in curve.points])
        ys = np.array([p[1] for p in curve.points])
        arr = np.asarray(arg, dtype=np.float64)
        if curve.mode == "step":
            index = np.clip(np.searchsorted(xs, arr, side="right") - 1, 0, xs.shape[0] - 1)
            out = ys[index]
        else:
            out = np.interp(arr, xs, ys)
        outside = (arr < xs[0]) | (arr > xs[-1])
        count = int(np.count_nonzero(outside & np.isfinite(arr)))
        if count:
            self.curve_outside[curve.name] = self.curve_outside.get(curve.name, 0) + count
            if curve.outside == "none":
                out = np.where(outside, np.nan, out)
            elif curve.outside == "error":
                raise EvalError(f"curve {curve.name!r} is used outside its range ({xs[0]:g} .. {xs[-1]:g})")
        return np.where(np.isfinite(arr), out, np.nan)

    def _at(self, node: B) -> float:
        inner = node.args[0]
        times = self.event_times(node.value)
        if times.shape[0] == 0:
            return math.nan
        names = _signals_of(inner)
        grid = self.grid_for(names)
        values = self.series(inner, grid)
        tau = float(times[0])
        t = grid.t
        if t.shape[0] == 0 or tau < t[0] or tau > t[-1]:
            return math.nan
        finite = np.isfinite(values)
        if inner.type.discrete:
            k = int(np.searchsorted(t, tau, side="right")) - 1
            return float(values[max(k, 0)])
        if not finite.any():
            return math.nan
        from ..verify.reference import reconstruct_linear

        return float(reconstruct_linear(t, values, np.array([tau]))[0])

    def aggregate_time(self, node: B, grid: Grid | None, window: Tri | None) -> float | None:
        """The time of the sample a max, min, initial or final aggregate took its value from."""
        fn = node.value
        inner = node.args[0]
        if grid is None:
            grid = self.grid_for(_signals_of(inner))
        active = np.ones(grid.n, dtype=bool) if window is None else window.broadcast(grid.n).active()
        x = np.asarray(self.series(inner, grid), dtype=np.float64)
        if x.ndim != 1:
            return None
        ok = active & np.isfinite(x)
        if not ok.any():
            return None
        if fn in ("max", "min"):
            picked = np.where(ok, x, -np.inf if fn == "max" else np.inf)
            index = int(np.argmax(picked) if fn == "max" else np.argmin(picked))
        else:
            hits = np.flatnonzero(ok)
            index = int(hits[0] if fn == "initial" else hits[-1])
        return float(grid.t[index])

    def _aggregate(self, node: B, grid: Grid | None, window: Tri | None) -> float:
        fn = node.value
        inner = node.args[0]
        if grid is None:
            grid = self.grid_for(_signals_of(inner))
        active = np.ones(grid.n, dtype=bool) if window is None else window.broadcast(grid.n).active()
        if fn == "duration":
            truth = self.truth(inner, grid)
            return measure(grid.t, active & truth.active(), discrete=grid.discrete)
        x = np.asarray(self.series(inner, grid), dtype=np.float64)
        ok = active & np.isfinite(x)
        if not ok.any():
            return math.nan
        if fn in ("max", "min"):
            picked = np.where(ok, x, -np.inf if fn == "max" else np.inf)
            return float(picked[np.argmax(picked) if fn == "max" else np.argmin(picked)])
        if fn == "initial":
            return float(x[np.flatnonzero(ok)[0]])
        if fn == "final":
            return float(x[np.flatnonzero(ok)[-1]])
        discrete = grid.discrete or inner.type.discrete
        if fn == "integral":
            return time_weighted(grid.t, x, active, discrete=discrete, what="value")[0]
        area, span = time_weighted(grid.t, x, active, discrete=discrete, what="square" if fn == "rms" else "value")
        if span <= 0:
            values = x[ok]
            return float(np.sqrt(np.mean(values ** 2)) if fn == "rms" else np.mean(values))
        return float(math.sqrt(area / span) if fn == "rms" else area / span)

    # ----- binding helpers --------------------------------------------------------------------------------

    def bind(self, text: str, role: str = "check") -> BoundExpr:
        return self.binder.bind(text, role=role)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _signals_of(node: B) -> list[str]:
    out: list[str] = []

    def walk(n: B) -> None:
        if n.op == "sig" and n.value[0] not in out:
            out.append(n.value[0])
        for arg in n.args:
            if isinstance(arg, B):
                walk(arg)

    walk(node)
    return out

