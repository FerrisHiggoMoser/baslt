"""Named events of a run: when they happen, found once per run.

    signal + falls_below / rises_above V   threshold crossings (hysteresis, debounce), crossing times interpolated
    signal + equals V                      the first sample of every run where the state equals V
    when: <condition>                      every time the condition becomes true; a comparison gives the exact
                                           crossing time, any other condition the midpoint between samples
    param: <name>                          a time given by the run's parameters

Every trigger counts toward `count(E)`; the event's `occurrence` (first by default) picks the time the condition
helpers use, and `E#last`, `E#2`, `E#all` override it where the event is named.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..ops import threshold_crossing
from ..ops._common import run_starts
from .expr import ExprError
from .units_ext import UnitsError

__all__ = ["EventTimes", "detect_event", "split_reference"]


@dataclass
class EventTimes:
    name: str
    times: np.ndarray  # every trigger, ascending
    pending: bool = False  # a last change that the debounce had not confirmed when the run ended
    origin: str = ""
    occurrence: str = "first"

    @property
    def count(self) -> int:
        return int(self.times.shape[0])

    def selected(self, occurrence: str | None = None) -> np.ndarray:
        """The trigger times an occurrence picks: one time, all of them, or none."""
        which = (occurrence or self.occurrence or "first").lower()
        if self.count == 0:
            return self.times
        if which == "first":
            return self.times[:1]
        if which == "last":
            return self.times[-1:]
        if which == "all":
            return self.times
        k = int(which)
        return self.times[k - 1:k] if 1 <= k <= self.count else self.times[:0]

    def time(self, occurrence: str | None = None) -> float:
        chosen = self.selected(occurrence)
        return float(chosen[0]) if chosen.shape[0] else math.inf


def split_reference(text: str) -> tuple[str, str | None]:
    name, _, occurrence = text.partition("#")
    return name.strip(), (occurrence.strip().lower() or None)


def _quantity(ctx, text, unit: str | None, *, delta: bool) -> float:
    q = ctx.config.units.parse(text)
    try:
        return ctx.config.units.convert(q, unit, delta=delta)
    except UnitsError as exc:
        raise ExprError(str(exc)) from None


def _seconds(ctx, text) -> float:
    if text is None:
        return 0.0
    q = ctx.config.units.parse(text)
    if q.unit is None:
        return q.value
    return q.value * ctx.config.units.factor(q.unit, "s")


def detect_event(defn, ctx) -> EventTimes:
    """The triggers of one event definition on the run in `ctx` (a RunContext)."""
    origin = defn.loc or f"events.{defn.name}"
    debounce = _seconds(ctx, defn.debounce)
    if defn.param is not None:
        value = ctx.params.get(defn.param)
        times = np.empty(0)
        if value not in (None, ""):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ExprError(f"event {defn.name!r}: run parameter {defn.param!r} is {value!r}, not a time") from None
            unit = ctx.config.params.units.get(defn.param)
            if unit:
                number *= ctx.config.units.factor(unit, "s")
            if math.isfinite(number):
                times = np.array([number])
        return EventTimes(defn.name, times, False, origin, defn.occurrence)

    if defn.condition is not None:
        bound = ctx.binder.bind(defn.signal, role="event")
        node = bound.root
        if not node.type.series or node.type.components != 1:
            raise ExprError(f"event {defn.name!r}: {defn.signal!r} must be a single-valued signal")
        grid = ctx.grid_for(bound.signals)
        x = ctx.series(node, grid)
        if defn.condition == "equals":
            code = ctx.code_of(node, defn.value)
            starts = run_starts(x == code)
            return EventTimes(defn.name, grid.t[starts].astype(np.float64), False, origin, defn.occurrence)
        unit = node.type.unit
        level = _quantity(ctx, defn.value, unit, delta=False)
        hysteresis = _quantity(ctx, defn.hysteresis, unit, delta=True) if defn.hysteresis else 0.0
        edge = "falling" if defn.condition == "falls_below" else "rising"
        found = threshold_crossing.detect(grid.t, x, level, edge=edge, hysteresis=hysteresis, debounce=debounce)
        return EventTimes(defn.name, np.asarray(found.t, dtype=np.float64), bool(found.pending_at_end), origin,
                          defn.occurrence)

    bound = ctx.binder.bind(defn.when, role="event")
    node = bound.root
    if node.type.dtype != "bool":
        raise ExprError(f"event {defn.name!r}: {defn.when!r} is not a condition")
    grid = ctx.grid_for(bound.signals)
    if node.op == "cmp" and node.value in ("<", "<=", ">", ">=") and node.args[0].type.dtype == "num" \
            and node.args[1].type.dtype == "num" and node.type.series:
        left = ctx.series(node.args[0], grid)
        right = ctx.series(node.args[1], grid)
        difference = np.broadcast_to(np.asarray(left - right, dtype=np.float64), grid.t.shape)
        edge = "rising" if node.value in (">", ">=") else "falling"
        hysteresis = 0.0
        if defn.hysteresis:
            hysteresis = _quantity(ctx, defn.hysteresis, node.args[0].type.unit, delta=True)
        found = threshold_crossing.detect(grid.t, np.ascontiguousarray(difference), 0.0, edge=edge,
                                          hysteresis=hysteresis, debounce=debounce)
        return EventTimes(defn.name, np.asarray(found.t, dtype=np.float64), bool(found.pending_at_end), origin,
                          defn.occurrence)
    truth = ctx.truth(node, grid)
    indicator = np.where(truth.known, np.where(truth.val, 1.0, -1.0), np.nan)
    indicator = np.broadcast_to(indicator, grid.t.shape)
    found = threshold_crossing.detect(grid.t, np.ascontiguousarray(indicator), 0.0, edge="rising",
                                      debounce=debounce)
    return EventTimes(defn.name, np.asarray(found.t, dtype=np.float64), bool(found.pending_at_end), origin,
                      defn.occurrence)
