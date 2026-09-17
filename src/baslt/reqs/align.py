"""Time grids for requirement checks: which clock an expression is evaluated on, and signals resampled onto it.

A requirement is evaluated on the clock of the first signal its check names (then its condition, then its
cases'), or on the union of all its clocks with `grid: union`. Other signals are resampled onto that grid with the
reconstruction of docs/contracts.md: linear for continuous and vector signals, previous value for discrete ones,
NaN outside a signal's time span and across its non-finite samples. Signals that share a clock are not resampled.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..verify.reference import reconstruct_hold, reconstruct_linear

__all__ = ["Grid", "Signals"]


@dataclass(frozen=True)
class Grid:
    key: tuple
    t: np.ndarray
    discrete: bool
    ref: str | None  # the signal whose clock this is (None for a union grid)

    @property
    def n(self) -> int:
        return int(self.t.shape[0])


class Signals:
    """The loaded signals of one run, their clocks and a byte-bounded cache of resampled values."""

    def __init__(self, signals: Mapping, *, discrete: Mapping[str, bool] | None = None,
                 cache_bytes: int = 512 << 20) -> None:
        self.signals = dict(signals)
        self.discrete = {name: bool((discrete or {}).get(name, sig.kind == "discrete"))
                         for name, sig in self.signals.items()}
        self._clock: dict[str, int] = {}
        self._clocks: list[np.ndarray] = []
        self._values: dict[str, np.ndarray] = {}
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self.cache_limit = cache_bytes

    # ----- clocks -----------------------------------------------------------------------------------------

    def clock_id(self, name: str) -> int:
        """The same number for every signal whose timestamps are identical."""
        found = self._clock.get(name)
        if found is not None:
            return found
        t = self.signals[name].t
        for k, other in enumerate(self._clocks):
            if other is t or (other.shape == t.shape and (t.size == 0 or (other[0] == t[0] and other[-1] == t[-1]))
                              and np.array_equal(other, t)):
                self._clock[name] = k
                return k
        self._clocks.append(t)
        self._clock[name] = len(self._clocks) - 1
        return self._clock[name]

    def grid(self, name: str) -> Grid:
        return Grid(key=("clock", self.clock_id(name)), t=self.signals[name].t, discrete=self.discrete[name],
                    ref=name)

    def union(self, names: Sequence[str]) -> Grid:
        ids = sorted({self.clock_id(name) for name in names})
        if len(ids) == 1:
            return self.grid(names[0])
        t = np.unique(np.concatenate([self._clocks[k] for k in ids]))
        discrete = all(self.discrete[name] for name in names)
        return Grid(key=("union", tuple(ids)), t=t, discrete=discrete, ref=None)

    def default_grid(self) -> Grid:
        """For expressions that name no signal: the clock of the first signal of the run."""
        if not self.signals:
            raise LookupError("the run has no signals")
        return self.grid(next(iter(self.signals)))

    # ----- values -----------------------------------------------------------------------------------------

    def values(self, name: str) -> np.ndarray:
        found = self._values.get(name)
        if found is None:
            v = self.signals[name].v
            found = v if v.dtype == np.float64 else v.astype(np.float64)
            self._values[name] = found
        return found

    def on(self, name: str, grid: Grid) -> np.ndarray:
        """The signal's values at the grid's timestamps."""
        if grid.key == ("clock", self.clock_id(name)):
            return self.values(name)
        key = (name, grid.key)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        sig = self.signals[name]
        values = self.values(name)
        if self.discrete[name]:
            out = np.asarray(reconstruct_hold(sig.t, values, grid.t), dtype=np.float64)
        else:
            out = np.asarray(reconstruct_linear(sig.t, values, grid.t), dtype=np.float64)
        self.remember(key, out)
        return out

    def remember(self, key: tuple, value: np.ndarray) -> None:
        size = int(getattr(value, "nbytes", 0))
        if size > self.cache_limit:
            return
        self._cache[key] = value
        self._cache_bytes += size
        while self._cache_bytes > self.cache_limit and self._cache:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= int(getattr(old, "nbytes", 0))

    def cached(self, key: tuple):
        found = self._cache.get(key)
        if found is not None:
            self._cache.move_to_end(key)
        return found
