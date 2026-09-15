"""Sets of retained sample indices with role bitmasks.

A SampleSet stores disjoint half-open index ranges [start, stop), each with a uint64 role mask
(bit k = role legend entry k). A long event window costs one range until the set is materialized
for encoding, and unions are computed on range boundaries rather than on samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _np():
    import numpy as np

    return np


@dataclass(slots=True)
class SampleSet:
    starts: np.ndarray  # int64, sorted, disjoint
    stops: np.ndarray  # int64, starts < stops
    roles: np.ndarray  # uint64, non-zero

    # ------------------------------------------------------------------ construction

    @classmethod
    def empty(cls) -> SampleSet:
        np = _np()
        return cls(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.uint64))

    @classmethod
    def from_points(cls, idx, bit: int) -> SampleSet:
        """Single indices, all tagged with role `bit`. Duplicates are allowed."""
        np = _np()
        idx = np.unique(np.asarray(idx, dtype=np.int64).reshape(-1))
        mask = np.full(idx.shape[0], np.uint64(1) << np.uint64(bit), dtype=np.uint64)
        return cls._normalize(idx, idx + 1, mask)

    @classmethod
    def from_ranges(cls, starts, stops, bit: int) -> SampleSet:
        """Half-open ranges tagged with role `bit`. Ranges may overlap; empty ranges are ignored."""
        np = _np()
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        stops = np.asarray(stops, dtype=np.int64).reshape(-1)
        if starts.shape != stops.shape:
            raise ValueError("starts and stops must have the same length")
        mask = np.full(starts.shape[0], np.uint64(1) << np.uint64(bit), dtype=np.uint64)
        return cls._build(starts, stops, mask)

    # ------------------------------------------------------------------ queries

    def count(self) -> int:
        return int((self.stops - self.starts).sum())

    def __len__(self) -> int:
        return self.count()

    @property
    def is_empty(self) -> bool:
        return self.starts.shape[0] == 0

    def bits_present(self) -> int:
        np = _np()
        return int(np.bitwise_or.reduce(self.roles)) if self.roles.shape[0] else 0

    def materialize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (idx int64 strictly increasing, roles uint64) for every retained sample."""
        np = _np()
        lengths = self.stops - self.starts
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.uint64)
        offsets = np.cumsum(lengths) - lengths
        idx = np.arange(total, dtype=np.int64) + np.repeat(self.starts - offsets, lengths)
        roles = np.repeat(self.roles, lengths)
        return idx, roles

    def roles_at(self, idx) -> np.ndarray:
        """Role mask for each query index (0 where the index is not retained)."""
        np = _np()
        idx = np.asarray(idx, dtype=np.int64)
        pos = np.searchsorted(self.starts, idx, side="right") - 1
        out = np.zeros(idx.shape, dtype=np.uint64)
        valid = pos >= 0
        inside = np.zeros(idx.shape, dtype=bool)
        inside[valid] = idx[valid] < self.stops[pos[valid]]
        out[inside] = self.roles[pos[inside]]
        return out

    def contains(self, idx) -> np.ndarray:
        return self.roles_at(idx) != 0

    def with_bits(self, mask: int) -> SampleSet:
        """Keep only ranges carrying any bit of `mask`; other bits are cleared."""
        np = _np()
        m = np.uint64(mask)
        kept = self.roles & m
        sel = kept != 0
        return SampleSet._normalize(self.starts[sel], self.stops[sel], kept[sel])

    def indices_with(self, mask: int) -> np.ndarray:
        idx, _ = self.with_bits(mask).materialize()
        return idx

    # ------------------------------------------------------------------ combination

    def union(self, other: SampleSet) -> SampleSet:
        np = _np()
        if other.is_empty:
            return self
        if self.is_empty:
            return other
        return SampleSet._build(
            np.concatenate([self.starts, other.starts]),
            np.concatenate([self.stops, other.stops]),
            np.concatenate([self.roles, other.roles]),
        )

    def add_points(self, idx, bit: int) -> SampleSet:
        return self.union(SampleSet.from_points(idx, bit))

    def add_ranges(self, starts, stops, bit: int) -> SampleSet:
        return self.union(SampleSet.from_ranges(starts, stops, bit))

    def clip(self, n: int) -> SampleSet:
        """Restrict to indices in [0, n)."""
        np = _np()
        starts = np.clip(self.starts, 0, n)
        stops = np.clip(self.stops, 0, n)
        return SampleSet._normalize(starts, stops, self.roles)

    # ------------------------------------------------------------------ internals

    @classmethod
    def _build(cls, starts, stops, roles) -> SampleSet:
        """Union of possibly overlapping ranges with OR-ed roles."""
        np = _np()
        nonempty = (stops > starts) & (roles != 0)
        starts, stops, roles = starts[nonempty], stops[nonempty], roles[nonempty]
        if starts.shape[0] == 0:
            return cls.empty()
        # Fast path: already sorted and disjoint.
        order = np.argsort(starts, kind="stable")
        starts, stops, roles = starts[order], stops[order], roles[order]
        if starts.shape[0] == 1 or bool((starts[1:] >= stops[:-1]).all()):
            return cls._normalize(starts, stops, roles)

        bounds = np.unique(np.concatenate([starts, stops]))
        n_elem = bounds.shape[0] - 1
        acc = np.zeros(n_elem, dtype=np.uint64)
        lo = np.searchsorted(bounds, starts)
        hi = np.searchsorted(bounds, stops)
        # OR over ranges: coverage counting per distinct role mask (few distinct masks in practice).
        for mask in np.unique(roles):
            sel = roles == mask
            diff = np.zeros(n_elem + 1, dtype=np.int64)
            np.add.at(diff, lo[sel], 1)
            np.add.at(diff, hi[sel], -1)
            covered = np.cumsum(diff[:-1]) > 0
            acc[covered] |= mask
        return cls._normalize(bounds[:-1], bounds[1:], acc)

    @classmethod
    def _normalize(cls, starts, stops, roles) -> SampleSet:
        """Drop empty/zero ranges and merge adjacent ranges with identical roles. Input must be sorted and disjoint."""
        np = _np()
        keep = (stops > starts) & (roles != 0)
        starts = np.asarray(starts[keep], dtype=np.int64)
        stops = np.asarray(stops[keep], dtype=np.int64)
        roles = np.asarray(roles[keep], dtype=np.uint64)
        if starts.shape[0] <= 1:
            return cls(starts, stops, roles)
        joins = (starts[1:] == stops[:-1]) & (roles[1:] == roles[:-1])
        if not joins.any():
            return cls(starts, stops, roles)
        head = np.concatenate([[True], ~joins])
        group = np.cumsum(head) - 1
        new_starts = starts[head]
        new_stops = np.zeros(new_starts.shape[0], dtype=np.int64)
        np.maximum.at(new_stops, group, stops)
        return cls(new_starts, new_stops, roles[head])
