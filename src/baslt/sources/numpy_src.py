"""In-memory source: mappings of NumPy arrays, (t, v) tuples or Signals, or an existing Run.

Array values are never copied when they are contiguous float64: every signal that shares a time key gets the
same time array object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..errors import SourceError, UsageError
from ..signals import Run, Signal, SignalInfo, SourceMeta, normalize_signal
from .base import (
    canonical_name,
    guess_kind,
    merge_time_options,
    normalize_hints,
    oriented_shape,
    partition_time_signals,
    prescale_time,
    resolve_time,
    select_names,
)

# Top-level keys used as the shared clock after the sibling and global rules.
TIME_KEYS: tuple[str, ...] = ("t", "time", "tout")


class NumpySource:
    """Adapter over in-memory data.

    Accepted input:

    - ``{"t": t, "x": x, "nav/pos": pos}``: arrays sharing a time key ("t" or "time"; a sibling such as
      "nav/t" wins for keys in that group)
    - ``{"x": (t, x)}``: each signal with its own time
    - ``{"x": Signal(...)}``: already-built signals
    - a ``Run``
    """

    format: ClassVar[str] = "numpy"
    extensions: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        data: Mapping[str, Any] | Run,
        *,
        units: Mapping[str, str | None] | None = None,
        time_hints: Mapping[str, str] | None = None,
        global_time: str | None = None,
    ) -> None:
        self._run: Run | None = None
        self._entries: dict[str, Any] = {}
        self._arrays: dict[str, np.ndarray] = {}
        if isinstance(data, Run):
            self._run = data
        elif isinstance(data, Mapping):
            for key, value in data.items():
                if not isinstance(key, str):
                    raise UsageError(f"in-memory signal names must be strings, got {key!r}")
                name = canonical_name(key)
                if not name or name in self._entries:
                    raise UsageError(f"in-memory signal name {key!r} is empty or duplicated")
                if isinstance(value, Signal):
                    self._entries[name] = value
                elif isinstance(value, tuple):
                    if len(value) != 2:
                        raise UsageError(f"in-memory signal {key!r}: a tuple must be (t, v)")
                    self._entries[name] = value
                else:
                    arr = np.asarray(value)
                    self._entries[name] = arr
                    self._arrays[name] = arr
        else:
            raise UsageError(f"in-memory sources need a mapping or a Run, got {type(data).__name__}")
        self._units = {canonical_name(k): v for k, v in (units or {}).items()}
        self._time_hints = normalize_hints(time_hints)
        self._global_time = global_time

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool:
        return False

    # ----------------------------------------------------------------------------------------------------------

    def _is_time_array(self, name: str) -> bool:
        arr = self._arrays.get(name)
        if arr is None or arr.dtype.kind not in "fiu":
            return False
        return arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape)

    def _plan(self, hints: Mapping[str, str], global_time: str | None) -> tuple[list[str], dict[str, str | None]]:
        array_names = list(self._arrays)
        data, refs = partition_time_signals(array_names, exists=self._is_time_array, time_hints=hints,
                                            global_time=global_time, fallbacks=TIME_KEYS)
        keep = set(data)
        names = [n for n in self._entries if n in keep or n not in self._arrays]
        return names, {n: refs.get(n) for n in names}

    def _unit(self, name: str, default: str | None = None) -> str | None:
        return self._units.get(name, default)

    def list_signals(self) -> list[SignalInfo]:
        if self._run is not None:
            return self._run.infos()
        names, refs = self._plan(self._time_hints, self._global_time)
        infos: list[SignalInfo] = []
        for name in names:
            entry = self._entries[name]
            if isinstance(entry, Signal):
                info = entry.info()
                info.name = name
                info.unit = self._unit(name, entry.unit)
                infos.append(info)
                continue
            if isinstance(entry, tuple):
                t, v = np.asarray(entry[0]), np.asarray(entry[1])
                time_ref = None
            else:
                v = entry
                time_ref = refs[name]
                t = self._arrays[time_ref] if time_ref is not None else None
            n = int(t.size) if t is not None else (int(v.shape[0]) if v.ndim else 1)
            oriented = oriented_shape(v.shape, n)
            shape = oriented[0] if oriented is not None else tuple(int(s) for s in v.shape)
            infos.append(SignalInfo(name=name, path=name, shape=shape, dtype=v.dtype.str, unit=self._unit(name),
                                    kind=guess_kind(v.dtype.kind, shape), n=n, time_ref=time_ref))
        return infos

    def load(
        self,
        names: Sequence[str] | None = None,
        *,
        time_hints: Mapping[str, str] | None = None,
        global_time: str | None = None,
        time_scale: float = 1.0,
        on_non_monotonic: str = "error",
    ) -> Run:
        if self._run is not None:
            return self._load_run(names, time_scale, on_non_monotonic)
        hints, gtime = merge_time_options(self._time_hints, self._global_time, time_hints, global_time)
        listed, _ = self._plan(hints, gtime)
        selected = select_names(names, listed, self._entries)

        issues: list[str] = []
        signals: dict[str, Signal] = {}
        scaled_times: dict[str, tuple[np.ndarray, float]] = {}
        counted: dict[int, int] = {}
        for name in selected:
            entry = self._entries[name]
            if isinstance(entry, Signal):
                sig, sig_issues = normalize_signal(
                    name, entry.t, entry.v, path=entry.path, unit=self._unit(name, entry.unit), kind=entry.kind,
                    labels=entry.labels, time_scale=time_scale, on_non_monotonic=on_non_monotonic)
                sig.source_dtype = entry.source_dtype
                t_raw, v_raw = entry.t, entry.v
            elif isinstance(entry, tuple):
                t_raw, v_raw = entry
                sig, sig_issues = normalize_signal(name, t_raw, v_raw, path=name, unit=self._unit(name),
                                                   time_scale=time_scale, on_non_monotonic=on_non_monotonic)
            else:
                ref = resolve_time(name, exists=self._is_time_array, time_hints=hints, global_time=gtime,
                                   fallbacks=TIME_KEYS)
                if ref not in scaled_times:
                    scaled_times[ref] = prescale_time(self._arrays[ref], time_scale)
                t_shared, remaining = scaled_times[ref]
                t_raw, v_raw = self._arrays[ref], entry
                sig, sig_issues = _normalize_array(name, t_shared, entry, unit=self._unit(name),
                                                   time_scale=remaining, on_non_monotonic=on_non_monotonic)
            for arr in (t_raw, v_raw):
                if isinstance(arr, np.ndarray):
                    counted[id(arr)] = int(arr.nbytes)
            signals[name] = sig
            issues.extend(sig_issues)
        return Run(signals=signals, meta=SourceMeta(path=None, format="numpy", size_bytes=sum(counted.values()),
                                                     issues=issues))

    def _load_run(self, names: Sequence[str] | None, time_scale: float, on_non_monotonic: str) -> Run:
        run = self._run
        assert run is not None
        selected = select_names(names, list(run.signals))
        issues = list(run.meta.issues)
        signals: dict[str, Signal] = {}
        for name in selected:
            src = run.signals[name]
            if time_scale == 1.0:
                signals[name] = src
                continue
            sig, sig_issues = normalize_signal(name, src.t, src.v, path=src.path, unit=src.unit, kind=src.kind,
                                               labels=src.labels, time_scale=time_scale,
                                               on_non_monotonic=on_non_monotonic)
            sig.source_dtype = src.source_dtype
            signals[name] = sig
            issues.extend(sig_issues)
        meta = SourceMeta(path=run.meta.path, format=run.meta.format, size_bytes=run.meta.size_bytes, issues=issues)
        return Run(signals=signals, meta=meta)


def _normalize_array(name: str, t: np.ndarray, v: np.ndarray, *, unit: str | None, time_scale: float,
                     on_non_monotonic: str) -> tuple[Signal, list[str]]:
    if v.ndim == 2:
        oriented = oriented_shape(v.shape, int(t.size))
        if oriented is None:
            raise SourceError(f"signal {name!r}: values of shape {v.shape} do not match {t.size} timestamps")
        if oriented[1]:
            v = v.T
    return normalize_signal(name, t, v, path=name, unit=unit, time_scale=time_scale,
                            on_non_monotonic=on_non_monotonic)
