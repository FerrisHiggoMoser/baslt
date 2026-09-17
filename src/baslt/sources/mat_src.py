"""MATLAB MAT-file source (optional extra ``baslt[mat]``).

Versions 4 to 7.2 are read with ``scipy.io.loadmat``; version 7.3 files are HDF5 and are read with h5py
(``matv73.py``). Both readers build the same tree, which is flattened here into signals:

- a top-level numeric variable ``q`` is the signal ``q``; a struct field ``s.q`` is ``s/q``; element ``i`` of a
  struct array (1-based, as in MATLAB) is ``s/i/...``
- vectors and n-by-k matrices are signals; scalars, empty arrays and char arrays are not
- a cell array of character vectors is a discrete string signal
- a Simulink "structure with time" (fields ``time`` and ``signals``) becomes one signal per element of
  ``signals``, named by its label and timed by the struct's own ``time``
- MATLAB objects (timeseries, timetable, string, datetime, Simulink datasets), sparse and complex arrays cannot be
  read without MATLAB; they are skipped and listed in the run's issues

MAT-files carry no units: declare them in the policy under ``signals.decl``. As for HDF5, a top-level ``tout``
(the name Simulink uses) is the fallback clock.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..errors import BasltError, SourceError, UsageError
from ..signals import Run, Signal, SignalInfo, SourceMeta, normalize_signal
from .base import (
    file_has_hdf5_signature,
    file_size,
    guess_kind,
    merge_time_options,
    normalize_hints,
    oriented_shape,
    partition_time_signals,
    prescale_time,
    read_head,
    resolve_time,
    select_names,
)

TOP_LEVEL_FALLBACKS: tuple[str, ...] = ("tout",)
MAX_DEPTH = 64
MAT73_HEADER = b"MATLAB 7.3"


# --------------------------------------------------------------------------------------------------------------
# Neutral tree


@dataclass(slots=True)
class Leaf:
    """A numeric or string array. `shape` is in MATLAB order with singleton dimensions removed.

    `read(ctx)` returns the values with exactly that shape; `ctx` is whatever the reader opened for loading
    (None for version 5 files, whose data is already in memory).
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    read: Callable[[Any], np.ndarray]


@dataclass(slots=True)
class Skip:
    """A value that cannot become a signal. Quiet skips (placeholders, empty values) are not reported."""

    reason: str = ""
    quiet: bool = False


@dataclass(slots=True)
class _Var:
    leaf: Leaf
    time_attr: str | None = None


def _join(name: str, key: str) -> str:
    return f"{name}/{key}" if name else key


def _text(node: Any) -> str:
    return node.strip() if isinstance(node, str) else ""


def _is_time_leaf(node: Any) -> bool:
    return isinstance(node, Leaf) and len(node.shape) == 1 and node.dtype.kind in "iuf" and node.shape[0] > 0


def _structure_with_time(node: Mapping[str, Any]) -> bool:
    """A Simulink "structure with time": a time vector plus signals whose elements have `values`."""
    if not _is_time_leaf(node.get("time")):
        return False
    signals = node.get("signals")
    elements = [signals] if isinstance(signals, dict) else signals
    return (isinstance(elements, list) and bool(elements)
            and all(isinstance(e, dict) and "values" in e for e in elements))


def flatten(tree: Mapping[str, Any]) -> tuple[dict[str, _Var], list[str]]:
    """Turn a reader's tree into named variables and a list of notes about what was skipped."""
    variables: dict[str, _Var] = {}
    notes: list[str] = []

    def add(name: str, leaf: Leaf, time_attr: str | None) -> None:
        if name in variables:
            notes.append(f"{name}: duplicate name, later value ignored")
            return
        variables[name] = _Var(leaf, time_attr)

    def walk(node: Any, name: str, depth: int, time_attr: str | None = None) -> None:
        if depth > MAX_DEPTH:
            notes.append(f"{name}: nested too deeply")
        elif isinstance(node, Skip):
            if not node.quiet:
                notes.append(f"{name}: {node.reason}")
        elif isinstance(node, Leaf):
            if not node.shape or 0 in node.shape:
                return  # scalars and empty arrays are parameters, not signals
            if len(node.shape) > 2:
                notes.append(f"{name}: {len(node.shape)}-D arrays are not supported")
                return
            add(name, node, time_attr)
        elif isinstance(node, str):
            return
        elif isinstance(node, dict):
            if _structure_with_time(node):
                structure_with_time(node, name, depth)
                return
            for key, child in node.items():
                walk(child, _join(name, str(key)), depth + 1)
        elif isinstance(node, list):
            if not node:
                return
            if all(isinstance(x, dict) for x in node):
                for i, element in enumerate(node, 1):
                    walk(element, _join(name, str(i)), depth + 1)
            elif all(isinstance(x, str) for x in node):
                strings = list(node)
                add(name, Leaf((len(strings),), np.dtype(object),
                               lambda ctx, s=strings: np.array(s, dtype=object)), time_attr)
            else:
                notes.append(f"{name}: cell arrays other than lists of character vectors are not supported")
        else:
            notes.append(f"{name}: unsupported value of type {type(node).__name__}")

    def structure_with_time(node: Mapping[str, Any], name: str, depth: int) -> None:
        clock = _join(name, "time")
        add(clock, node["time"], None)
        signals = node["signals"]
        elements = [signals] if isinstance(signals, dict) else signals
        used: set[str] = set()
        for i, element in enumerate(elements, 1):
            label = _text(element.get("label"))
            if not label:
                label = _text(element.get("blockName")).rsplit("/", 1)[-1]
            label = label.replace("/", "_").strip() or f"signal{i}"
            if label in used or label == "time":
                label = f"{label}_{i}"
            used.add(label)
            walk(element["values"], _join(name, label), depth + 1, time_attr="/" + clock)

    for key, value in tree.items():
        walk(value, str(key), 0)
    return variables, notes


# --------------------------------------------------------------------------------------------------------------
# Version 5 (scipy)


def _scipy_io():
    try:
        import scipy.io
    except ImportError:
        raise UsageError("reading MAT-files needs scipy; install baslt[mat]") from None
    return scipy.io


COMPLEX_REASON = "complex values are not supported"


def _loadmat(sio: Any, path: Path, *, mat_dtype: bool) -> tuple[dict[str, Any], bool]:
    """Load a file; also report whether scipy warned that it cast complex values to real."""
    import warnings

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = sio.loadmat(path, simplify_cells=True, mat_dtype=mat_dtype, chars_as_strings=True)
    except MemoryError:
        raise
    except Exception as exc:  # scipy raises many types for damaged files
        raise SourceError(f"cannot read MAT-file {path}: {exc}") from None
    cast = any(issubclass(w.category, getattr(np.exceptions, "ComplexWarning", RuntimeWarning)) for w in caught)
    return raw, cast


def read_v5_tree(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Read every variable with the dtype MATLAB gives it.

    `mat_dtype=True` is needed because MATLAB stores doubles in the smallest integer type that holds them; without
    it an integer-valued double would come back as, say, uint8. scipy then also casts complex arrays to real,
    silently dropping the imaginary part, so a file that triggers that warning is read a second time with native
    types, only to find which values are complex and skip them.
    """
    sio = _scipy_io()
    raw, cast = _loadmat(sio, path, mat_dtype=True)
    tree = {key: _convert_v5(value) for key, value in raw.items() if not key.startswith("__")}
    if cast:
        native, _ = _loadmat(sio, path, mat_dtype=False)
        tree = _mark_complex(tree, {key: _convert_v5(v) for key, v in native.items() if not key.startswith("__")})
    return tree, []


def _mark_complex(typed: Any, native: Any) -> Any:
    if isinstance(native, Skip) and native.reason == COMPLEX_REASON:
        return native
    if isinstance(typed, dict) and isinstance(native, dict):
        return {key: _mark_complex(value, native.get(key)) for key, value in typed.items()}
    if isinstance(typed, list) and isinstance(native, list) and len(typed) == len(native):
        return [_mark_complex(a, b) for a, b in zip(typed, native)]
    return typed


def _convert_v5(value: Any) -> Any:
    from scipy.io.matlab import MatlabFunction, MatlabOpaque

    if isinstance(value, (MatlabOpaque, MatlabFunction)):
        return Skip("MATLAB objects and function handles cannot be read without MATLAB")
    if isinstance(value, dict):
        return {key: _convert_v5(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_v5(child) for child in value]
    if isinstance(value, str):
        return value
    if hasattr(value, "tocsc"):
        return Skip("sparse arrays are not supported")
    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None:
            return Skip("MATLAB objects cannot be read without MATLAB")
        kind = value.dtype.kind
        if kind == "O":
            return [_convert_v5(child) for child in value.ravel(order="F")]
        if kind in "US":
            return ""
        if kind == "c":
            return Skip(COMPLEX_REASON)
        if kind in "biuf":
            return Leaf(tuple(int(s) for s in value.shape), value.dtype, lambda ctx, a=value: a)
        return Skip(f"unsupported data type {value.dtype}")
    if isinstance(value, (bool, int, float, np.generic)):
        return Leaf((), np.asarray(value).dtype, lambda ctx, a=value: np.asarray(a))
    return Skip(f"unsupported value of type {type(value).__name__}")


# --------------------------------------------------------------------------------------------------------------
# Adapter


class MatSource:
    format: ClassVar[str] = "mat"
    extensions: ClassVar[tuple[str, ...]] = (".mat",)

    def __init__(
        self,
        path: str | Path,
        *,
        time_hints: Mapping[str, str] | None = None,
        global_time: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._time_hints = normalize_hints(time_hints)
        self._global_time = global_time
        self._version: str | None = None
        self._vars: dict[str, _Var] | None = None
        self._tree: dict[str, Any] = {}
        self._notes: list[str] = []

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool:
        return head.startswith(b"MATLAB")

    @property
    def version(self) -> str:
        """"7.3" for HDF5-based files, "5" for everything scipy reads."""
        if self._version is None:
            head = read_head(self.path)
            is_hdf5 = head.startswith(MAT73_HEADER) or file_has_hdf5_signature(self.path, head)
            self._version = "7.3" if is_hdf5 else "5"
        return self._version

    def _scan(self) -> dict[str, _Var]:
        if self._vars is not None:
            return self._vars
        if self.version == "7.3":
            from .matv73 import read_tree

            tree, notes = read_tree(self.path)
        else:
            tree, notes = read_v5_tree(self.path)
        variables, flat_notes = flatten(tree)
        self._vars = variables
        self._tree = tree
        self._notes = notes + flat_notes
        return variables

    def parameters(self) -> dict[str, object]:
        """Scalars and character vectors of the file, by their path (`params/mass`): numbers, bools or text."""
        self._scan()
        found: dict[str, Leaf | str] = {}

        def walk(node: Any, name: str, depth: int) -> None:
            if depth > MAX_DEPTH:
                return
            if isinstance(node, str):
                found[name] = node
            elif isinstance(node, Leaf):
                if node.dtype.kind in "biuf" and (node.shape == () or node.shape == (1,)):
                    found[name] = node
            elif isinstance(node, dict) and not _structure_with_time(node):
                for key, child in node.items():
                    walk(child, _join(name, str(key)), depth + 1)

        for key, value in self._tree.items():
            walk(value, str(key), 0)
        out: dict[str, object] = {}
        with self._context() as ctx:
            for name, value in found.items():
                if isinstance(value, str):
                    out[name] = value
                    continue
                item = np.asarray(value.read(ctx)).reshape(-1)[0]
                out[name] = bool(item) if value.dtype.kind == "b" else item.item()
        return out

    def _is_time_array(self, name: str) -> bool:
        var = self._scan().get(name)
        return var is not None and _is_time_leaf(var.leaf)

    def _attrs(self) -> dict[str, str | None]:
        return {name: var.time_attr for name, var in self._scan().items()}

    def _plan(self, hints: Mapping[str, str], global_time: str | None) -> tuple[list[str], dict[str, str | None]]:
        return partition_time_signals(list(self._scan()), exists=self._is_time_array, attrs=self._attrs(),
                                      time_hints=hints, global_time=global_time, fallbacks=TOP_LEVEL_FALLBACKS)

    def _resolve(self, name: str, hints: Mapping[str, str], global_time: str | None) -> str:
        return resolve_time(name, exists=self._is_time_array, attr=self._scan()[name].time_attr, time_hints=hints,
                            global_time=global_time, fallbacks=TOP_LEVEL_FALLBACKS)

    def _context(self):
        if self.version == "7.3":
            from .matv73 import open_file

            return open_file(self.path)
        return nullcontext(None)

    def list_signals(self) -> list[SignalInfo]:
        variables = self._scan()
        data, refs = self._plan(self._time_hints, self._global_time)
        infos: list[SignalInfo] = []
        for name in data:
            leaf = variables[name].leaf
            ref = refs[name]
            n = variables[ref].leaf.shape[0] if ref is not None else None
            oriented = oriented_shape(leaf.shape, n)
            shape = oriented[0] if oriented is not None else leaf.shape
            infos.append(SignalInfo(name=name, path=name, shape=shape, dtype=leaf.dtype.str, unit=None,
                                    kind=guess_kind(leaf.dtype.kind, shape), n=n if n is not None else shape[0],
                                    time_ref=ref))
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
        variables = self._scan()
        hints, gtime = merge_time_options(self._time_hints, self._global_time, time_hints, global_time)
        data, _ = self._plan(hints, gtime)
        selected = select_names(names, data, variables)
        issues: list[str] = list(self._notes) if names is None else []
        signals: dict[str, Signal] = {}
        times: dict[str, tuple[np.ndarray, float]] = {}
        try:
            with self._context() as ctx:
                for name in selected:
                    ref = self._resolve(name, hints, gtime)
                    if ref not in times:
                        times[ref] = prescale_time(variables[ref].leaf.read(ctx), time_scale)
                    t, remaining = times[ref]
                    v = variables[name].leaf.read(ctx)
                    if v.ndim == 2:
                        oriented = oriented_shape(v.shape, int(t.size))
                        if oriented is None:
                            raise SourceError(f"signal {name!r}: array of size {v.shape[0]}-by-{v.shape[1]} "
                                              f"does not match the {t.size} samples of {ref!r}")
                        if oriented[1]:
                            v = v.T
                    sig, sig_issues = normalize_signal(name, t, v, path=name, unit=None, time_scale=remaining,
                                                       on_non_monotonic=on_non_monotonic)
                    signals[name] = sig
                    issues.extend(sig_issues)
        except BasltError:
            raise
        except OSError as exc:
            raise SourceError(f"cannot read MAT-file {self.path}: {exc}") from None
        fmt = "mat73" if self.version == "7.3" else "mat"
        return Run(signals=signals, meta=SourceMeta(path=str(self.path), format=fmt,
                                                     size_bytes=file_size(self.path), issues=issues))
