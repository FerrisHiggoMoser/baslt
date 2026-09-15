"""Source adapter protocol, format registry, `open_source` dispatch and the shared time association rules.

This module imports only the standard library at import time; adapters are imported when a format is
first needed.
"""

from __future__ import annotations

import difflib
import inspect
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from ..errors import SourceError, UsageError

if TYPE_CHECKING:
    from ..signals import Kind, Run, SignalInfo

TIME_SIBLINGS: tuple[str, ...] = ("t", "time", "Time", "timestamp", "tout")
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
SNIFF_BYTES = 8192
MAT_UNAVAILABLE = "MAT-file support is not available yet"


@runtime_checkable
class SourceAdapter(Protocol):
    """What every source adapter provides. See docs/architecture.md."""

    format: ClassVar[str]
    extensions: ClassVar[tuple[str, ...]]

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool: ...

    def list_signals(self) -> list[SignalInfo]: ...

    def load(
        self,
        names: Sequence[str] | None = None,
        *,
        time_hints: Mapping[str, str] | None = None,
        global_time: str | None = None,
        time_scale: float = 1.0,
        on_non_monotonic: str = "error",
    ) -> Run: ...


# --------------------------------------------------------------------------------------------------------------
# Registry


@dataclass(frozen=True, slots=True)
class AdapterEntry:
    """A registered format. `target` is an adapter class or a lazy "module:attribute" reference."""

    format: str
    target: str | type
    extensions: tuple[str, ...] = ()
    sniffs: bool = True

    def load(self) -> type:
        if isinstance(self.target, str):
            module_name, _, attr = self.target.partition(":")
            return getattr(import_module(module_name), attr)
        return self.target


_REGISTRY: dict[str, AdapterEntry] = {}
_ALIASES: dict[str, str] = {"h5": "hdf5", "hdf": "hdf5", "he5": "hdf5", "tsv": "csv", "txt": "csv", "memory": "numpy"}
_UNAVAILABLE: dict[str, str] = {"mat": MAT_UNAVAILABLE, "mat73": MAT_UNAVAILABLE}
_UNAVAILABLE_EXTENSIONS: dict[str, str] = {".mat": "mat"}


def register_adapter(
    format: str,
    target: str | type,
    extensions: Iterable[str] | None = None,
    *,
    sniffs: bool = True,
    replace: bool = False,
) -> None:
    """Register a source format. Formats are tried for sniffing in registration order."""
    key = format.lower()
    if key in _REGISTRY and not replace:
        raise ValueError(f"source format {key!r} is already registered")
    if extensions is None:
        extensions = getattr(target, "extensions", ()) if not isinstance(target, str) else ()
    _REGISTRY[key] = AdapterEntry(key, target, tuple(e.lower() for e in extensions), sniffs)


def available_formats() -> list[str]:
    return list(_REGISTRY)


def _format_key(format: str) -> str:
    key = format.lower()
    return _ALIASES.get(key, key)


def adapter_class(format: str) -> type:
    """Return the adapter class for a format name or alias."""
    key = _format_key(format)
    entry = _REGISTRY.get(key)
    if entry is None:
        if key in _UNAVAILABLE:
            raise UsageError(_UNAVAILABLE[key])
        raise UsageError(f"unknown source format {format!r}; expected one of {', '.join(_REGISTRY)}")
    return entry.load()


register_adapter("numpy", "baslt.sources.numpy_src:NumpySource", (), sniffs=False)
register_adapter("hdf5", "baslt.sources.hdf5_src:Hdf5Source", (".h5", ".hdf5", ".he5"))
register_adapter("csv", "baslt.sources.csv_src:CsvSource", (".csv", ".tsv", ".txt"))


# --------------------------------------------------------------------------------------------------------------
# Dispatch


def open_source(source: str | os.PathLike[str] | Mapping[str, Any] | Run, *, format: str | None = None,
                **options: Any) -> SourceAdapter:
    """Open a source for listing and loading.

    A `Mapping` or `Run` opens the in-memory adapter. A path is resolved by explicit `format`, then by
    extension, then by sniffing the first bytes of the file. `options` are passed to the adapter.
    """
    from ..signals import Run

    if isinstance(source, (Run, Mapping)):
        if format is not None and _format_key(format) != "numpy":
            raise UsageError(f"in-memory sources use format 'numpy', got format={format!r}")
        return _construct(adapter_class("numpy"), source, options)
    if not isinstance(source, (str, os.PathLike)):
        raise UsageError(
            f"cannot open a source of type {type(source).__name__}; pass a file path, a mapping of arrays or a Run"
        )
    path = Path(source)

    if format is not None:
        cls = adapter_class(format)
        if cls.format == "numpy":
            raise UsageError("format 'numpy' needs an in-memory mapping or Run, not a path")
        _require_file(path)
        return _construct(cls, path, options)

    suffix = path.suffix.lower()
    for entry in _REGISTRY.values():
        if suffix and suffix in entry.extensions:
            cls = entry.load()
            _require_file(path)
            return _construct(cls, path, options)
    unavailable = _UNAVAILABLE_EXTENSIONS.get(suffix)
    if unavailable is not None and unavailable not in _REGISTRY:
        raise UsageError(_UNAVAILABLE[unavailable])

    _require_file(path)
    head = read_head(path)
    if head.startswith(b"MATLAB") and "mat" not in _REGISTRY:
        raise UsageError(MAT_UNAVAILABLE)
    for entry in _REGISTRY.values():
        if not entry.sniffs:
            continue
        cls = entry.load()
        if cls.sniff(path, head):
            return _construct(cls, path, options)
    formats = ", ".join(e.format for e in _REGISTRY.values() if e.sniffs)
    raise SourceError(f"cannot determine the format of {path}; pass format= one of {formats}")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SourceError(f"source file not found: {path}")
    if path.is_dir():
        raise SourceError(f"source is a directory, not a file: {path}")


def read_head(path: Path, size: int = SNIFF_BYTES) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(size)
    except OSError as exc:
        raise SourceError(f"cannot read source {path}: {exc.strerror or exc}") from None


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _construct(cls: type, source: object, options: Mapping[str, Any]) -> SourceAdapter:
    try:
        inspect.signature(cls).bind(source, **options)
    except TypeError as exc:
        raise UsageError(f"invalid option for {cls.format} sources: {exc}") from None
    return cls(source, **options)


def hdf5_signature_offsets(limit: int) -> list[int]:
    """Every legal superblock offset up to `limit`: 0, then one per user block size (512 * 2**k)."""
    offsets = [0]
    offset = 512
    while offset <= limit:
        offsets.append(offset)
        offset *= 2
    return offsets


def has_hdf5_signature(head: bytes) -> bool:
    """Whether `head` holds the HDF5 signature at offset 0 or after a user block inside it."""
    size = len(HDF5_SIGNATURE)
    return any(head[o : o + size] == HDF5_SIGNATURE for o in hdf5_signature_offsets(max(len(head) - size, 0)))


def file_has_hdf5_signature(path: Path, head: bytes) -> bool:
    """Like `has_hdf5_signature`, but also reads past `head` for user blocks larger than the sniff buffer."""
    if has_hdf5_signature(head):
        return True
    size = file_size(path)
    width = len(HDF5_SIGNATURE)
    if size is None:
        return False
    beyond = [o for o in hdf5_signature_offsets(size - width) if o + width > len(head)]
    if not beyond:
        return False
    try:
        with open(path, "rb") as fh:
            for offset in beyond:
                fh.seek(offset)
                if fh.read(width) == HDF5_SIGNATURE:
                    return True
    except OSError:
        return False
    return False


# --------------------------------------------------------------------------------------------------------------
# Names


def canonical_name(path: str) -> str:
    """A canonical signal name is its source path without a leading "/"."""
    return path.lstrip("/")


def split_name(name: str) -> tuple[str, str]:
    group, _, leaf = name.rpartition("/")
    return group, leaf


def normalize_hints(time_hints: Mapping[str, str] | None) -> dict[str, str]:
    if not time_hints:
        return {}
    return {canonical_name(str(k)): str(v) for k, v in time_hints.items()}


def merge_time_options(
    base_hints: Mapping[str, str],
    base_global: str | None,
    time_hints: Mapping[str, str] | None,
    global_time: str | None,
) -> tuple[dict[str, str], str | None]:
    """Options given to `load` override those given when the source was opened."""
    hints = dict(base_hints)
    hints.update(normalize_hints(time_hints))
    return hints, global_time if global_time is not None else base_global


def select_names(names: Sequence[str] | str | None, listed: Sequence[str], known: Iterable[str] | None = None,
                 *, what: str = "signal") -> list[str]:
    """Resolve requested names to canonical names: order kept, duplicates dropped, unknown names rejected."""
    if names is None:
        return list(listed)
    if isinstance(names, str):
        names = [names]
    allowed = set(listed) if known is None else set(known) | set(listed)
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = canonical_name(str(raw))
        if name in seen:
            continue
        if name not in allowed:
            close = difflib.get_close_matches(name, list(listed), n=3)
            hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
            raise SourceError(f"unknown {what} {name!r}{hint}")
        seen.add(name)
        out.append(name)
    return out


# --------------------------------------------------------------------------------------------------------------
# Time association


def _lookup(ref: str, group: str, exists: Callable[[str], bool], *, relative_first: bool) -> str | None:
    ref = ref.strip()
    if not ref:
        return None
    if ref.startswith("/"):
        candidates = [canonical_name(ref)]
    else:
        relative = f"{group}/{ref}" if group else ref
        candidates = [relative, ref] if relative_first else [ref, relative]
    for candidate in candidates:
        if exists(candidate):
            return candidate
    return None


def resolve_time(
    name: str,
    *,
    exists: Callable[[str], bool],
    attr: str | None = None,
    time_hints: Mapping[str, str] | None = None,
    global_time: str | None = None,
    fallbacks: Sequence[str] = ("tout",),
) -> str:
    """Return the canonical name of the time signal of `name`.

    Order: `time_hints[name]` -> the signal's `time`/`t` attribute -> a sibling named t, time, Time, timestamp
    or tout in the same group -> `global_time` -> `fallbacks` (top-level `tout` for files) -> SourceError.
    `exists(name)` tells whether a usable time array with that canonical name exists.
    """
    group, _ = split_name(name)
    if time_hints and name in time_hints:
        ref = time_hints[name]
        found = _lookup(ref, group, exists, relative_first=False)
        if found is None:
            raise SourceError(f"time for {name!r}: time_hints names {ref!r}, which is not a time array in the source")
        return found
    if attr:
        found = _lookup(attr, group, exists, relative_first=True)
        if found is None:
            raise SourceError(f"time for {name!r}: its time attribute names {attr!r}, which is not a time array "
                              "in the source")
        return found
    for leaf in TIME_SIBLINGS:
        candidate = f"{group}/{leaf}" if group else leaf
        if candidate != name and exists(candidate):
            return candidate
    if global_time:
        found = _lookup(global_time, "", exists, relative_first=False)
        if found is None:
            raise SourceError(f"global time signal {global_time!r} is not a time array in the source")
        return found
    for candidate in fallbacks:
        if candidate != name and exists(candidate):
            return candidate
    raise SourceError(
        f"no time signal for {name!r}: add a sibling named t, time, Time, timestamp or tout, give it a time "
        "attribute, or set a global time signal (signals.time in the policy)"
    )


def partition_time_signals(
    names: Sequence[str],
    *,
    exists: Callable[[str], bool],
    attrs: Mapping[str, str | None] | None = None,
    time_hints: Mapping[str, str] | None = None,
    global_time: str | None = None,
    fallbacks: Sequence[str] = ("tout",),
) -> tuple[list[str], dict[str, str | None]]:
    """Split array names into data signals and time signals.

    Time signals are arrays named like a time sibling plus every array some other array resolves to as its
    time. Returns the data signal names in source order and each one's time reference (None if unresolved).
    """
    attrs = attrs or {}
    time_set = {n for n in names if split_name(n)[1] in TIME_SIBLINGS}
    refs: dict[str, str | None] = {}
    for name in names:
        if name in time_set:
            continue
        try:
            refs[name] = resolve_time(name, exists=exists, attr=attrs.get(name), time_hints=time_hints,
                                      global_time=global_time, fallbacks=fallbacks)
        except SourceError:
            refs[name] = None
    time_set.update(ref for ref in refs.values() if ref is not None)
    data = [n for n in names if n not in time_set]
    return data, {n: refs[n] for n in data}


def oriented_shape(shape: Sequence[int], n: int | None) -> tuple[tuple[int, ...], bool] | None:
    """Shape of a signal's values with samples along axis 0, and whether a 2-D array must be transposed.

    Rows are samples when shape[0] equals the time length, columns when shape[1] does. Single-component
    2-D arrays become 1-D. Returns None when a 2-D shape does not match the time length.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) != 2:
        return shape, False
    rows, cols = shape
    if n is None or rows == n:
        return ((rows,) if cols == 1 else shape), False
    if cols == n:
        return ((cols,) if rows == 1 else (cols, rows)), True
    return None


def guess_kind(dtype_kind: str, shape: Sequence[int]) -> Kind:
    """Signal kind from dtype and oriented shape, mirroring `baslt.signals.infer_kind`."""
    if len(shape) == 2 and shape[1] > 1:
        return "vector"
    if dtype_kind in "biuUSO":
        return "discrete"
    return "continuous"


def prescale_time(t: Any, time_scale: float) -> tuple[Any, float]:
    """Scale a shared numeric time array once, so every signal using it shares the scaled array.

    Numeric input that is not already float64 is converted here too, even when there is nothing to scale:
    `normalize_signal` would otherwise convert it again for every signal, one full copy of the clock each time.
    Returns the array and the scale still to apply (1.0 once applied, unchanged for non-numeric input so that
    `normalize_signal` reports the problem).
    """
    import numpy as np

    arr = np.asarray(t)
    if arr.dtype.kind not in "fiu":
        return arr, time_scale
    if arr.dtype != np.dtype(np.float64):
        scaled = np.asarray(arr, dtype=np.float64)
        if time_scale != 1.0:
            scaled *= float(time_scale)  # a fresh array, so scaling it in place copies nothing
        return scaled, 1.0
    if time_scale != 1.0:
        return arr * float(time_scale), 1.0
    return arr, 1.0
