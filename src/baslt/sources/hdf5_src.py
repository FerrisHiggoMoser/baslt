"""HDF5 source adapter built on h5py (optional extra ``baslt[hdf5]``).

Every numeric 1-D or 2-D dataset in any group is a signal, except time arrays. Signals are read one at a time:
contiguous, unfiltered, little-endian datasets are memory-mapped straight from the file; everything else is read
with ``read_direct`` into a preallocated array through a large chunk cache.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..errors import SourceError, UsageError
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
    resolve_time,
    select_names,
)

CHUNK_CACHE_BYTES = 64 * 2**20
CHUNK_CACHE_SLOTS = 12_421
PATHOLOGICAL_CHUNK_ELEMENTS = 1024
PATHOLOGICAL_DATASET_ELEMENTS = 1_000_000
TOP_LEVEL_FALLBACKS: tuple[str, ...] = ("tout",)
MAX_GROUP_DEPTH = 64


def _h5py():
    try:
        import h5py
    except ImportError:
        raise UsageError("reading HDF5 files needs h5py; install baslt[hdf5]") from None
    return h5py


@dataclass(slots=True)
class _ReadState:
    """What one `load` call shares between reads: the file size, its issue list and its memory map."""

    size: int | None
    issues: list[str]
    mapping: np.ndarray | None = None
    mapped: bool = False


@dataclass(slots=True)
class _Dataset:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    unit: str | None
    time_attr: str | None
    chunks: tuple[int, ...] | None

    @property
    def size(self) -> int:
        return math.prod(self.shape)


class Hdf5Source:
    format: ClassVar[str] = "hdf5"
    extensions: ClassVar[tuple[str, ...]] = (".h5", ".hdf5", ".he5")

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
        self._datasets: dict[str, _Dataset] | None = None
        self._skipped: list[str] = []

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool:
        return file_has_hdf5_signature(path, head)

    # ----------------------------------------------------------------------------------------------------------

    def _open(self):
        h5py = _h5py()
        try:
            return h5py.File(self.path, "r", rdcc_nbytes=CHUNK_CACHE_BYTES, rdcc_nslots=CHUNK_CACHE_SLOTS)
        except OSError as exc:
            raise SourceError(f"cannot open HDF5 source {self.path}: {exc}") from None

    def _scan(self) -> dict[str, _Dataset]:
        if self._datasets is not None:
            return self._datasets
        datasets: dict[str, _Dataset] = {}
        skipped: list[str] = []
        with self._open() as f:
            self._walk(f, "", f, datasets, skipped, (_object_id(f["/"]),))
        self._datasets = datasets
        self._skipped = skipped
        return datasets

    def _walk(self, group: Any, prefix: str, f: Any, datasets: dict[str, _Dataset], skipped: list[str],
              ancestors: tuple[object, ...]) -> None:
        """Register every dataset reachable by name, under that name.

        `visititems` follows hard links only, so a time array published under a conventional name through a soft
        link would never be seen. Links into other files are skipped: their offsets are not offsets in this file.
        """
        h5py = _h5py()
        for key in group:
            if group.get(key, getclass=True, getlink=True) is h5py.ExternalLink:
                continue
            obj = group.get(key)  # None for a link that resolves to nothing
            name = f"{prefix}{key}"
            if isinstance(obj, h5py.Group):
                ident = _object_id(obj)
                if len(ancestors) >= MAX_GROUP_DEPTH or (ident is not None and ident in ancestors):
                    continue  # a link back into an enclosing group
                self._walk(obj, f"{name}/", f, datasets, skipped, (*ancestors, ident))
            elif isinstance(obj, h5py.Dataset):
                dtype = obj.dtype
                if dtype.kind not in "biuf" or dtype.fields is not None or obj.ndim not in (1, 2):
                    skipped.append(name)
                    continue
                attrs = obj.attrs
                unit = _attr_text(attrs.get("units"), f) or _attr_text(attrs.get("unit"), f)
                time_attr = _attr_text(attrs.get("time"), f) or _attr_text(attrs.get("t"), f)
                datasets[name] = _Dataset(name=name, shape=tuple(int(s) for s in obj.shape), dtype=dtype, unit=unit,
                                          time_attr=time_attr, chunks=obj.chunks)

    def parameters(self) -> dict[str, object]:
        """Scalar datasets and the file's own attributes, by their path: numbers, bools or text."""
        h5py = _h5py()
        out: dict[str, object] = {}
        with self._open() as f:
            for key, value in f.attrs.items():
                item = _scalar(value, f)
                if item is not None:
                    out[key] = item

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.size == 1 and obj.ndim <= 1 and obj.dtype.fields is None:
                    item = _scalar(obj[()], f)
                    if item is not None:
                        out[name] = item

            f.visititems(visit)
        return out

    def _is_time_array(self, name: str) -> bool:
        ds = self._scan().get(name)
        if ds is None:
            return False
        return len(ds.shape) == 1 or 1 in ds.shape

    def _plan(self, hints: Mapping[str, str], global_time: str | None) -> tuple[list[str], dict[str, str | None]]:
        datasets = self._scan()
        attrs = {name: ds.time_attr for name, ds in datasets.items()}
        return partition_time_signals(list(datasets), exists=self._is_time_array, attrs=attrs, time_hints=hints,
                                      global_time=global_time, fallbacks=TOP_LEVEL_FALLBACKS)

    def _resolve(self, name: str, hints: Mapping[str, str], global_time: str | None) -> str:
        return resolve_time(name, exists=self._is_time_array, attr=self._scan()[name].time_attr, time_hints=hints,
                            global_time=global_time, fallbacks=TOP_LEVEL_FALLBACKS)

    def list_signals(self) -> list[SignalInfo]:
        datasets = self._scan()
        data, refs = self._plan(self._time_hints, self._global_time)
        infos: list[SignalInfo] = []
        for name in data:
            ds = datasets[name]
            ref = refs[name]
            n = datasets[ref].size if ref is not None else None
            oriented = oriented_shape(ds.shape, n)
            shape = oriented[0] if oriented is not None else ds.shape
            infos.append(SignalInfo(name=name, path=name, shape=shape, dtype=ds.dtype.str, unit=ds.unit,
                                    kind=guess_kind(ds.dtype.kind, shape), n=n if n is not None else shape[0],
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
        datasets = self._scan()
        hints, gtime = merge_time_options(self._time_hints, self._global_time, time_hints, global_time)
        data, _ = self._plan(hints, gtime)
        selected = select_names(names, data, datasets)
        issues: list[str] = []
        if names is None and self._skipped:
            shown = ", ".join(self._skipped[:10]) + (", ..." if len(self._skipped) > 10 else "")
            issues.append(f"skipped {len(self._skipped)} datasets that are not numeric 1-D or 2-D arrays: {shown}")
        size = file_size(self.path)
        state = _ReadState(size=size, issues=issues)
        signals: dict[str, Signal] = {}
        times: dict[str, tuple[np.ndarray, float]] = {}
        with self._open() as f:
            for name in selected:
                ds = datasets[name]
                ref = self._resolve(name, hints, gtime)
                if ref not in times:
                    times[ref] = prescale_time(self._read(f, datasets[ref], state), time_scale)
                t, remaining = times[ref]
                v = self._read(f, ds, state)
                if v.ndim == 2:
                    oriented = oriented_shape(v.shape, int(t.size))
                    if oriented is None:
                        raise SourceError(f"signal {name!r}: dataset shape {v.shape} does not match the "
                                          f"{t.size} timestamps of {ref!r}")
                    if oriented[1]:
                        v = v.T
                sig, sig_issues = normalize_signal(name, t, v, path=name, unit=ds.unit, time_scale=remaining,
                                                   on_non_monotonic=on_non_monotonic)
                signals[name] = sig
                issues.extend(sig_issues)
        meta = SourceMeta(path=str(self.path), format="hdf5", size_bytes=size, issues=issues)
        return Run(signals=signals, meta=meta)

    # ----------------------------------------------------------------------------------------------------------

    def _read(self, f: Any, ds: _Dataset, state: _ReadState) -> np.ndarray:
        native = np.dtype(ds.dtype.str).newbyteorder("=")
        if ds.size == 0:
            return np.empty(ds.shape, dtype=native)
        if (ds.chunks is not None and ds.size > PATHOLOGICAL_DATASET_ELEMENTS
                and math.prod(ds.chunks) < PATHOLOGICAL_CHUNK_ELEMENTS):
            state.issues.append(f"{ds.name}: {ds.size} elements stored in chunks of {math.prod(ds.chunks)} elements "
                                f"(chunks={ds.chunks}); reading is slow, consider rechunking the file")
        dset = f[ds.name]
        offset = _contiguous_offset(dset, state.size)
        if offset is not None:
            view = self._mapped(state, ds, offset)
            if view is not None:
                return view
        out = np.empty(ds.shape, dtype=native)
        try:
            dset.read_direct(out)
        except OSError as exc:
            raise SourceError(f"cannot read dataset {ds.name!r} of {self.path}: {exc}") from None
        return out

    def _mapped(self, state: _ReadState, ds: _Dataset, offset: int) -> np.ndarray | None:
        """A view into the single memory map of the file, or None when it cannot be mapped.

        One map serves every dataset: a map per dataset would hold a duplicated file descriptor per signal and
        exhaust the descriptor limit on a file with many signals.
        """
        if state.mapping is None:
            if state.mapped:
                return None
            state.mapped = True
            try:
                state.mapping = np.memmap(self.path, dtype=np.uint8, mode="r")
            except (OSError, ValueError):
                return None
        dtype = np.dtype(ds.dtype.str)
        try:
            return state.mapping[offset : offset + ds.size * dtype.itemsize].view(dtype).reshape(ds.shape)
        except (OSError, ValueError):
            return None


def _contiguous_offset(dset: Any, size: int | None) -> int | None:
    """File offset of a dataset that can be memory-mapped as is, or None."""
    h5py = _h5py()
    if dset.chunks is not None:
        return None
    dtype = dset.dtype
    if dtype.kind not in "iuf" or dtype.metadata or sys.byteorder != "little" or dtype.byteorder == ">":
        return None
    dsid = dset.id
    plist = dsid.get_create_plist()
    if plist.get_nfilters() != 0 or plist.get_external_count() != 0:
        return None
    if dsid.get_space_status() != h5py.h5d.SPACE_STATUS_ALLOCATED:
        return None
    offset = dsid.get_offset()
    if offset is None:
        return None
    nbytes = math.prod(dset.shape) * dtype.itemsize
    if size is None or offset + nbytes > size:
        return None
    return int(offset)


def _object_id(obj: Any) -> tuple[int, int] | None:
    """An object's identity in its file, used to stop a link cycle from being walked forever."""
    h5py = _h5py()
    try:
        info = h5py.h5o.get_info(obj.id)
        return int(info.fileno), int(info.addr)
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return None


def _scalar(value: Any, f: Any) -> object:
    """A number, bool or text from a one-element HDF5 value, else None."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (int, float)):
        return value
    return _attr_text(value, f)


def _attr_text(value: Any, f: Any) -> str | None:
    """Decode a text-like attribute (str, bytes, one-element array or object reference)."""
    if value is None:
        return None
    h5py = _h5py()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, h5py.Reference):
        if not value:
            return None
        try:
            return f[value].name
        except (KeyError, ValueError):
            return None
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", "replace")
    if isinstance(value, str):
        return value.strip() or None
    return None
