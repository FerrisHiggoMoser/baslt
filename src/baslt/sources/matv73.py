"""Reader for MATLAB v7.3 MAT-files: HDF5 files in MATLAB's own layout.

MATLAB writes dimensions in reverse (an m-by-n double is an HDF5 dataset of shape (n, m)), marks datasets and
groups with a ``MATLAB_class`` attribute, stores char arrays as UTF-16 code units, and keeps the contents of cell
and struct arrays in the ``#refs#`` group behind object references. ``read_tree`` turns a file into the neutral
tree of ``mat_src``; numeric data is read only when a signal is loaded, memory-mapped when the dataset is stored
contiguously.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import SourceError, UsageError
from .hdf5_src import CHUNK_CACHE_BYTES, CHUNK_CACHE_SLOTS, _contiguous_offset
from .mat_src import MAX_DEPTH, Leaf, Skip

NUMERIC_CLASSES = frozenset(
    {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
)
SKIPPED_ROOT_MEMBERS = frozenset({"#refs#", "#subsystem#"})
MAX_TEXT_ELEMENTS = 1_000_000
MAX_REFERENCES = 1_000_000


def _h5py():
    try:
        import h5py
    except ImportError:
        raise UsageError("reading MATLAB v7.3 MAT-files needs h5py; install baslt[mat]") from None
    return h5py


@dataclass(slots=True)
class ReadContext:
    file: Any
    size: int | None
    mapping: np.ndarray | None = None
    mapped: bool = False


def _open(path: Path):
    h5py = _h5py()
    try:
        return h5py.File(path, "r", rdcc_nbytes=CHUNK_CACHE_BYTES, rdcc_nslots=CHUNK_CACHE_SLOTS)
    except OSError as exc:
        raise SourceError(f"cannot open MAT-file {path}: {exc}") from None


@contextmanager
def open_file(path: Path) -> Iterator[ReadContext]:
    """Open a file for loading; leaves read their data through the yielded context."""
    f = _open(path)
    try:
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        yield ReadContext(file=f, size=size)
    finally:
        f.close()


def read_tree(path: Path) -> tuple[dict[str, Any], list[str]]:
    with _open(path) as f:
        tree = {key: _node(f[key], f, 0) for key in f if key not in SKIPPED_ROOT_MEMBERS}
    return tree, []


# --------------------------------------------------------------------------------------------------------------
# Attributes


def _decode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("ascii", "replace").strip("\x00 ")
    if isinstance(value, str):
        return value.strip()
    return None


def _matlab_class(obj: Any) -> str | None:
    return _decode(obj.attrs.get("MATLAB_class"))


def _flag(obj: Any, name: str) -> bool:
    value = obj.attrs.get(name)
    if value is None:
        return False
    arr = np.asarray(value)
    return bool(arr.size) and bool(arr.reshape(-1)[0])


def _field_order(group: Any, members: list[str]) -> list[str]:
    """Struct fields in MATLAB order when the file records it, otherwise in HDF5 order."""
    raw = group.attrs.get("MATLAB_fields")
    if raw is None:
        return members
    try:
        names = [b"".join(np.asarray(chars).tolist()).decode("ascii") for chars in np.asarray(raw, dtype=object)]
    except (TypeError, ValueError, UnicodeDecodeError):
        return members
    ordered = [name for name in names if name in members]
    return ordered + [name for name in members if name not in ordered]


# --------------------------------------------------------------------------------------------------------------
# Tree


def _node(obj: Any, f: Any, depth: int) -> Any:
    h5py = _h5py()
    if depth > MAX_DEPTH:
        return Skip("nested too deeply")
    cls = _matlab_class(obj)
    if isinstance(obj, h5py.Group):
        if obj.attrs.get("MATLAB_sparse") is not None:
            return Skip("sparse arrays are not supported")
        if cls not in (None, "struct"):
            return Skip(f"MATLAB {cls} objects cannot be read without MATLAB")
        return _struct(obj, f, depth)
    if not isinstance(obj, h5py.Dataset):
        return Skip("unsupported HDF5 object")
    if _flag(obj, "MATLAB_empty"):
        return "" if cls == "char" else Skip(quiet=True)
    if obj.attrs.get("MATLAB_object_decode") is not None:
        return Skip(f"MATLAB {cls or 'object'} objects cannot be read without MATLAB")
    if cls == "char":
        return _char(obj)
    if h5py.check_ref_dtype(obj.dtype) is not None:
        return _cell(obj, f, depth)
    if cls is not None and cls != "logical" and cls not in NUMERIC_CLASSES:
        return Skip(f"MATLAB {cls} values cannot be read")
    dtype = obj.dtype
    if dtype.fields is not None:
        fields = set(dtype.fields)
        return Skip("complex values are not supported" if {"real", "imag"} <= fields else "compound data")
    if dtype.kind not in "biuf":
        return Skip(f"unsupported data type {dtype}")
    h5shape = tuple(int(s) for s in obj.shape)
    matlab_shape = tuple(reversed(h5shape))
    squeezed = tuple(d for d in matlab_shape if d != 1)
    if 0 in matlab_shape:
        squeezed = (0,)
    logical = cls == "logical"
    out_dtype = np.dtype(bool) if logical else np.dtype(dtype.str).newbyteorder("=")
    return Leaf(squeezed, out_dtype, partial(_read_numeric, obj.name, h5shape, squeezed, logical))


def _references(ds: Any) -> list[Any]:
    """Object references of a cell or struct-array field in MATLAB (column-major) order."""
    refs = ds[()]
    return list(np.asarray(refs, dtype=object).T.ravel(order="F"))


def _resolve(ref: Any, f: Any, depth: int) -> Any:
    if not ref:
        return Skip(quiet=True)
    try:
        return _node(f[ref], f, depth)
    except (KeyError, ValueError):
        return Skip("broken object reference")


def _cell(ds: Any, f: Any, depth: int) -> Any:
    if ds.size > MAX_REFERENCES:
        return Skip(f"cell array with {ds.size} elements is too large")
    return [_resolve(ref, f, depth + 1) for ref in _references(ds)]


def _struct(group: Any, f: Any, depth: int) -> Any:
    h5py = _h5py()
    members = list(group)
    plain: dict[str, Any] = {}
    element_fields: dict[str, Any] = {}
    for key in members:
        child = group.get(key)
        if child is None:
            continue
        is_refs = isinstance(child, h5py.Dataset) and h5py.check_ref_dtype(child.dtype) is not None
        if is_refs and _matlab_class(child) != "cell":
            element_fields[key] = child
        else:
            plain[key] = child
    order = _field_order(group, [k for k in members if k in plain or k in element_fields])

    if element_fields and not plain:
        # A struct array: every field holds one reference per element.
        sizes = {ds.size for ds in element_fields.values()}
        if len(sizes) != 1:
            return Skip("struct array whose fields have different sizes")
        count = sizes.pop()
        if count > MAX_REFERENCES:
            return Skip(f"struct array with {count} elements is too large")
        refs = {key: _references(ds) for key, ds in element_fields.items()}
        return [{key: _resolve(refs[key][i], f, depth + 1) for key in order} for i in range(count)]

    out: dict[str, Any] = {}
    for key in order:
        if key in plain:
            out[key] = _node(plain[key], f, depth + 1)
        else:
            out[key] = _cell(element_fields[key], f, depth + 1)
    return out


def _char(ds: Any) -> str:
    if ds.size > MAX_TEXT_ELEMENTS:
        return ""
    codes = np.asarray(ds[()])
    if codes.dtype.kind not in "iu":
        return ""
    matlab = codes.T  # rows of the MATLAB char matrix
    if matlab.ndim == 1:
        matlab = matlab.reshape(1, -1)
    elif matlab.ndim > 2:
        return ""
    rows = [np.ascontiguousarray(row, dtype="<u2").tobytes().decode("utf-16-le", "replace") for row in matlab]
    return "\n".join(rows)


# --------------------------------------------------------------------------------------------------------------
# Data


def _read_numeric(path: str, h5shape: tuple[int, ...], shape: tuple[int, ...], logical: bool,
                  ctx: ReadContext) -> np.ndarray:
    dset = ctx.file[path]
    raw = _mapped(ctx, dset, h5shape)
    if raw is None:
        raw = np.empty(h5shape, dtype=np.dtype(dset.dtype.str).newbyteorder("="))
        if raw.size:
            try:
                dset.read_direct(raw)
            except OSError as exc:
                raise SourceError(f"cannot read {path!r}: {exc}") from None
    values = raw.T.reshape(shape)
    return values.astype(bool) if logical else values


def _mapped(ctx: ReadContext, dset: Any, h5shape: tuple[int, ...]) -> np.ndarray | None:
    offset = _contiguous_offset(dset, ctx.size)
    if offset is None:
        return None
    if ctx.mapping is None:
        if ctx.mapped:
            return None
        ctx.mapped = True
        try:
            ctx.mapping = np.memmap(ctx.file.filename, dtype=np.uint8, mode="r")
        except (OSError, ValueError):
            return None
    dtype = np.dtype(dset.dtype.str)
    nbytes = math.prod(h5shape) * dtype.itemsize
    try:
        return ctx.mapping[offset : offset + nbytes].view(dtype).reshape(h5shape)
    except (OSError, ValueError):
        return None
