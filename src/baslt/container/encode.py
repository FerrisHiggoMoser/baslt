"""Array encodings of the `.baslt` container: raw, byte-plane shuffle, and delta+shuffle for indices.

Vector arrays `(n, k)` are laid out component-sequentially (all of component 0, then component 1, ...).
"""

from __future__ import annotations

import numpy as np

from .spec import (
    DELTA_DTYPES,
    DTYPE_WIDTHS,
    ENC_DELTA_SHUFFLE,
    ENC_RAW,
    ENC_SHUFFLE,
    ENCODINGS,
    MAX_EXACT_FLOAT_INT,
    ArrayDesc,
    check_desc,
)

__all__ = ["container_dtype", "decode_array", "encode_array"]


def container_dtype(dtype: object) -> str:
    """Container dtype string for a numpy dtype (`bool` maps to `|u1`). Raises ValueError otherwise."""
    dt = np.dtype(dtype)
    if dt.kind == "b":
        return "|u1"
    if dt.kind in "iuf":
        code = ("|" if dt.itemsize == 1 else "<") + dt.kind + str(dt.itemsize)
        if code in DTYPE_WIDTHS:
            return code
    raise ValueError(f"dtype {dt} cannot be stored; supported: {', '.join(DTYPE_WIDTHS)}")


def _sequential(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """Contiguous 1-D little-endian array in component-sequential order, and its container dtype."""
    a = np.asarray(arr)
    if a.ndim not in (1, 2):
        raise ValueError(f"arrays must have shape (n,) or (n, k), got {a.shape}")
    if a.dtype.kind == "b":
        a = a.view(np.uint8)
    code = container_dtype(a.dtype)
    a = a.astype(code, copy=False)
    flat = a.ravel(order="F") if a.ndim == 2 else a
    return np.ascontiguousarray(flat), code


def _shuffle_bytes(flat: np.ndarray) -> bytes:
    width = flat.dtype.itemsize
    if width == 1:
        return flat.tobytes()
    # planes[p, i] = raw[i*w + p]; C-order bytes give out[p*N + i] = raw[i*w + p]
    return flat.view(np.uint8).reshape(flat.shape[0], width).T.tobytes()


def _check_idx_like(a: np.ndarray, code: str) -> None:
    if code not in DELTA_DTYPES:
        raise ValueError(f"delta+shuffle is only for idx arrays stored as <u4 or <f8, got {code}")
    if code == "<f8":
        if not np.isfinite(a).all():
            raise ValueError("delta+shuffle <f8 values must be finite")
        if (a < 0).any():
            raise ValueError("delta+shuffle values must be non-negative")
        if (np.floor(a) != a).any():
            raise ValueError("delta+shuffle <f8 values must be exact integers")
        if (a >= MAX_EXACT_FLOAT_INT).any():
            raise ValueError("delta+shuffle <f8 values must be below 2**53")
    if a.shape[0] > 1 and not (a[1:] > a[:-1]).all():
        raise ValueError("delta+shuffle values must be strictly increasing")


def encode_array(arr: np.ndarray, enc: str) -> bytes:
    """Encode `arr` of shape (n,) or (n, k) as `raw`, `shuffle` or `delta+shuffle` bytes."""
    if enc not in ENCODINGS:
        raise ValueError(f"unknown encoding {enc!r}; expected one of {', '.join(ENCODINGS)}")
    if enc == ENC_DELTA_SHUFFLE and np.ndim(arr) != 1:
        raise ValueError(f"delta+shuffle needs a 1-D idx array, got shape {np.shape(arr)}")
    flat, code = _sequential(arr)
    if enc == ENC_RAW:
        return flat.tobytes()
    if enc == ENC_SHUFFLE:
        return _shuffle_bytes(flat)
    _check_idx_like(flat, code)
    deltas = np.empty_like(flat)
    if flat.shape[0]:
        deltas[0] = flat[0]
        np.subtract(flat[1:], flat[:-1], out=deltas[1:])
        if code == "<f8":
            deltas += 0.0  # normalise a leading -0.0
    return _shuffle_bytes(deltas)


def decode_array(buf: bytes | memoryview, desc: ArrayDesc) -> np.ndarray:
    """Decode the array described by `desc` from its member's uncompressed bytes `buf`.

    Returns shape (n,) when components == 1, else (n, components), as a new writable array in the
    descriptor dtype. `delta+shuffle` arrays are decoded by cumulative sum in float64 and returned as
    float64 holding exact integers.
    """
    width = check_desc(desc)
    view = memoryview(buf).cast("B")
    end = desc.offset + desc.nbytes
    if end > view.nbytes:
        raise ValueError(
            f"array {desc.name!r} spans bytes {desc.offset}..{end} but member {desc.member!r} "
            f"has {view.nbytes} bytes"
        )
    n, k = desc.n, desc.components
    count = n * k
    out_dtype = np.dtype(np.float64) if desc.enc == ENC_DELTA_SHUFFLE else np.dtype(desc.dtype)
    shape = (n,) if k == 1 else (n, k)
    if count == 0:
        return np.empty(shape, dtype=out_dtype)
    if desc.enc == ENC_RAW:
        flat = np.frombuffer(view, dtype=desc.dtype, count=count, offset=desc.offset)
    else:
        planes = np.frombuffer(view, dtype=np.uint8, count=count * width, offset=desc.offset)
        flat = planes.reshape(width, count).T.copy().view(desc.dtype).reshape(count)
        if desc.enc == ENC_DELTA_SHUFFLE:
            flat = np.cumsum(flat, dtype=np.float64)
    if desc.enc == ENC_RAW:
        # `flat` is a view on `buf`: always copy, even when the (k, n) transpose is already contiguous
        return flat.copy() if k == 1 else flat.reshape(k, n).T.copy()
    # `flat` is already a private copy here
    return flat if k == 1 else np.ascontiguousarray(flat.reshape(k, n).T)
