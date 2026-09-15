"""Constants and small helpers that define the `.baslt` container profile v1.

This module is numpy-free so that header inspection and size planning stay cheap.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

# --- format identity -------------------------------------------------------------------------

FORMAT = "baslt"
CONTAINER_VERSION = 1  # version written by this build
SUPPORTED_CONTAINER = 1  # highest container version this build can read
READER_MIN = 1  # oldest reader version able to read what this build writes
BYTE_ORDER = "little"

# --- member names ----------------------------------------------------------------------------

MEMBER_HEADER = "baslt.json"
MEMBER_INDEX = "index.json"
MEMBER_MANIFEST = "manifest.json"
MEMBER_POLICY = "policy.json"
JSON_MEMBERS: tuple[str, ...] = (MEMBER_HEADER, MEMBER_INDEX, MEMBER_MANIFEST, MEMBER_POLICY)
ALWAYS_STORED: frozenset[str] = frozenset({MEMBER_HEADER, MEMBER_MANIFEST})
SIGNAL_MEMBER_PREFIX = "s/"
TIME_MEMBER_PREFIX = "t/"

_NAME_RE = re.compile(r"[a-z0-9._/]+")

# --- ZIP subset ------------------------------------------------------------------------------

METHOD_STORE = 0
METHOD_DEFLATE = 8
METHOD_ZSTD = 93
METHODS: frozenset[int] = frozenset({METHOD_STORE, METHOD_DEFLATE, METHOD_ZSTD})
CODEC_METHODS: dict[str, int] = {"store": METHOD_STORE, "deflate": METHOD_DEFLATE, "zstd": METHOD_ZSTD}

VERSION_MADE_BY = 20
VERSION_NEEDED = 20
VERSION_NEEDED_ZSTD = 63
DOS_TIME = 0
DOS_DATE = 0x0021  # 1980-01-01
DEFAULT_LEVEL = 6

LOCAL_SIGNATURE = b"PK\x03\x04"
CENTRAL_SIGNATURE = b"PK\x01\x02"
EOCD_SIGNATURE = b"PK\x05\x06"
LOCAL_HEADER_SIZE = 30
CENTRAL_HEADER_SIZE = 46
EOCD_SIZE = 22
LOCAL_HEADER_FORMAT = "<4sHHHHHIIIHH"  # 30 bytes
CENTRAL_HEADER_FORMAT = "<4sHHHHHHIIIHHHHHII"  # 46 bytes
EOCD_FORMAT = "<4sHHHHIIH"  # 22 bytes

MAX_MEMBERS = 65534  # fewer than 65535 members (0xFFFF would signal ZIP64)
MAX_FILE_SIZE = 2**32 - 1  # file < 4 GiB, so no field ever reaches the 0xFFFFFFFF ZIP64 sentinel
MAX_MEMBER_SIZE = 2**32 - 2  # uncompressed size must also stay below the sentinel

# --- arrays ----------------------------------------------------------------------------------

DTYPE_WIDTHS: dict[str, int] = {
    "<f8": 8,
    "<f4": 4,
    "<i8": 8,
    "<i4": 4,
    "<i2": 2,
    "|i1": 1,
    "<u8": 8,
    "<u4": 4,
    "<u2": 2,
    "|u1": 1,
}
ENC_RAW = "raw"
ENC_SHUFFLE = "shuffle"
ENC_DELTA_SHUFFLE = "delta+shuffle"
ENCODINGS: tuple[str, ...] = (ENC_RAW, ENC_SHUFFLE, ENC_DELTA_SHUFFLE)
DELTA_DTYPES: tuple[str, ...] = ("<u4", "<f8")
ARRAY_NAMES: tuple[str, ...] = ("t", "v", "idx", "roles")
ROLE_DTYPES: tuple[tuple[int, str], ...] = ((8, "|u1"), (16, "<u2"), (32, "<u4"), (64, "<u8"))
MAX_EXACT_FLOAT_INT = 2**53


@dataclass(slots=True)
class ArrayDesc:
    """Location and encoding of one array inside a data member."""

    name: str
    member: str
    offset: int
    nbytes: int
    n: int
    components: int
    dtype: str
    enc: str

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "member": self.member,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "n": self.n,
            "components": self.components,
            "dtype": self.dtype,
            "enc": self.enc,
        }

    @classmethod
    def from_json(cls, obj: object) -> ArrayDesc:
        """Build a descriptor from its index.json form. Raises ValueError on a malformed entry."""
        if not isinstance(obj, dict):
            raise ValueError("array descriptor is not an object")
        missing = [k for k in ("name", "member", "offset", "nbytes", "n", "components", "dtype", "enc") if k not in obj]
        if missing:
            raise ValueError(f"array descriptor is missing {', '.join(missing)}")
        for key in ("name", "member", "dtype", "enc"):
            if not isinstance(obj[key], str):
                raise ValueError(f"array descriptor field {key!r} must be a string")
        for key in ("offset", "nbytes", "n", "components"):
            value = obj[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"array descriptor field {key!r} must be a non-negative integer")
        return cls(
            name=obj["name"],
            member=obj["member"],
            offset=obj["offset"],
            nbytes=obj["nbytes"],
            n=obj["n"],
            components=obj["components"],
            dtype=obj["dtype"],
            enc=obj["enc"],
        )


def check_desc(desc: ArrayDesc) -> int:
    """Validate a descriptor's dtype, encoding and byte count. Returns the element width."""
    width = DTYPE_WIDTHS.get(desc.dtype)
    if width is None:
        raise ValueError(f"unsupported dtype {desc.dtype!r}; expected one of {', '.join(DTYPE_WIDTHS)}")
    if desc.enc not in ENCODINGS:
        raise ValueError(f"unsupported encoding {desc.enc!r}; expected one of {', '.join(ENCODINGS)}")
    if desc.n < 0 or desc.offset < 0:
        raise ValueError("n and offset must be non-negative")
    if desc.components < 1:
        raise ValueError("components must be at least 1")
    if desc.enc == ENC_DELTA_SHUFFLE:
        if desc.dtype not in DELTA_DTYPES:
            raise ValueError(f"delta+shuffle needs dtype <u4 or <f8, got {desc.dtype!r}")
        if desc.components != 1:
            raise ValueError("delta+shuffle arrays must have exactly one component")
    expected = desc.n * desc.components * width
    if desc.nbytes != expected:
        raise ValueError(
            f"nbytes {desc.nbytes} does not equal n*components*width = "
            f"{desc.n}*{desc.components}*{width} = {expected}"
        )
    return width


def check_member_name(name: str) -> None:
    """Raise ValueError unless `name` is a valid profile member name."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"member name {name!r} must match [a-z0-9._/]+")
    if name.startswith("/") or name.endswith("/"):
        raise ValueError(f"member name {name!r} must not start or end with '/' (no directory entries)")
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"member name {name!r} has an empty, '.' or '..' path segment")


def idx_dtype(n_source: int) -> str:
    """dtype of the `idx` array: `<u4` when n_source < 2**32, else `<f8` holding exact integers."""
    return "<u4" if n_source < 2**32 else "<f8"


def roles_dtype(n_bits: int) -> str:
    """Narrowest unsigned dtype holding a role legend with `n_bits` entries."""
    for bits, code in ROLE_DTYPES:
        if n_bits <= bits:
            return code
    raise ValueError(f"a role legend holds at most 64 entries, got {n_bits}")


def zstd_available() -> bool:
    """True when this interpreter ships `compression.zstd` (Python 3.14+ built with zstd)."""
    try:
        import compression.zstd  # noqa: F401
    except ImportError:
        return False
    return True


# --- JSON ------------------------------------------------------------------------------------


def _json_safe(obj: object) -> object:
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if type(obj).__module__ == "numpy":
        if hasattr(obj, "tolist"):
            return _json_safe(obj.tolist())
    return obj


def _unencodable_path(obj: object, path: str = "$") -> str | None:
    """JSON path of the first string (key or value) that is not valid Unicode, or None."""
    if isinstance(obj, str):
        try:
            obj.encode("utf-8")
        except UnicodeEncodeError:
            return path
        return None
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and _unencodable_path(key) is not None:
                return f"{path} key {key!r}"
            found = _unencodable_path(value, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found = _unencodable_path(value, f"{path}[{i}]")
            if found is not None:
                return found
    return None


def canonical_json(obj: object) -> bytes:
    """UTF-8 JSON with sorted keys, compact separators and non-finite floats written as strings.

    Raises ContainerError when a string cannot be written as UTF-8 (for example a lone surrogate from
    an undecodable file name).
    """
    safe = _json_safe(obj)
    text = json.dumps(
        safe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        from ..errors import ContainerError

        where = _unencodable_path(safe) or "$"
        bad = exc.object[exc.start : exc.end]
        raise ContainerError(
            f"cannot write JSON: the string at {where} is not valid Unicode (contains {bad!r}, "
            "typically an undecodable file name); rename it to valid UTF-8"
        ) from None


def header_dict(codec: str = "deflate", level: int = DEFAULT_LEVEL) -> dict:
    """Content of `baslt.json`."""
    if codec not in CODEC_METHODS:
        raise ValueError(f"unknown codec {codec!r}; expected one of {', '.join(CODEC_METHODS)}")
    return {
        "byte_order": BYTE_ORDER,
        "codec": codec,
        "container": CONTAINER_VERSION,
        "format": FORMAT,
        "level": level,
        "reader_min": READER_MIN,
    }


def header_json(codec: str = "deflate", level: int = DEFAULT_LEVEL) -> bytes:
    """Canonical bytes of `baslt.json`."""
    return canonical_json(header_dict(codec, level))
