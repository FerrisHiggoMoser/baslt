"""Deterministic writer for the strict ZIP subset used by `.baslt` files."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..errors import ContainerError
from .spec import (
    ALWAYS_STORED,
    CENTRAL_HEADER_FORMAT,
    CENTRAL_SIGNATURE,
    DEFAULT_LEVEL,
    DOS_DATE,
    DOS_TIME,
    EOCD_FORMAT,
    EOCD_SIGNATURE,
    EOCD_SIZE,
    LOCAL_HEADER_FORMAT,
    LOCAL_HEADER_SIZE,
    LOCAL_SIGNATURE,
    MAX_FILE_SIZE,
    MAX_MEMBER_SIZE,
    MAX_MEMBERS,
    METHOD_DEFLATE,
    METHOD_STORE,
    METHOD_ZSTD,
    VERSION_MADE_BY,
    VERSION_NEEDED,
    VERSION_NEEDED_ZSTD,
    check_member_name,
    zstd_available,
)

__all__ = ["Member", "compress_member", "write_zip", "zip_size"]


@dataclass(slots=True)
class Member:
    """One container member before compression."""

    name: str
    data: bytes  # uncompressed
    method: int  # 0 store, 8 deflate, 93 zstd (downgraded to 0 when compression does not help)


def _zstd_compress(data: bytes, level: int) -> bytes:
    if not zstd_available():
        raise ContainerError(
            "the zstd codec needs Python 3.14+ with compression.zstd; recompile with --codec deflate"
        )
    from compression import zstd

    try:
        return zstd.compress(data, level=level)
    except (ValueError, zstd.ZstdError) as exc:
        raise ContainerError(f"zstd compression failed at level {level}: {exc}") from exc


def compress_member(m: Member, *, level: int = DEFAULT_LEVEL) -> tuple[int, bytes, int]:
    """Compress one member. Returns (method actually used, compressed bytes, CRC-32 of the data).

    `baslt.json` and `manifest.json` are always STORED; any member whose compressed form is not
    smaller than its data is STORED as well. Bytes-like data is taken as its raw bytes.
    """
    return _compress(m.name, bytes(m.data), m.method, level)


def _compress(name: str, data: bytes, method: int, level: int) -> tuple[int, bytes, int]:
    crc = zlib.crc32(data) & 0xFFFFFFFF
    if method not in (METHOD_STORE, METHOD_DEFLATE, METHOD_ZSTD):
        raise ContainerError(f"member {name!r}: unsupported compression method {method}")
    if method == METHOD_STORE or name in ALWAYS_STORED:
        return METHOD_STORE, data, crc
    if method == METHOD_DEFLATE:
        if not isinstance(level, int) or not 0 <= level <= 9:
            raise ContainerError(f"deflate level must be an integer 0..9, got {level!r}")
        compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
        packed = compressor.compress(data) + compressor.flush()
    else:
        packed = _zstd_compress(data, level)
    if len(packed) >= len(data):
        return METHOD_STORE, data, crc
    return method, packed, crc


def write_zip(members: Sequence[Member], *, level: int = DEFAULT_LEVEL) -> bytes:
    """Write members, in the given order, as a profile v1 ZIP file and return its bytes."""
    members = list(members)
    if not members:
        raise ContainerError("a container needs at least one member")
    if len(members) > MAX_MEMBERS:
        raise ContainerError(f"too many members ({len(members)}); the profile allows at most {MAX_MEMBERS}")
    seen: set[str] = set()
    chunks: list[bytes] = []
    central: list[bytes] = []
    offset = 0
    for m in members:
        try:
            check_member_name(m.name)
        except ValueError as exc:
            raise ContainerError(str(exc)) from exc
        if m.name in seen:
            raise ContainerError(f"duplicate member name {m.name!r}")
        seen.add(m.name)
        data = bytes(m.data)  # the byte count of a bytes-like object is not always its len()
        if len(data) > MAX_MEMBER_SIZE:
            raise ContainerError(f"member {m.name!r} is {len(data)} bytes; the profile has no ZIP64")
        method, packed, crc = _compress(m.name, data, m.method, level)
        name = m.name.encode("ascii")
        needed = VERSION_NEEDED_ZSTD if method == METHOD_ZSTD else VERSION_NEEDED
        local = struct.pack(
            LOCAL_HEADER_FORMAT,
            LOCAL_SIGNATURE,
            needed,
            0,  # flags
            method,
            DOS_TIME,
            DOS_DATE,
            crc,
            len(packed),
            len(data),
            len(name),
            0,  # extra length
        )
        entry = struct.pack(
            CENTRAL_HEADER_FORMAT,
            CENTRAL_SIGNATURE,
            VERSION_MADE_BY,
            needed,
            0,  # flags
            method,
            DOS_TIME,
            DOS_DATE,
            crc,
            len(packed),
            len(data),
            len(name),
            0,  # extra length
            0,  # comment length
            0,  # disk number start
            0,  # internal attributes
            0,  # external attributes
            offset,
        )
        chunks.extend((local, name, packed))
        central.append(entry + name)
        offset += LOCAL_HEADER_SIZE + len(name) + len(packed)
        if offset > MAX_FILE_SIZE:
            raise ContainerError("container would reach 4 GiB; the profile has no ZIP64")
    directory = b"".join(central)
    total = offset + len(directory) + EOCD_SIZE
    if total > MAX_FILE_SIZE:
        raise ContainerError("container would reach 4 GiB; the profile has no ZIP64")
    eocd = struct.pack(
        EOCD_FORMAT,
        EOCD_SIGNATURE,
        0,  # this disk
        0,  # central directory disk
        len(members),
        len(members),
        len(directory),
        offset,
        0,  # comment length
    )
    chunks.append(directory)
    chunks.append(eocd)
    return b"".join(chunks)


def zip_size(names_and_csizes: Iterable[tuple[str, int]]) -> int:
    """Exact file size for members with these names and compressed sizes, without writing."""
    return EOCD_SIZE + sum(76 + 2 * len(name) + int(csize) for name, csize in names_and_csizes)
