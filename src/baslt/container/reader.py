"""Reader for `.baslt` files: explicit profile v1 checks, a stdlib zipfile cross-check, strict decompression."""

from __future__ import annotations

import io
import json
import math
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ContainerError
from . import spec
from .spec import (
    CENTRAL_HEADER_FORMAT,
    CENTRAL_HEADER_SIZE,
    CENTRAL_SIGNATURE,
    EOCD_FORMAT,
    EOCD_SIGNATURE,
    EOCD_SIZE,
    FORMAT,
    JSON_MEMBERS,
    LOCAL_HEADER_FORMAT,
    LOCAL_HEADER_SIZE,
    LOCAL_SIGNATURE,
    MAX_MEMBERS,
    MEMBER_HEADER,
    MEMBER_INDEX,
    MEMBER_MANIFEST,
    MEMBER_POLICY,
    METHOD_DEFLATE,
    METHOD_STORE,
    METHOD_ZSTD,
    METHODS,
    SUPPORTED_CONTAINER,
    VERSION_NEEDED,
    VERSION_NEEDED_ZSTD,
    ArrayDesc,
    check_desc,
    check_member_name,
)

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Artifact", "MemberEntry", "read_artifact"]


@dataclass(slots=True)
class MemberEntry:
    """Central directory facts about one member."""

    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    offset: int  # local header offset


@dataclass
class Artifact:
    """A parsed and profile-checked `.baslt` file."""

    header: dict
    index: dict
    manifest: dict
    policy: dict
    size: int
    entries: list[MemberEntry] = field(default_factory=list, repr=False)
    members: dict[str, bytes] = field(default_factory=dict, repr=False)

    def signal_names(self) -> list[str]:
        return [s["name"] for s in self.index.get("signals", [])]

    def signal(self, signal: str) -> dict:
        """The index.json entry of one signal."""
        for entry in self.index.get("signals", []):
            if entry["name"] == signal:
                return entry
        names = ", ".join(self.signal_names()) or "none"
        raise ContainerError(f"no signal {signal!r} in artifact (signals: {names})")

    def descriptor(self, signal: str, name: str) -> ArrayDesc:
        entry = self.signal(signal)
        for obj in entry["arrays"]:
            if obj["name"] == name:
                return ArrayDesc.from_json(obj)
        names = ", ".join(a["name"] for a in entry["arrays"]) or "none"
        raise ContainerError(f"signal {signal!r} has no array {name!r} (arrays: {names})")

    def array(self, signal: str, name: str) -> np.ndarray:
        """Decode array `name` ("t", "v", "idx", "roles") of `signal`."""
        from .encode import decode_array

        desc = self.descriptor(signal, name)
        try:
            return decode_array(self.members[desc.member], desc)
        except (KeyError, ValueError) as exc:
            raise ContainerError(f"signal {signal!r} array {name!r}: {exc}") from exc


def _fail(message: str) -> ContainerError:
    return ContainerError(f"not a valid .baslt file: {message}")


def _load_bytes(source: str | Path | bytes | bytearray | memoryview) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), "<bytes>"
    path = Path(source)
    try:
        return path.read_bytes(), str(path)
    except OSError as exc:
        raise ContainerError(f"cannot read {path}: {exc.strerror or exc}") from exc


def _leading_header(data: bytes) -> dict | None:
    """The baslt.json object stored at byte 0, read before any other structure check, or None.

    Only the local header at offset 0 is used, so a file from a newer container version is identified
    (and refused with an upgrade message) even when it uses ZIP features outside profile v1.
    """
    if len(data) < LOCAL_HEADER_SIZE:
        return None
    (sig, _needed, _flags, method, _time, _date, crc, csize, usize, name_len,
     extra_len) = struct.unpack_from(LOCAL_HEADER_FORMAT, data, 0)
    if sig != LOCAL_SIGNATURE or method != METHOD_STORE or csize != usize:
        return None
    if data[LOCAL_HEADER_SIZE : LOCAL_HEADER_SIZE + name_len] != MEMBER_HEADER.encode("ascii"):
        return None
    start = LOCAL_HEADER_SIZE + name_len + extra_len
    raw = data[start : start + csize]
    if len(raw) != csize or zlib.crc32(raw) & 0xFFFFFFFF != crc:
        return None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) and obj.get("format") == FORMAT else None


def _parse_directory(data: bytes) -> list[MemberEntry]:
    size = len(data)
    if size < EOCD_SIZE:
        raise _fail(f"file is {size} bytes, shorter than the 22-byte end of central directory")
    eocd_at = size - EOCD_SIZE
    sig, disk, cd_disk, n_disk, n_total, cd_size, cd_offset, comment_len = struct.unpack_from(
        EOCD_FORMAT, data, eocd_at
    )
    if sig != EOCD_SIGNATURE:
        raise _fail("end of central directory is not at the last 22 bytes (truncated or has a comment)")
    if disk != 0 or cd_disk != 0 or n_disk != n_total:
        raise _fail("multi-disk archives are not allowed")
    if comment_len != 0:
        raise _fail("archive comments are not allowed")
    if n_total == 0:
        raise _fail("archive has no members")
    if n_total > MAX_MEMBERS:
        raise _fail(f"archive has {n_total} members; at most {MAX_MEMBERS} are allowed")
    if cd_offset + cd_size != eocd_at:
        raise _fail("central directory does not end where the end of central directory begins")

    entries: list[MemberEntry] = []
    seen: set[str] = set()
    pos = cd_offset
    expected_offset = 0
    for i in range(n_total):
        if pos + CENTRAL_HEADER_SIZE > eocd_at:
            raise _fail(f"central directory entry {i} runs past the directory")
        (sig, _made_by, needed, flags, method, _time, _date, crc, csize, usize, name_len, extra_len,
         comment_len, _disk_start, _internal, _external, offset) = struct.unpack_from(
            CENTRAL_HEADER_FORMAT, data, pos
        )
        if sig != CENTRAL_SIGNATURE:
            raise _fail(f"bad central directory signature for entry {i}")
        raw_name = data[pos + CENTRAL_HEADER_SIZE : pos + CENTRAL_HEADER_SIZE + name_len]
        pos += CENTRAL_HEADER_SIZE + name_len + extra_len + comment_len
        if pos > eocd_at:
            raise _fail(f"central directory entry {i} runs past the directory")
        try:
            name = raw_name.decode("ascii")
            check_member_name(name)
        except (UnicodeDecodeError, ValueError) as exc:
            raise _fail(f"member {i}: invalid name {raw_name!r}: {exc}") from None
        where = f"member {name!r}"
        if name in seen:
            raise _fail(f"duplicate member name {name!r}")
        seen.add(name)
        if flags != 0:
            raise _fail(f"{where}: general purpose flags must be 0, got {flags:#06x}")
        if extra_len != 0:
            raise _fail(f"{where}: central directory extra field length must be 0, got {extra_len}")
        if comment_len != 0:
            raise _fail(f"{where}: member comments are not allowed")
        _check_method(name, method)
        _check_version_needed(f"{where}: central directory", method, needed)
        if method == METHOD_STORE and csize != usize:
            raise _fail(f"{where}: STORED member has compressed size {csize} != uncompressed size {usize}")
        if offset != expected_offset:
            raise _fail(f"{where}: local header at offset {offset}, expected {expected_offset}")
        _check_local_header(data, name, method, crc, csize, usize, offset, cd_offset)
        expected_offset = offset + LOCAL_HEADER_SIZE + name_len + csize
        entries.append(MemberEntry(name, method, crc, csize, usize, offset))
    if pos != eocd_at:
        raise _fail("central directory size does not match its entries")
    if expected_offset != cd_offset:
        raise _fail("member data does not end where the central directory begins")
    return entries


def _check_method(name: str, method: int) -> None:
    if method not in METHODS:
        raise _fail(f"member {name!r}: compression method {method} is not allowed (0, 8 or 93)")
    if method == METHOD_ZSTD and not spec.zstd_available():
        raise ContainerError(
            f"member {name!r} is zstd-compressed (method 93), which needs Python 3.14+ with "
            "compression.zstd; recompile with --codec deflate"
        )


def _check_version_needed(where: str, method: int, needed: int) -> None:
    expected = VERSION_NEEDED_ZSTD if method == METHOD_ZSTD else VERSION_NEEDED
    if needed != expected:
        raise _fail(f"{where} version needed to extract is {needed}, expected {expected} for method {method}")


def _check_local_header(
    data: bytes, name: str, method: int, crc: int, csize: int, usize: int, offset: int, limit: int
) -> None:
    where = f"member {name!r}"
    if offset + LOCAL_HEADER_SIZE > limit:
        raise _fail(f"{where}: local header lies outside the member area")
    (sig, needed, flags, l_method, _time, _date, l_crc, l_csize, l_usize, name_len,
     extra_len) = struct.unpack_from(LOCAL_HEADER_FORMAT, data, offset)
    if sig != LOCAL_SIGNATURE:
        raise _fail(f"{where}: bad local header signature")
    if flags != 0:
        raise _fail(f"{where}: local header flags must be 0, got {flags:#06x}")
    if extra_len != 0:
        raise _fail(f"{where}: local header extra field length must be 0, got {extra_len}")
    local_name = data[offset + LOCAL_HEADER_SIZE : offset + LOCAL_HEADER_SIZE + name_len]
    if local_name != name.encode("ascii"):
        raise _fail(f"{where}: local header name {local_name!r} differs from the central directory")
    if l_method != method:
        raise _fail(f"{where}: local header method {l_method} differs from central directory method {method}")
    _check_version_needed(f"{where}: local header", method, needed)
    if (l_crc, l_csize, l_usize) != (crc, csize, usize):
        raise _fail(f"{where}: local header CRC-32 or sizes differ from the central directory")
    if offset + LOCAL_HEADER_SIZE + name_len + csize > limit:
        raise _fail(f"{where}: data runs past the central directory")


def _decompress(where: str, method: int, payload: bytes, usize: int) -> bytes:
    """Decode exactly `payload` (the member's compressed bytes), as MATLAB and browser readers do.

    The compressed stream must end exactly at the compressed size: bytes after the end of the
    stream, a truncated stream, or output beyond the uncompressed size are errors.
    """
    if method == METHOD_STORE:
        return payload
    if method == METHOD_DEFLATE:
        codec = "deflate"
        decoder = zlib.decompressobj(-15)
        errors: tuple[type[BaseException], ...] = (zlib.error,)
    else:
        from compression import zstd

        codec = "zstd"
        decoder = zstd.ZstdDecompressor()
        errors = (zstd.ZstdError,)
    try:
        content = decoder.decompress(payload, usize + 1)  # at most one byte beyond the expected size
    except (*errors, ValueError, EOFError, MemoryError) as exc:
        raise _fail(f"{where}: cannot decompress: {exc}") from exc
    if len(content) > usize:
        raise _fail(f"{where}: {codec} data decompresses to more than {usize} bytes")
    if not decoder.eof:
        raise _fail(f"{where}: {codec} stream is truncated")
    if decoder.unused_data or getattr(decoder, "unconsumed_tail", b""):
        raise _fail(f"{where}: {codec} stream does not end at its compressed size")
    return content


def _read_members(data: bytes, entries: list[MemberEntry]) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, NotImplementedError) as exc:
        raise _fail(f"zip structure rejected: {exc}") from exc
    if names != [e.name for e in entries]:
        raise _fail("zipfile member list differs from the central directory")
    members: dict[str, bytes] = {}
    for entry in entries:
        where = f"member {entry.name!r}"
        start = entry.offset + LOCAL_HEADER_SIZE + len(entry.name)  # local extra length is checked to be 0
        payload = data[start : start + entry.compressed_size]
        content = _decompress(where, entry.method, payload, entry.uncompressed_size)
        if len(content) != entry.uncompressed_size:
            raise _fail(f"{where}: decompressed to {len(content)} bytes, expected {entry.uncompressed_size}")
        if zlib.crc32(content) & 0xFFFFFFFF != entry.crc32:
            raise _fail(f"{where}: CRC-32 mismatch")
        members[entry.name] = content
    return members


def _reject_constant(token: str) -> object:
    raise ValueError(f"non-standard JSON token {token}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"number {token} is not finite")
    return value


def _parse_json(members: dict[str, bytes], name: str) -> dict:
    if name not in members:
        raise _fail(f"required member {name!r} is missing")
    try:
        obj = json.loads(members[name].decode("utf-8"), parse_constant=_reject_constant, parse_float=_finite_float)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise _fail(f"member {name!r} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise _fail(f"member {name!r} must hold a JSON object")
    return obj


def _check_version(header: dict, key: str) -> int:
    value = header.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise _fail(f"baslt.json field {key!r} must be a positive integer, got {value!r}")
    return value


def _check_header(header: dict) -> None:
    if header.get("format") != FORMAT:
        raise _fail(f"baslt.json format is {header.get('format')!r}, expected {FORMAT!r}")
    container = _check_version(header, "container")
    reader_min = _check_version(header, "reader_min")
    if container > SUPPORTED_CONTAINER:
        raise ContainerError(
            f"artifact uses container version {container}, but this reader supports up to "
            f"{SUPPORTED_CONTAINER}; upgrade baslt"
        )
    if reader_min > SUPPORTED_CONTAINER:
        raise ContainerError(
            f"artifact needs reader version {reader_min} or newer, but this reader is version "
            f"{SUPPORTED_CONTAINER}; upgrade baslt"
        )
    if "byte_order" in header and header["byte_order"] != "little":
        raise _fail(f"baslt.json byte_order is {header['byte_order']!r}, expected 'little'")


def _check_index(index: dict, sizes: dict[str, int]) -> None:
    signals = index.get("signals")
    if not isinstance(signals, list):
        raise _fail("index.json must have a 'signals' list")
    container = index.get("container", 1)
    if not isinstance(container, int) or isinstance(container, bool) or container > SUPPORTED_CONTAINER:
        raise _fail(f"index.json container {container!r} is not supported")
    seen: set[str] = set()
    for i, sig in enumerate(signals):
        if not isinstance(sig, dict) or not isinstance(sig.get("name"), str):
            raise _fail(f"index.json signals[{i}] must be an object with a string 'name'")
        sname = sig["name"]
        if sname in seen:
            raise _fail(f"index.json lists signal {sname!r} twice")
        seen.add(sname)
        arrays = sig.get("arrays")
        if not isinstance(arrays, list):
            raise _fail(f"signal {sname!r} must have an 'arrays' list")
        names: set[str] = set()
        for j, obj in enumerate(arrays):
            where = f"signal {sname!r} arrays[{j}]"
            try:
                desc = ArrayDesc.from_json(obj)
                check_desc(desc)
            except ValueError as exc:
                raise _fail(f"{where}: {exc}") from exc
            where = f"signal {sname!r} array {desc.name!r}"
            if desc.name in names:
                raise _fail(f"signal {sname!r} lists array {desc.name!r} twice")
            names.add(desc.name)
            if desc.member in JSON_MEMBERS or desc.member not in sizes:
                raise _fail(f"{where}: member {desc.member!r} is not a data member of this file")
            if desc.offset + desc.nbytes > sizes[desc.member]:
                raise _fail(
                    f"{where}: bytes {desc.offset}..{desc.offset + desc.nbytes} lie outside member "
                    f"{desc.member!r} ({sizes[desc.member]} bytes)"
                )


def read_artifact(source: str | Path | bytes) -> Artifact:
    """Read and check a `.baslt` file from a path or from bytes. Raises ContainerError on any problem.

    The header versions are checked first, from the STORED baslt.json at byte 0, so a newer file gets
    an upgrade message instead of a profile violation; only then is the ZIP profile checked and every
    member decompressed.
    """
    data, _label = _load_bytes(source)
    leading = _leading_header(data)
    if leading is not None:
        _check_header(leading)
    entries = _parse_directory(data)
    if entries[0].name != MEMBER_HEADER:
        raise _fail(f"first member is {entries[0].name!r}, expected {MEMBER_HEADER!r}")
    if entries[0].method != METHOD_STORE:
        raise _fail(f"{MEMBER_HEADER} must be STORED, found method {entries[0].method}")
    members = _read_members(data, entries)
    header = _parse_json(members, MEMBER_HEADER)
    _check_header(header)
    index = _parse_json(members, MEMBER_INDEX)
    manifest = _parse_json(members, MEMBER_MANIFEST)
    policy = _parse_json(members, MEMBER_POLICY)
    _check_index(index, {e.name: e.uncompressed_size for e in entries})
    return Artifact(
        header=header,
        index=index,
        manifest=manifest,
        policy=policy,
        size=len(data),
        entries=entries,
        members=members,
    )
