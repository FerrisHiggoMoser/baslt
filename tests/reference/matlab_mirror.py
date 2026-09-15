"""Independent `.baslt` decoder that follows the MATLAB load.m and JavaScript reader algorithms.

It never uses zipfile or any baslt module: the end of central directory is taken from the last 22
bytes, the central directory is walked with struct, member data starts at local offset + 30 + name
length, deflate is inflated raw, shuffle is undone by reshaping byte planes column-major, and
delta arrays are rebuilt by a float64 cumulative sum.
"""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np

WIDTHS = {"<f8": 8, "<f4": 4, "<i8": 8, "<i4": 4, "<i2": 2, "|i1": 1, "<u8": 8, "<u4": 4, "<u2": 2, "|u1": 1}
# MATLAB typecast classes for each dtype
MATLAB_CLASS = {
    "<f8": "double",
    "<f4": "single",
    "<i8": "int64",
    "<i4": "int32",
    "<i2": "int16",
    "|i1": "int8",
    "<u8": "uint64",
    "<u4": "uint32",
    "<u2": "uint16",
    "|u1": "uint8",
}
_TYPECAST = {
    "double": np.float64,
    "single": np.float32,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint64": np.uint64,
    "uint32": np.uint32,
    "uint16": np.uint16,
    "uint8": np.uint8,
}


class MirrorError(Exception):
    """The mirror decoder could not decode the file."""


def _u16(data: bytes, pos: int) -> int:
    return data[pos] | (data[pos + 1] << 8)


def _u32(data: bytes, pos: int) -> int:
    return struct.unpack_from("<I", data, pos)[0]


def read_members(data: bytes) -> dict[str, bytes]:
    """Return every member's uncompressed bytes, in central directory order."""
    size = len(data)
    eocd = size - 22
    if eocd < 0 or data[eocd : eocd + 4] != b"PK\x05\x06":
        raise MirrorError("no end of central directory in the last 22 bytes")
    entries = _u16(data, eocd + 10)
    cd_offset = _u32(data, eocd + 16)
    members: dict[str, bytes] = {}
    pos = cd_offset
    for _ in range(entries):
        if data[pos : pos + 4] != b"PK\x01\x02":
            raise MirrorError("bad central directory signature")
        method = _u16(data, pos + 10)
        crc = _u32(data, pos + 16)
        csize = _u32(data, pos + 20)
        usize = _u32(data, pos + 24)
        name_len = _u16(data, pos + 28)
        extra_len = _u16(data, pos + 30)
        comment_len = _u16(data, pos + 32)
        local = _u32(data, pos + 42)
        name = data[pos + 46 : pos + 46 + name_len].decode("ascii")
        pos += 46 + name_len + extra_len + comment_len
        if data[local : local + 4] != b"PK\x03\x04":
            raise MirrorError(f"bad local header signature for {name}")
        start = local + 30 + name_len  # the profile guarantees a zero-length local extra field
        cdata = data[start : start + csize]
        if method == 0:
            raw = cdata
        elif method == 8:
            # MATLAB: InflaterOutputStream(Inflater(true)) fed [cdata; 0]
            inflater = zlib.decompressobj(-15)
            raw = inflater.decompress(cdata + b"\x00") + inflater.flush()
            if not inflater.eof or inflater.unused_data != b"\x00":
                raise MirrorError(f"deflate stream of {name} does not end exactly at its compressed size")
        else:
            raise MirrorError(f"method {method} of {name} is not decodable by load.m")
        if len(raw) != usize:
            raise MirrorError(f"size mismatch for {name}")
        if zlib.crc32(raw) & 0xFFFFFFFF != crc:
            raise MirrorError(f"CRC-32 mismatch for {name}")
        members[name] = raw
    return members


def parse_json(raw: bytes) -> object:
    """jsondecode equivalent that refuses NaN/Infinity tokens."""

    def reject(token: str) -> object:
        raise MirrorError(f"JSON token {token} is not allowed")

    return json.loads(raw.decode("utf-8"), parse_constant=reject)


def unshuffle(planes_bytes: np.ndarray, count: int, width: int) -> np.ndarray:
    """MATLAB: raw = reshape(reshape(b, N, w).', [], 1)."""
    planes = np.reshape(planes_bytes, (count, width), order="F")
    return np.reshape(planes.T, (count * width,), order="F")


def element_bytes(member: bytes, desc: dict) -> np.ndarray:
    """Little-endian element bytes (unshuffled) of one array, as uint8."""
    width = WIDTHS[desc["dtype"]]
    count = desc["n"] * desc["components"]
    if desc["nbytes"] != count * width:
        raise MirrorError("nbytes does not match n*components*width")
    b = np.frombuffer(member, dtype=np.uint8)[desc["offset"] : desc["offset"] + desc["nbytes"]]
    if b.shape[0] != desc["nbytes"]:
        raise MirrorError("array lies outside its member")
    if desc["enc"] == "raw" or count == 0:
        return b
    if desc["enc"] in ("shuffle", "delta+shuffle"):
        return unshuffle(b, count, width)
    raise MirrorError(f"unknown encoding {desc['enc']}")


def decode_member(member: bytes, desc: dict) -> np.ndarray:
    """Decode one array descriptor. Shape (n,) for one component, else (n, k) column-major."""
    raw = element_bytes(member, desc)
    values = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=_TYPECAST[MATLAB_CLASS[desc["dtype"]]])
    if desc["enc"] == "delta+shuffle":
        values = np.cumsum(values.astype(np.float64))
    n, k = desc["n"], desc["components"]
    if k == 1:
        return values.reshape(n)
    return np.reshape(values, (n, k), order="F")


def uint32_pairs(member: bytes, desc: dict) -> tuple[np.ndarray, np.ndarray]:
    """JavaScript view of an 8-byte integer array: (low, high) uint32 words per element."""
    if WIDTHS[desc["dtype"]] != 8:
        raise MirrorError("uint32 pairs are only for 8-byte dtypes")
    raw = np.ascontiguousarray(element_bytes(member, desc)).tobytes()
    words = np.frombuffer(raw, dtype="<u4").reshape(-1, 2)
    return words[:, 0].copy(), words[:, 1].copy()


def combine_pairs(low: np.ndarray, high: np.ndarray, *, signed: bool) -> list[int]:
    """Exact integers from (low, high) words using Python integers only."""
    out = []
    for lo, hi in zip(low.tolist(), high.tolist()):
        value = lo + hi * 4294967296
        if signed and hi >= 2147483648:
            value -= 18446744073709551616
        out.append(value)
    return out


def load(data: bytes) -> dict:
    """load.m equivalent: JSON members plus every decoded array keyed by signal and array name."""
    if data[30:40] != b"baslt.json" or _u16(data, 8) != 0:
        raise MirrorError("byte 30 does not start a STORED baslt.json member")
    members = read_members(data)
    header = parse_json(members["baslt.json"])
    index = parse_json(members["index.json"])
    out = {
        "header": header,
        "index": index,
        "manifest": parse_json(members["manifest.json"]),
        "policy": parse_json(members["policy.json"]),
        "signals": {},
    }
    for sig in index["signals"]:
        out["signals"][sig["name"]] = {
            desc["name"]: decode_member(members[desc["member"]], desc) for desc in sig["arrays"]
        }
    return out


def check_uniform_keys(obj: object, path: str = "$") -> None:
    """Assert every list of objects in a JSON tree holds objects with identical key sets, at every level."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            check_uniform_keys(value, f"{path}.{key}")
    elif isinstance(obj, list):
        if any(isinstance(item, dict) for item in obj):
            if not all(isinstance(item, dict) for item in obj):
                raise AssertionError(f"{path}: list mixes objects and non-objects")
            keys = set(obj[0])
            for i, item in enumerate(obj):
                if set(item) != keys:
                    raise AssertionError(
                        f"{path}[{i}]: keys {sorted(item)} differ from {path}[0] keys {sorted(keys)}"
                    )
        for i, item in enumerate(obj):
            check_uniform_keys(item, f"{path}[{i}]")
