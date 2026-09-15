"""Source digests recorded in the manifest.

Four modes:

    full     streamed sha256 of the whole file
    sampled  sha256 of a fixed, reproducible selection of byte ranges (cheap on multi-GB files)
    arrays   sha256 over names, dtypes, shapes and bytes of in-memory arrays
    none     no digest

A sampled digest is a fingerprint, not a proof: it never counts as a sha256 match. `baslt verify`
reports it as such.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

CHUNK = 4 * 1024 * 1024
SAMPLED_PREFIX = b"baslt-sampled-v1"
SAMPLED_EDGE = 4 * 1024 * 1024
SAMPLED_BLOCK = 256 * 1024
SAMPLED_BLOCKS = 32

MODES = ("full", "sampled", "arrays", "none")


@dataclass(slots=True)
class HashInfo:
    algorithm: str
    mode: str
    covered_bytes: int
    value: str

    def to_json(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "mode": self.mode,
            "covered_bytes": int(self.covered_bytes),
            "value": self.value,
        }

    @classmethod
    def from_json(cls, obj: Mapping) -> HashInfo:
        return cls(
            algorithm=str(obj.get("algorithm", "")),
            mode=str(obj.get("mode", "none")),
            covered_bytes=int(obj.get("covered_bytes", 0)),
            value=str(obj.get("value", "")),
        )

    @property
    def is_proof(self) -> bool:
        """True when the digest covers every byte of the source."""
        return self.mode in ("full", "arrays")


def sampled_ranges(size: int) -> list[tuple[int, int]]:
    """The byte ranges a sampled digest covers, merged and in ascending order."""
    if size <= 0:
        return []
    raw: list[tuple[int, int]] = [(0, min(SAMPLED_EDGE, size))]
    raw.append((max(0, size - SAMPLED_EDGE), size))
    span = max(0, size - SAMPLED_BLOCK)
    for i in range(SAMPLED_BLOCKS):
        start = (i * span) // (SAMPLED_BLOCKS - 1) if SAMPLED_BLOCKS > 1 else 0
        raw.append((start, min(start + SAMPLED_BLOCK, size)))
    raw.sort()
    merged: list[tuple[int, int]] = []
    for start, stop in raw:
        if stop <= start:
            continue
        if merged and start <= merged[-1][1]:
            prev_start, prev_stop = merged[-1]
            merged[-1] = (prev_start, max(prev_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def hash_file(path: str | Path, mode: str = "full") -> HashInfo:
    """Digest a file in the given mode."""
    path = Path(path)
    if mode == "none":
        return hash_none()
    if mode not in ("full", "sampled"):
        raise ValueError(f"unknown file hash mode {mode!r}")
    size = path.stat().st_size
    digest = hashlib.sha256()
    if mode == "full":
        with path.open("rb") as handle:
            while True:
                block = handle.read(CHUNK)
                if not block:
                    break
                digest.update(block)
        return HashInfo("sha256", "full", size, digest.hexdigest())

    ranges = sampled_ranges(size)
    digest.update(SAMPLED_PREFIX)
    digest.update(struct.pack("<Q", size))
    covered = 0
    with path.open("rb") as handle:
        for start, stop in ranges:
            handle.seek(start)
            remaining = stop - start
            while remaining > 0:
                block = handle.read(min(CHUNK, remaining))
                if not block:
                    break
                digest.update(block)
                covered += len(block)
                remaining -= len(block)
    return HashInfo("sha256-sampled-v1", "sampled", covered, digest.hexdigest())


def hash_arrays(arrays: Mapping[str, np.ndarray]) -> HashInfo:
    """Digest in-memory arrays: name, dtype, shape and bytes, in sorted name order."""
    import numpy as np

    digest = hashlib.sha256()
    digest.update(b"baslt-arrays-v1")
    covered = 0
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(arr.dtype.str.encode("ascii"))
        digest.update(struct.pack("<I", arr.ndim))
        for dim in arr.shape:
            digest.update(struct.pack("<Q", int(dim)))
        view = arr.view(np.uint8).reshape(-1) if arr.size else arr
        digest.update(view.tobytes())
        covered += int(arr.nbytes)
    return HashInfo("sha256-arrays-v1", "arrays", covered, digest.hexdigest())


def hash_none() -> HashInfo:
    return HashInfo("none", "none", 0, "")


def recompute(source: str | Path | Mapping, info: HashInfo) -> HashInfo:
    """Recompute a digest in the mode recorded in `info`."""
    if info.mode == "none":
        return hash_none()
    if info.mode == "arrays":
        if isinstance(source, (str, Path)):
            raise ValueError("an arrays digest cannot be recomputed from a file path")
        return hash_arrays(source)
    if isinstance(source, (str, Path)):
        return hash_file(source, info.mode)
    raise ValueError(f"cannot recompute a {info.mode!r} digest from in-memory arrays")


def matches(source: str | Path | Mapping, info: HashInfo) -> bool:
    """True when the source still digests to the recorded value."""
    if info.mode == "none":
        return True
    return recompute(source, info).value == info.value


def default_mode(source: str | Path | Mapping) -> str:
    """Files default to a full sha256; in-memory arrays to an arrays digest."""
    return "full" if isinstance(source, (str, Path)) else "arrays"
