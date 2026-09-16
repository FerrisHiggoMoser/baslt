"""Decode an artifact, change its arrays or JSON, and write it back with valid CRCs and sizes.

Tamper tests use this to change what an artifact claims without tripping the container's own integrity checks, so
the verifier's contract checks are what must catch the change.
"""

from __future__ import annotations

import copy
import io
import zipfile

import numpy as np

from baslt.container.encode import container_dtype, encode_array
from baslt.container.reader import read_artifact
from baslt.container.spec import canonical_json
from baslt.container.zipwriter import Member, write_zip


def account(manifest: dict, index: dict, sizes: dict[str, int], size: int) -> None:
    """Set the manifest's budget bytes to what a file with these member sizes holds (docs/container.md)."""
    budget = manifest.get("budget")
    if not isinstance(budget, dict):
        return
    owner: dict[str, str] = {}
    owned: dict[str, list[str]] = {}
    for entry in index["signals"]:
        mine = owned.setdefault(entry["name"], [])
        for descriptor in entry["arrays"]:
            member = descriptor["member"]
            if owner.setdefault(member, entry["name"]) == entry["name"] and member not in mine:
                mine.append(member)
    for row in budget.get("signals", []):
        row["bytes"] = sum(sizes.get(member, 0) for member in owned.get(row["name"], []))
    data = sum(value for name, value in sizes.items() if name.startswith(("s/", "t/")))
    budget["required_bytes"] = data - budget.get("discretionary_bytes", 0)
    budget["overhead_bytes"] = size - data


def settle(members, manifest: dict, index: dict | None = None) -> bytes:
    """Write `members()` until `size_bytes` (and, given `index`, the budget bytes) describe the written file."""
    previous = None
    for _ in range(16):
        data = write_zip(members())
        if data == previous:
            return data
        if index is not None:
            sizes = {info.filename: info.compress_size for info in zipfile.ZipFile(io.BytesIO(data)).infolist()}
            account(manifest, index, sizes, len(data))
        manifest["artifact"]["size_bytes"] = f"{len(data):020d}"
        previous = data
    raise AssertionError("the artifact size did not settle")


def rebuild(data, *, index=None, manifest=None, policy=None, arrays=None, budget=True) -> bytes:
    """Apply the edits and write the artifact back. With `budget`, the byte accounting follows the new sizes;
    pass False to keep the manifest's budget exactly as edited."""
    artifact = read_artifact(data)
    index_obj = copy.deepcopy(artifact.index)
    manifest_obj = copy.deepcopy(artifact.manifest)
    policy_obj = copy.deepcopy(artifact.policy)
    decoded = {
        entry["name"]: {d["name"]: artifact.array(entry["name"], d["name"]) for d in entry["arrays"]}
        for entry in index_obj["signals"]
    }
    for edit, target in ((arrays, decoded), (index, index_obj), (manifest, manifest_obj), (policy, policy_obj)):
        if edit is not None:
            edit(target)

    payloads: dict[str, bytearray] = {}
    clocks: dict[tuple[str, bytes], int] = {}  # a shared clock is written once and pointed at by every signal
    for entry in index_obj["signals"]:
        for descriptor in entry["arrays"]:
            array = decoded[entry["name"]][descriptor["name"]]
            if descriptor["enc"] == "delta+shuffle":
                array = array.astype(descriptor["dtype"])
            else:
                descriptor["dtype"] = container_dtype(array.dtype)
            raw = encode_array(array, descriptor["enc"])
            buffer = payloads.setdefault(descriptor["member"], bytearray())
            key = (descriptor["member"], raw)
            if descriptor["name"] == "t" and key in clocks:
                descriptor["offset"] = clocks[key]
            else:
                descriptor["offset"] = len(buffer)
                buffer += raw
                if descriptor["name"] == "t":
                    clocks[key] = descriptor["offset"]
            descriptor["nbytes"] = len(raw)
            descriptor["n"] = int(array.shape[0])

    def members():
        parts = [
            Member("baslt.json", canonical_json(artifact.header), 0),
            Member("index.json", canonical_json(index_obj), 8),
            Member("manifest.json", canonical_json(manifest_obj), 0),
            Member("policy.json", canonical_json(policy_obj), 8),
        ]
        for entry in artifact.entries[4:]:
            parts.append(Member(entry.name, bytes(payloads.get(entry.name, artifact.members[entry.name])), 8))
        return parts

    return settle(members, manifest_obj, index_obj if budget else None)


def position(decoded_signal, source_index: int) -> int:
    return int(np.flatnonzero(decoded_signal["idx"].astype(np.int64) == source_index)[0])
