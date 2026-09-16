"""Decode an artifact, change its arrays or JSON, and write it back with valid CRCs and sizes.

Tamper tests use this to change what an artifact claims without tripping the container's own integrity checks, so
the verifier's contract checks are what must catch the change.
"""

from __future__ import annotations

import copy

import numpy as np

from baslt.container.encode import container_dtype, encode_array
from baslt.container.reader import read_artifact
from baslt.container.spec import canonical_json
from baslt.container.zipwriter import Member, write_zip


def rebuild(data, *, index=None, manifest=None, policy=None, arrays=None) -> bytes:
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
    written: dict[tuple[str, int], tuple[int, int]] = {}
    for entry in index_obj["signals"]:
        for descriptor in entry["arrays"]:
            array = decoded[entry["name"]][descriptor["name"]]
            if descriptor["enc"] == "delta+shuffle":
                array = array.astype(descriptor["dtype"])
            else:
                descriptor["dtype"] = container_dtype(array.dtype)
            raw = encode_array(array, descriptor["enc"])
            buffer = payloads.setdefault(descriptor["member"], bytearray())
            descriptor["offset"] = len(buffer)
            descriptor["nbytes"] = len(raw)
            descriptor["n"] = int(array.shape[0])
            buffer += raw
            written[(descriptor["member"], descriptor["offset"])] = (len(raw), 0)

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

    size = len(write_zip(members()))
    manifest_obj["artifact"]["size_bytes"] = f"{size:020d}"
    return write_zip(members())


def position(decoded_signal, source_index: int) -> int:
    return int(np.flatnonzero(decoded_signal["idx"].astype(np.int64) == source_index)[0])
