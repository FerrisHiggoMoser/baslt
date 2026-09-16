"""The `structure.*` checks: a well-formed artifact passes them all, and each tamper fails its own check.

Artifacts are built here with `baslt.container` directly, never with the planner, so the expected bytes are the
ones this test wrote. `rebuild` is the tamper tool docs/verification.md describes: decode an artifact, mutate an
array or a JSON member, re-encode with valid CRCs and sizes -- so every failure below is a contract failure the
verifier found, not a corrupt file the reader tripped over.
"""

from __future__ import annotations

import copy
import json
import math
import struct
import zipfile

import numpy as np
import pytest

from baslt.container.encode import container_dtype, encode_array
from baslt.container.reader import read_artifact
from baslt.container.spec import ArrayDesc, canonical_json, header_json
from baslt.container.zipwriter import Member, write_zip
from baslt.policy import load_policy
from baslt.verify.structure import PASS, rebind, structure_checks

pytestmark = pytest.mark.minimal

N_SOURCE = 11
KEPT = np.array([0, 3, 6, 10], dtype=np.uint32)
T = np.arange(N_SOURCE, dtype=np.float64)
Q = np.array([0.0, 10.0, 20.0, 100.0, 90.0, 30.0, 5.0, 5.0, 60.0, 70.0, 20.0])
ROLES = np.array([0b011, 0b010, 0b100, 0b001], dtype=np.uint16)  # extent on the first and last kept sample

STRUCTURE_IDS = (
    "structure.container",
    "structure.descriptors",
    "structure.invariants",
    "structure.size",
    "structure.policy",
    "structure.bind",
)

POLICY = {
    "version": 1,
    "name": "structure_probe",
    "artifact": {"max_size": "1 MiB"},
    "signals": {"decl": {"q_dyn": {"unit": "Pa"}}},
    "hard": {
        "q_dyn": {
            "global_extrema": {},
            "window_extrema": {"interval": "5 s"},
            "threshold_crossing": [{"value": "50 Pa", "debounce": "100 ms"}],
        }
    },
}

REQUIREMENTS = [
    {"id": "hard.q_dyn.global_extrema", "op": "global_extrema", "bits": [1], "params": []},
    {
        "id": "hard.q_dyn.window_extrema",
        "op": "window_extrema",
        "bits": [2],
        "params": [{"name": "interval", "value": 5.0}, {"name": "origin", "value": 0.0}],
    },
    {
        "id": "hard.q_dyn.threshold_crossing[0]",
        "op": "threshold_crossing",
        "bits": [2],
        "params": [
            {"name": "value", "value": 50.0},
            {"name": "edge", "value": "both"},
            {"name": "hysteresis", "value": 0.0},
            {"name": "debounce", "value": 0.1},
            {"name": "tolerance", "value": 0.0},
            {"name": "interpolate", "value": "linear"},
        ],
    },
]


# --- building ----------------------------------------------------------------------------------------------


def descriptors(member, t, values, idx, roles, idx_enc="delta+shuffle"):
    buffer = bytearray()
    out = []
    for name, array, enc in (("t", t, "shuffle"), ("v", values, "shuffle"), ("idx", idx, idx_enc),
                             ("roles", roles, "shuffle")):
        raw = encode_array(array, enc)
        out.append(
            ArrayDesc(
                name, member, len(buffer), len(raw), int(array.shape[0]),
                1 if array.ndim == 1 else int(array.shape[1]), container_dtype(array.dtype), enc,
            ).to_json()
        )
        buffer += raw
    return out, bytes(buffer)


def build(*, t=None, values=None, idx=None, roles=None, idx_enc="delta+shuffle", policy_dict=None,
          order=None, edit=None, fix_size=True) -> bytes:
    """A valid one-signal artifact, or a deliberately broken one."""
    policy = load_policy(policy_dict or POLICY)
    idx = KEPT if idx is None else idx
    t = T[KEPT] if t is None else t
    values = Q[KEPT] if values is None else values
    roles = ROLES if roles is None else roles
    arrays, payload = descriptors("s/0", t, values, idx, roles, idx_enc)

    index = {
        "container": 1,
        "signals": [
            {
                "name": "q_dyn", "path": "q_dyn", "kind": "continuous", "interp": "linear", "unit": "Pa",
                "labels": [], "n_source": N_SOURCE, "n": int(KEPT.size), "components": 1,
                "arrays": arrays,
                "roles": [
                    {"bit": 0, "id": "extent"},
                    {"bit": 1, "id": "hard.q_dyn.global_extrema"},
                    {"bit": 2, "id": "hard.q_dyn.window_extrema"},
                ],
                "requirements": copy.deepcopy(REQUIREMENTS),
            }
        ],
    }
    manifest = {
        "status": "pass",
        "baslt": {"version": "0.1.0", "container": 1},
        "source": {"path": None, "format": "numpy", "size_bytes": 176,
                   "digest": {"algorithm": "none", "mode": "none", "covered_bytes": 0, "value": ""},
                   "signals": 1, "samples_total": 11, "issues": []},
        "policy": {"name": policy.name, "sha256": policy.sha256},
        "artifact": {"size_bytes": "0" * 20, "max_bytes": 1048576, "ratio": "0.000000e+00",
                     "codec": "deflate", "level": 6},
        "budget": {"source": "policy", "required_bytes": 0, "discretionary_bytes": 0, "overhead_bytes": 0,
                   "signals": []},
        "requirements": [], "events": [], "trajectories": [], "sync_groups": [],
    }
    policy_member = {"canonical": policy.canonical, "sha256": policy.sha256,
                     "source_format": policy.source_format, "source_text": policy.source_text}
    if edit is not None:
        edit(index, manifest, policy_member)

    def members():
        parts = [
            Member("baslt.json", header_json("deflate"), 0),
            Member("index.json", canonical_json(index), 8),
            Member("manifest.json", canonical_json(manifest), 8),
            Member("policy.json", canonical_json(policy_member), 8),
            Member("s/0", payload, 8),
        ]
        return [parts[i] for i in order] if order is not None else parts

    if fix_size:
        size = len(write_zip(members()))
        manifest["artifact"]["size_bytes"] = f"{size:020d}"
        manifest["artifact"]["ratio"] = f"{176 / size:.6e}"
    return write_zip(members())


def rebuild(data, *, index=None, manifest=None, policy=None, arrays=None, fix_size=True) -> bytes:
    """Decode an artifact, mutate arrays or JSON, and re-encode it with valid CRCs and sizes."""
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
            descriptor["components"] = 1 if array.ndim == 1 else int(array.shape[1])
            buffer += raw

    def members():
        parts = [
            Member("baslt.json", canonical_json(artifact.header), 0),
            Member("index.json", canonical_json(index_obj), 8),
            Member("manifest.json", canonical_json(manifest_obj), 8),
            Member("policy.json", canonical_json(policy_obj), 8),
        ]
        for entry in artifact.entries[4:]:
            parts.append(Member(entry.name, bytes(payloads.get(entry.name, artifact.members[entry.name])), 8))
        return parts

    if fix_size:
        size = len(write_zip(members()))
        manifest_obj["artifact"]["size_bytes"] = f"{size:020d}"
    return write_zip(members())


def checks_of(data):
    artifact, checks = structure_checks(data)
    return artifact, {check.id: check for check in checks}


def assert_fails(data, check_id, *, fragment=None):
    _artifact, checks = checks_of(data)
    check = checks[check_id]
    assert check.status == "fail", f"{check_id} is {check.status}: {check.message}"
    if fragment is not None:
        assert fragment in check.message, check.message
    return check


# --- a valid artifact --------------------------------------------------------------------------------------


def test_valid_artifact_passes_every_structure_check():
    artifact, checks = checks_of(build())
    assert artifact is not None
    assert list(checks) == list(STRUCTURE_IDS)
    assert all(check.status == PASS for check in checks.values()), {
        k: v.message for k, v in checks.items() if v.status != PASS
    }


def test_rebuild_of_an_untouched_artifact_still_passes():
    _artifact, checks = checks_of(rebuild(build()))
    assert all(check.status == PASS for check in checks.values())


def test_bind_counts_every_parameter_it_reproduced():
    _artifact, checks = checks_of(build())
    assert checks["structure.bind"].measured == "8"  # two window and six crossing parameters


def test_a_path_is_accepted(tmp_path):
    path = tmp_path / "run.baslt"
    path.write_bytes(build())
    artifact, checks = checks_of(path)
    assert artifact is not None
    assert all(check.status == PASS for check in checks.values())


# --- structure.container -----------------------------------------------------------------------------------


def test_flipped_array_byte_breaks_the_crc():
    data = bytearray(build())
    with zipfile.ZipFile(__import__("io").BytesIO(bytes(data))) as archive:
        start = archive.getinfo("s/0").header_offset + 30 + len("s/0")
    data[start] ^= 0xFF
    check = assert_fails(bytes(data), "structure.container")
    assert "CRC" in check.message or "decompress" in check.message


def test_truncated_file_is_not_a_container():
    assert_fails(build()[:-1], "structure.container")


def test_members_out_of_order_are_refused():
    data = build(order=[0, 3, 2, 1, 4])  # policy.json where index.json belongs
    assert_fails(data, "structure.container", fragment="members start with")


def test_unreadable_container_leaves_the_rest_not_applicable():
    _artifact, checks = checks_of(build()[:-1])
    assert checks["structure.container"].status == "fail"
    for name in ("descriptors", "invariants", "size", "policy", "bind"):
        assert checks[f"structure.{name}"].status == "not_applicable"


# --- structure.descriptors ---------------------------------------------------------------------------------


def test_descriptor_nbytes_must_match_dtype_and_shape():
    def edit(index, _manifest, _policy):
        index["signals"][0]["arrays"][1]["nbytes"] += 8

    assert_fails(build(edit=edit), "structure.descriptors", fragment="n*components*width")


def test_descriptor_must_lie_inside_its_member():
    def edit(index, _manifest, _policy):
        index["signals"][0]["arrays"][0]["offset"] = 4096

    assert_fails(build(edit=edit), "structure.descriptors", fragment="lie outside member")


def test_descriptor_member_must_be_a_data_member():
    def edit(index, _manifest, _policy):
        index["signals"][0]["arrays"][0]["member"] = "manifest.json"

    assert_fails(build(edit=edit), "structure.descriptors", fragment="not a data member")


def test_unsupported_dtype_is_named():
    def edit(index, _manifest, _policy):
        index["signals"][0]["arrays"][1]["dtype"] = "<f2"

    assert_fails(build(edit=edit), "structure.descriptors", fragment="unsupported dtype")


# --- structure.invariants ----------------------------------------------------------------------------------


def test_timestamps_must_not_decrease():
    data = rebuild(build(), arrays=lambda a: a["q_dyn"]["t"].__setitem__(2, 1.0))
    assert_fails(data, "structure.invariants", fragment="time decreases")


def test_timestamps_must_be_finite():
    data = rebuild(build(), arrays=lambda a: a["q_dyn"]["t"].__setitem__(1, np.nan))
    assert_fails(data, "structure.invariants", fragment="not finite")


def test_idx_must_be_strictly_increasing():
    idx = np.array([0, 6, 3, 10], dtype=np.uint32)  # delta+shuffle would refuse these, so store them plainly
    assert_fails(build(idx=idx, idx_enc="shuffle"), "structure.invariants", fragment="strictly increasing")


def test_idx_must_stay_inside_the_source():
    idx = np.array([0, 3, 6, 99], dtype=np.uint32)
    assert_fails(build(idx=idx), "structure.invariants", fragment="but the source has 11 samples")


def test_arrays_must_have_equal_lengths():
    short = np.array([0b011, 0b010, 0b101], dtype=np.uint16)
    assert_fails(build(roles=short), "structure.invariants", fragment="n is 3, expected 4")


def test_role_bits_must_be_in_the_legend():
    roles = ROLES.copy()
    roles[1] = 1 << 9
    assert_fails(build(roles=roles), "structure.invariants", fragment="outside the legend")


def test_extent_must_mark_the_first_and_last_source_sample():
    roles = ROLES.copy()
    roles[0] &= ~np.uint16(1)  # drop the extent bit from source sample 0
    assert_fails(build(roles=roles), "structure.invariants", fragment="the extent role marks")


def test_first_and_last_source_samples_must_be_retained():
    idx = np.array([1, 3, 6, 10], dtype=np.uint32)
    assert_fails(build(idx=idx), "structure.invariants", fragment="extent retention needs source samples")


def test_legend_without_extent_is_refused():
    def edit(index, _manifest, _policy):
        index["signals"][0]["roles"][0]["id"] = "soft"

    assert_fails(build(edit=edit), "structure.invariants", fragment="no 'extent' entry")


# --- structure.size ----------------------------------------------------------------------------------------


def test_size_bytes_must_equal_the_file_length():
    def edit(_index, manifest, _policy):
        manifest["artifact"]["size_bytes"] = f"{12345:020d}"

    assert_fails(build(edit=edit, fix_size=False), "structure.size", fragment="but the file is")


def test_size_bytes_must_be_zero_padded_to_twenty_digits():
    def edit(_index, manifest, _policy):
        manifest["artifact"]["size_bytes"] = "123"

    assert_fails(build(edit=edit, fix_size=False), "structure.size", fragment="20-digit")


def test_size_must_not_exceed_max_bytes():
    def edit(_index, manifest, _policy):
        manifest["artifact"]["max_bytes"] = 100
        manifest["budget"]["source"] = "cli"  # the ceiling no longer comes from the policy

    assert_fails(build(edit=edit), "structure.size", fragment="over the 100-byte budget")


# --- structure.policy --------------------------------------------------------------------------------------


def test_recorded_policy_hash_must_match_the_canonical_policy():
    def edit(_index, _manifest, policy):
        policy["sha256"] = "0" * 64

    assert_fails(build(edit=edit), "structure.policy", fragment="policy.json records")


def test_mutated_canonical_policy_is_caught():
    def edit(_index, _manifest, policy):
        policy["canonical"]["hard"]["q_dyn"]["window_extrema"]["interval"] = "10 s"

    assert_fails(build(edit=edit), "structure.policy", fragment="hashes to")


def test_manifest_and_policy_member_must_agree():
    def edit(_index, manifest, _policy):
        manifest["policy"]["sha256"] = "f" * 64

    assert_fails(build(edit=edit), "structure.policy", fragment="manifest.policy.sha256")


# --- structure.bind ----------------------------------------------------------------------------------------


def test_bound_parameter_must_match_the_policy():
    def edit(index, _manifest, _policy):
        index["signals"][0]["requirements"][2]["params"][0]["value"] = 51.0

    assert_fails(build(edit=edit), "structure.bind", fragment="hard.q_dyn.threshold_crossing[0].value")


def test_one_ulp_off_is_a_different_parameter():
    def edit(index, _manifest, _policy):
        index["signals"][0]["requirements"][1]["params"][0]["value"] = math.nextafter(5.0, math.inf)

    assert_fails(build(edit=edit), "structure.bind", fragment="hard.q_dyn.window_extrema.interval")


def test_duration_binds_to_seconds():
    def edit(index, _manifest, _policy):
        index["signals"][0]["requirements"][2]["params"][3]["value"] = 100.0  # '100 ms' is 0.1 s, not 100

    assert_fails(build(edit=edit), "structure.bind", fragment="debounce")


def test_enum_parameter_must_match():
    def edit(index, _manifest, _policy):
        index["signals"][0]["requirements"][2]["params"][1]["value"] = "rising"

    assert_fails(build(edit=edit), "structure.bind", fragment="edge")


def test_parameter_absent_from_the_policy_is_refused():
    def edit(index, _manifest, _policy):
        index["signals"][0]["requirements"][0]["params"].append({"name": "value", "value": 1.0})

    assert_fails(build(edit=edit), "structure.bind", fragment="has no parameter")


def test_requirement_absent_from_the_policy_is_refused():
    def edit(index, _manifest, _policy):
        index["signals"][0]["requirements"][0]["id"] = "hard.q_dyn.state_transitions"
        index["signals"][0]["requirements"][0]["op"] = "state_transitions"

    assert_fails(build(edit=edit), "structure.bind", fragment="no state_transitions for signal 'q_dyn'")


def test_max_size_binds_to_max_bytes():
    def edit(_index, manifest, _policy):
        manifest["artifact"]["max_bytes"] = 2097152  # the policy says 1 MiB

    assert_fails(build(edit=edit), "structure.bind", fragment="artifact.max_size")


def test_a_value_in_another_unit_binds_to_the_signal_unit():
    policy = copy.deepcopy(POLICY)
    policy["hard"]["q_dyn"]["threshold_crossing"] = [{"value": "50 kPa", "debounce": "100 ms"}]

    def edit(index, _manifest, _policy):
        index["signals"][0]["requirements"][2]["params"][0]["value"] = 50000.0

    _artifact, checks = checks_of(build(policy_dict=policy, edit=edit))
    assert checks["structure.bind"].status == PASS, checks["structure.bind"].message

    def wrong(index, _manifest, _policy):
        index["signals"][0]["requirements"][2]["params"][0]["value"] = 50.0

    assert_fails(build(policy_dict=policy, edit=wrong), "structure.bind", fragment="'50 kPa' binds to 50000.0")


# --- the re-binding arithmetic -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "written", "unit", "expected"),
    [
        ("value", "50 Pa", "Pa", 50.0),
        ("value", "65 kPa", "Pa", 65000.0),
        ("value", "50", "Pa", 50.0),  # a bare number is already in the signal's unit
        ("value", "7 deg", "rad", 0.12217304763960307),
        ("value", "25 degC", "K", 298.15),
        ("delta", "2 degC", "K", 2.0),  # a difference ignores the affine offset
        ("delta", "1 kPa", "Pa", 1000.0),
        ("duration", "100 ms", "Pa", 0.1),
        ("duration", "0 s", "Pa", 0.0),
        ("duration", "2 min", None, 120.0),
        ("enum", "rising", "Pa", "rising"),
    ],
)
def test_rebind_matches_the_documented_arithmetic(kind, written, unit, expected):
    bound, problem = rebind(kind, written, unit)
    assert problem is None
    if isinstance(expected, str):
        assert bound == expected
    else:
        assert struct.pack("<d", bound) == struct.pack("<d", expected), (bound, expected)


@pytest.mark.parametrize(
    ("kind", "written", "unit", "fragment"),
    [
        ("value", "7 deg", "Pa", "is a angle but the signal unit Pa is a pressure"),
        ("value", "7 furlong", "Pa", "unknown unit"),
        ("value", "7 Pa", None, "has a unit but the signal has none"),
        ("duration", "7 Pa", "Pa", "but a duration is required"),
        ("value", "not a number", "Pa", "is not a quantity"),
    ],
)
def test_rebind_reports_what_the_policy_would_have_rejected(kind, written, unit, fragment):
    bound, problem = rebind(kind, written, unit)
    assert bound is None
    assert fragment in problem


def test_json_members_stay_canonical():
    """The re-hashing in structure.policy only works because the members are canonical JSON."""
    artifact = read_artifact(build())
    raw = artifact.members["policy.json"].decode("utf-8")
    assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
