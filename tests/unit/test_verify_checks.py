"""`verify_artifact` on hand-made artifacts: every check id, every status, and a tamper for each check.

The artifact below is written by hand with `baslt.container`, never by the planner, and every number in its
manifest is worked out from docs/contracts.md in the comments beside it. The signal is

    t = 0..10 s,  q_dyn = [0, 10, 20, 100, 90, 30, 5, 5, 60, 70, 20] Pa,  mode = [0,0,1,1,1,2,2,0,0,0,3]

so the facts the requirements claim can be read off by eye:

- global extrema: max 100 Pa at sample 3, min 0 Pa at sample 0.
- window extrema over 5 s buckets: bucket 0 holds samples 0..5 (sample 5 sits on the boundary and is borrowed),
  bucket 1 holds 5..10, bucket 2 holds sample 10 alone.
- crossings of 50 Pa: the state x >= 50 is [F,F,F,T,T,F,F,F,T,T,F], so it flips at samples 3, 5, 8 and 10, and
  each flip takes the last level crossing in its direction.
- violation above 80 Pa: one run over samples 3 and 4, worst sample 3.
- transitions of mode: samples 0, 2, 5, 7 and 10.

`rebuild` is the tamper tool docs/verification.md asks for: decode, mutate an array or a JSON member, re-encode
with valid CRCs and sizes. Every failure below therefore reaches the verifier as a well-formed artifact.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from baslt import hashing
from baslt.container.encode import container_dtype, encode_array
from baslt.container.reader import read_artifact
from baslt.container.spec import ArrayDesc, canonical_json, header_json
from baslt.container.zipwriter import Member, write_zip
from baslt.policy import load_policy
from baslt.verify.checks import verify_artifact
from reference.artifact_edit import settle

pytestmark = pytest.mark.minimal

N_SOURCE = 11
T = np.arange(N_SOURCE, dtype=np.float64)
Q = np.array([0.0, 10.0, 20.0, 100.0, 90.0, 30.0, 5.0, 5.0, 60.0, 70.0, 20.0])
MODE = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 3], dtype=np.int32)
DEAD = np.full(N_SOURCE, np.nan)

LEVEL = 50.0
LIMIT = 80.0

Q_ROLES = [
    {"bit": 0, "id": "extent"},
    {"bit": 1, "id": "hard.q_dyn.global_extrema"},
    {"bit": 2, "id": "hard.q_dyn.window_extrema"},
    {"bit": 3, "id": "hard.q_dyn.threshold_crossing[0]"},
    {"bit": 4, "id": "hard.q_dyn.violation[0]#edge"},
    {"bit": 5, "id": "hard.q_dyn.violation[0]#worst"},
]
MODE_ROLES = [{"bit": 0, "id": "extent"}, {"bit": 1, "id": "hard.mode.state_transitions"}]
DEAD_ROLES = [{"bit": 0, "id": "extent"}, {"bit": 1, "id": "hard.dead.global_extrema"}]

EXPECTED_IDS = (
    "structure.container",
    "structure.descriptors",
    "structure.invariants",
    "structure.size",
    "structure.policy",
    "structure.bind",
    "hard.q_dyn.global_extrema.max",
    "hard.q_dyn.global_extrema.min",
    "hard.q_dyn.window_extrema.buckets",
    "hard.q_dyn.threshold_crossing[0].brackets",
    "hard.q_dyn.threshold_crossing[0].fidelity",
    "hard.q_dyn.violation[0].runs",
    "hard.q_dyn.violation[0].fidelity",
    "hard.mode.state_transitions.transitions",
    "budget.accounting",
)


def policy_dict(*, debounce="0 s", dead=False, on_not_applicable="warn") -> dict:
    hard = {
        "q_dyn": {
            "global_extrema": {},
            "window_extrema": {"interval": "5 s"},
            "threshold_crossing": [{"value": "50 Pa", "debounce": debounce}],
            "violation": [{"above": "80 Pa"}],
        },
        "mode": {"state_transitions": {}},
    }
    if dead:
        hard["dead"] = {"global_extrema": {}}
    return {
        "version": 1,
        "name": "checks_probe",
        "artifact": {"max_size": "1 MiB", "on_not_applicable": on_not_applicable},
        "signals": {"decl": {"q_dyn": {"unit": "Pa"}, "mode": {"kind": "discrete"}}},
        "hard": hard,
    }


def crossing_time(before: int, after: int, level: float) -> float:
    """The contract's formula, evaluated in its own order so the artifact and the verifier agree bitwise."""
    return T[before] + (T[after] - T[before]) * (level - Q[before]) / (Q[after] - Q[before])


# --- building ----------------------------------------------------------------------------------------------


def retained() -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Retained source index -> role mask, per signal, worked out from docs/contracts.md."""
    q: dict[int, int] = {}

    def mark(indices, bit):
        for i in indices:
            q[i] = q.get(i, 0) | (1 << bit)

    mark([0, 10], 0)  # extent: the first and last source sample
    mark([3, 0], 1)  # global max (100 at 3) and min (0 at 0)
    mark([3, 0, 9, 6, 10], 2)  # bucket 0 max/min, bucket 1 max/min, bucket 2
    mark([2, 3, 4, 5, 7, 8, 9, 10], 3)  # the bracketing pair of every step-3 flip
    mark([2, 3, 4, 5], 4)  # both boundary pairs of the violating run
    mark([3], 5)  # its worst sample

    mode = {i: 1 for i in (0, 10)}
    for i in (0, 2, 5, 7, 10):
        mode[i] = mode.get(i, 0) | 2
    dead = {0: 1, 10: 1}
    return q, mode, dead


def descriptors(member, t, values, idx, roles):
    buffer = bytearray()
    out = []
    for name, array, enc in (("t", t, "shuffle"), ("v", values, "shuffle"), ("idx", idx, "delta+shuffle"),
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


def signal_entry(name, member, picks, values, roles_legend, requirements, *, unit=None, kind="continuous"):
    idx = np.array(sorted(picks), dtype=np.uint32)
    masks = np.array([picks[int(i)] for i in idx], dtype=np.uint16)
    arrays, payload = descriptors(member, T[idx], values[idx], idx, masks)
    entry = {
        "name": name, "path": name, "kind": kind, "interp": "linear" if kind == "continuous" else "hold",
        "unit": unit, "labels": [], "n_source": N_SOURCE, "n": int(idx.size), "components": 1,
        "arrays": arrays, "roles": roles_legend, "requirements": requirements,
    }
    return entry, payload


def build(*, debounce=0.0, dead=False, on_not_applicable="warn", digest=None, edit=None) -> bytes:
    policy = load_policy(
        policy_dict(debounce="2 s" if debounce else "0 s", dead=dead, on_not_applicable=on_not_applicable)
    )
    q_picks, mode_picks, dead_picks = retained()

    q_entry, q_payload = signal_entry(
        "q_dyn", "s/0", q_picks, Q, Q_ROLES,
        [
            {"id": "hard.q_dyn.global_extrema", "op": "global_extrema", "bits": [1], "params": []},
            {"id": "hard.q_dyn.window_extrema", "op": "window_extrema", "bits": [2],
             "params": [{"name": "interval", "value": 5.0}, {"name": "origin", "value": 0.0}]},
            {"id": "hard.q_dyn.threshold_crossing[0]", "op": "threshold_crossing", "bits": [3],
             "params": [{"name": "value", "value": LEVEL}, {"name": "edge", "value": "both"},
                        {"name": "hysteresis", "value": 0.0}, {"name": "debounce", "value": debounce},
                        {"name": "tolerance", "value": 0.0}, {"name": "interpolate", "value": "linear"}]},
            {"id": "hard.q_dyn.violation[0]", "op": "violation", "bits": [4, 5],
             "params": [{"name": "above", "value": LIMIT}, {"name": "min_duration", "value": 0.0}]},
        ],
        unit="Pa",
    )
    mode_entry, mode_payload = signal_entry(
        "mode", "s/1", mode_picks, MODE, MODE_ROLES,
        [{"id": "hard.mode.state_transitions", "op": "state_transitions", "bits": [1], "params": []}],
        kind="discrete",
    )
    signals = [q_entry, mode_entry]
    payloads = {"s/0": q_payload, "s/1": mode_payload}
    if dead:
        dead_entry, dead_payload = signal_entry(
            "dead", "s/2", dead_picks, DEAD, DEAD_ROLES,
            [{"id": "hard.dead.global_extrema", "op": "global_extrema", "bits": [1], "params": []}],
        )
        signals.append(dead_entry)
        payloads["s/2"] = dead_payload

    index = {"container": 1, "signals": signals}
    manifest = {
        "status": "pass",
        "baslt": {"version": "0.1.0", "container": 1},
        "source": {"path": None, "format": "numpy", "size_bytes": 264,
                   "digest": digest or hashing.hash_none().to_json(),
                   "signals": len(signals), "samples_total": 11 * len(signals), "issues": []},
        "policy": {"name": policy.name, "sha256": policy.sha256},
        "artifact": {"size_bytes": "0" * 20, "max_bytes": 1048576, "ratio": "0.000000e+00",
                     "codec": "deflate", "level": 6},
        "budget": {"source": "policy", "required_bytes": 0, "discretionary_bytes": 0, "overhead_bytes": 0,
                   "signals": [{"name": entry["name"], "retained": entry["n"], "hard": entry["n"], "soft": 0,
                                "bytes": 0, "soft_max_abs_err": "0.000000e+00"} for entry in signals]},
        "requirements": _requirements(debounce, dead),
        "events": [], "trajectories": [], "sync_groups": [],
    }
    policy_member = {"canonical": policy.canonical, "sha256": policy.sha256,
                     "source_format": policy.source_format, "source_text": policy.source_text}
    if edit is not None:
        edit(index, manifest, policy_member)

    def members():
        return [
            Member("baslt.json", header_json("deflate"), 0),
            Member("index.json", canonical_json(index), 8),
            Member("manifest.json", canonical_json(manifest), 8),
            Member("policy.json", canonical_json(policy_member), 8),
            *(Member(name, payloads[name], 8) for name in sorted(payloads)),
        ]

    # The ratio has a fixed width, so it can be set from a first sizing without moving the size.
    manifest["artifact"]["ratio"] = f"{264 / len(settle(members, manifest, index)):.6e}"
    data = settle(members, manifest, index)
    assert int(manifest["artifact"]["size_bytes"]) == len(data)
    return data


def _requirements(debounce: float, dead: bool) -> list[dict]:
    start = crossing_time(2, 3, LIMIT)  # 2 + (80 - 20) / (100 - 20) = 2.75
    end = crossing_time(4, 5, LIMIT)  # 4 + (80 - 90) / (30 - 90) = 4.1666...
    every = [
        {"t": crossing_time(2, 3, LEVEL), "edge": "rising", "index_before": 2, "component": 0},
        {"t": crossing_time(4, 5, LEVEL), "edge": "falling", "index_before": 4, "component": 0},
        {"t": crossing_time(7, 8, LEVEL), "edge": "rising", "index_before": 7, "component": 0},
        {"t": crossing_time(9, 10, LEVEL), "edge": "falling", "index_before": 9, "component": 0},
    ]
    if debounce:
        # Runs between flips last 2.29, 3.15, 1.58 and 0.6 s, so with a 2 s debounce only the first two flips
        # are accepted and the final run is still pending at the end of the signal.
        crossings = {"count": 2, "rising": 1, "falling": 1, "crossings": every[:2],
                     "pending_at_end": True, "gap_flips": 0}
    else:
        crossings = {"count": 4, "rising": 2, "falling": 2, "crossings": every,
                     "pending_at_end": False, "gap_flips": 0}

    requirements = [
        {"id": "hard.q_dyn.global_extrema", "signal": "q_dyn", "op": "global_extrema", "severity": "info",
         "status": "pass", "evidence": {"components": [
             {"component": 0, "max": {"index": 3, "t": 3.0, "value": 100.0},
              "min": {"index": 0, "t": 0.0, "value": 0.0}}]}},
        {"id": "hard.q_dyn.window_extrema", "signal": "q_dyn", "op": "window_extrema", "severity": "info",
         "status": "pass",
         "evidence": {"buckets": 3, "buckets_with_finite": 3, "interval": 5.0, "origin": 0.0}},
        {"id": "hard.q_dyn.threshold_crossing[0]", "signal": "q_dyn", "op": "threshold_crossing",
         "severity": "info", "status": "warn" if debounce else "pass", "evidence": crossings},
        {"id": "hard.q_dyn.violation[0]", "signal": "q_dyn", "op": "violation", "severity": "limit",
         "status": "pass", "evidence": {"count": 1, "total_duration": end - start, "runs": [
             {"start": start, "end": end, "worst_t": 3.0, "worst_value": 100.0, "component": 0, "flags": []}]}},
        {"id": "hard.mode.state_transitions", "signal": "mode", "op": "state_transitions", "severity": "info",
         "status": "pass",
         "evidence": {"transitions": 4, "distinct_count": 4, "distinct_values": [0, 1, 2, 3]}},
    ]
    if dead:
        requirements.append(
            {"id": "hard.dead.global_extrema", "signal": "dead", "op": "global_extrema", "severity": "info",
             "status": "not_applicable", "notes": ["no finite sample"],
             "evidence": {"components": [{"component": 0, "max": None, "min": None}]}}
        )
    return requirements


def rebuild(data, *, index=None, manifest=None, policy=None, arrays=None, budget=True) -> bytes:
    """Decode an artifact, mutate arrays or JSON, and re-encode it with valid CRCs and sizes.

    With `budget` the byte accounting follows the re-encoded sizes; False keeps the budget exactly as edited."""
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

    return settle(members, manifest_obj, index_obj if budget else None)


# --- helpers -----------------------------------------------------------------------------------------------


def run(data, **kwargs):
    result = verify_artifact(data, **kwargs)
    return result, {check.id: check for check in result.checks}


def requirement(manifest, req_id):
    return next(item for item in manifest["requirements"] if item["id"] == req_id)


def expect_fail(data, check_id, *, fragment=None, **kwargs):
    result, checks = run(data, **kwargs)
    check = checks[check_id]
    assert check.status == "fail", f"{check_id} is {check.status}: {check.message}"
    if fragment is not None:
        assert fragment in check.message, check.message
    assert result.status == "fail"
    assert result.exit_code == 1
    return check


def clear_bit(data, signal, source_index, bit):
    def edit(decoded):
        arrays = decoded[signal]
        position = int(np.flatnonzero(arrays["idx"].astype(np.int64) == source_index)[0])
        arrays["roles"][position] &= ~np.uint16(1 << bit)

    return rebuild(data, arrays=edit)


# --- the well-formed artifact ------------------------------------------------------------------------------


def test_a_well_formed_artifact_passes_every_check():
    result, checks = run(build())
    assert tuple(checks) == EXPECTED_IDS
    assert result.status == "pass"
    assert result.exit_code == 0
    assert [c.id for c in result.failures()] == []


def test_bases_follow_the_contracts():
    _result, checks = run(build())
    assert checks["hard.q_dyn.global_extrema.max"].basis == "attested"
    assert checks["hard.q_dyn.window_extrema.buckets"].basis == "attested"
    assert checks["hard.q_dyn.threshold_crossing[0].brackets"].basis == "artifact"
    assert checks["hard.q_dyn.threshold_crossing[0].fidelity"].basis == "artifact"
    assert checks["hard.q_dyn.violation[0].runs"].basis == "artifact"
    assert checks["structure.container"].basis == "artifact"


def test_the_report_renders_and_serializes():
    result, _checks = run(build())
    text = result.render()
    assert "Result: PASS" in text
    assert result.to_json()["status"] == "pass"


# --- statuses ----------------------------------------------------------------------------------------------


def test_a_debounced_run_pending_at_the_end_warns():
    result, checks = run(build(debounce=2.0))
    check = checks["hard.q_dyn.threshold_crossing[0].fidelity"]
    assert check.status == "warn"
    assert "pending_at_end" in check.message
    assert check.claimed == "2" and check.measured == "2"
    assert result.status == "pass_with_warnings"
    assert result.exit_code == 0


def test_strict_turns_a_warning_into_a_failing_exit_code():
    result, _checks = run(build(debounce=2.0), strict=True)
    assert result.status == "pass_with_warnings"
    assert result.exit_code == 1


def test_a_requirement_on_an_all_nan_signal_is_not_applicable():
    result, checks = run(build(dead=True))
    check = checks["hard.dead.global_extrema"]
    assert check.status == "not_applicable"
    assert check.basis == "-"
    assert "no finite sample" in check.message
    assert result.status == "pass_with_warnings"


def test_on_not_applicable_fail_makes_it_a_failure():
    result, _checks = run(build(dead=True, on_not_applicable="fail"))
    assert result.on_not_applicable == "fail"
    assert result.status == "fail"


def test_a_not_applicable_requirement_must_not_flag_samples():
    def edit(decoded):
        decoded["dead"]["roles"][0] |= np.uint16(0b10)

    data = rebuild(build(dead=True), arrays=edit)
    expect_fail(data, "hard.dead.global_extrema", fragment="sample(s) carry its role")


# --- global_extrema ----------------------------------------------------------------------------------------


def test_a_claimed_extremum_value_must_match_the_artifact_bitwise():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.global_extrema")["evidence"]["components"][0]["max"]["value"] = 101.0

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.global_extrema.max", fragment="in the manifest")


def test_a_claimed_extremum_time_must_match_the_artifact_bitwise():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.global_extrema")["evidence"]["components"][0]["min"]["t"] = 1.0

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.global_extrema.min", fragment="t is")


def test_a_claimed_extremum_must_be_retained():
    def edit(manifest):
        claim = requirement(manifest, "hard.q_dyn.global_extrema")["evidence"]["components"][0]["max"]
        claim.update({"index": 1, "t": 1.0, "value": 10.0})  # sample 1 is the one sample nothing kept

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.global_extrema.max", fragment="not retained")


def test_a_retained_sample_may_not_beat_the_claimed_extremum():
    def edit(decoded):
        arrays = decoded["q_dyn"]
        position = int(np.flatnonzero(arrays["idx"].astype(np.int64) == 6)[0])
        arrays["v"][position] = 200.0

    expect_fail(rebuild(build(), arrays=edit), "hard.q_dyn.global_extrema.max", fragment="beats the claimed max")


def test_the_claimed_extremum_must_carry_its_role():
    data = clear_bit(build(), "q_dyn", 3, 1)
    expect_fail(data, "hard.q_dyn.global_extrema.max", fragment="does not carry the requirement's role")


# --- window_extrema ----------------------------------------------------------------------------------------


def test_a_buckets_extremum_must_be_flagged():
    data = clear_bit(build(), "q_dyn", 9, 2)  # the max of bucket 1
    expect_fail(data, "hard.q_dyn.window_extrema.buckets", fragment="is not flagged")


def test_the_evidence_interval_must_match_the_bound_parameter():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.window_extrema")["evidence"]["interval"] = 2.0

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.window_extrema.buckets", fragment="index.json binds")


def test_more_retained_buckets_than_claimed_is_inconsistent():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.window_extrema")["evidence"]["buckets_with_finite"] = 1

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.window_extrema.buckets", fragment="more than the 1")


# --- threshold_crossing ------------------------------------------------------------------------------------


def test_every_claimed_crossing_is_bracketed_and_straddles_the_level():
    _result, checks = run(build())
    check = checks["hard.q_dyn.threshold_crossing[0].brackets"]
    assert check.status == "pass"
    assert check.claimed == "4" and check.measured == "4"


def test_a_bracket_that_does_not_straddle_the_level_fails():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.threshold_crossing[0]")["evidence"]["crossings"][0]["index_before"] = 5

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.threshold_crossing[0].brackets",
                fragment="does not straddle")


def test_a_bracket_that_was_not_retained_fails():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.threshold_crossing[0]")["evidence"]["crossings"][0]["index_before"] = 1

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.threshold_crossing[0].brackets",
                fragment="is not retained")


def test_an_unflagged_bracket_fails():
    data = clear_bit(build(), "q_dyn", 2, 3)
    expect_fail(data, "hard.q_dyn.threshold_crossing[0].brackets", fragment="not flagged")


def test_a_missing_crossing_is_caught_by_re_detection():
    def edit(manifest):
        evidence = requirement(manifest, "hard.q_dyn.threshold_crossing[0]")["evidence"]
        evidence["crossings"] = evidence["crossings"][:3]
        evidence["count"] = 3
        evidence["falling"] = 1

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.threshold_crossing[0].fidelity",
                fragment="detection finds 4")


def test_a_crossing_time_outside_the_tolerance_is_caught():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.threshold_crossing[0]")["evidence"]["crossings"][0]["t"] += 0.5

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.threshold_crossing[0].fidelity", fragment="outside")


def test_a_crossing_time_one_ulp_away_is_inside_the_floating_point_slack():
    def edit(manifest):
        crossings = requirement(manifest, "hard.q_dyn.threshold_crossing[0]")["evidence"]["crossings"]
        crossings[0]["t"] = np.nextafter(crossings[0]["t"], np.inf)

    _result, checks = run(rebuild(build(), manifest=edit))
    assert checks["hard.q_dyn.threshold_crossing[0].fidelity"].status == "pass"


def test_mutating_a_bracket_value_changes_what_re_detection_finds():
    def edit(decoded):
        arrays = decoded["q_dyn"]
        position = int(np.flatnonzero(arrays["idx"].astype(np.int64) == 8)[0])
        arrays["v"][position] = 5.0  # sample 8 no longer rises above the level

    expect_fail(rebuild(build(), arrays=edit), "hard.q_dyn.threshold_crossing[0].fidelity")


# --- violation ---------------------------------------------------------------------------------------------


def test_a_claimed_run_holds_its_worst_sample():
    _result, checks = run(build())
    check = checks["hard.q_dyn.violation[0].runs"]
    assert check.status == "pass"
    assert check.claimed == "1" and check.measured == "1"


def test_a_wrong_worst_value_fails():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.violation[0]")["evidence"]["runs"][0]["worst_value"] = 90.0

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.violation[0].runs", fragment="worst value")


def test_a_worst_sample_outside_the_run_fails():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.violation[0]")["evidence"]["runs"][0]["worst_t"] = 9.0

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.violation[0].runs", fragment="outside the run")


def test_an_unflagged_worst_sample_fails():
    data = clear_bit(build(), "q_dyn", 3, 5)
    expect_fail(data, "hard.q_dyn.violation[0].runs", fragment="worst role")


def test_an_unflagged_boundary_pair_fails():
    data = clear_bit(build(), "q_dyn", 2, 4)
    expect_fail(data, "hard.q_dyn.violation[0].runs", fragment="edge role")


def test_a_run_shorter_than_min_duration_fails():
    def edit(index, manifest, _policy):
        for entry in index["signals"]:
            for item in entry["requirements"]:
                if item["id"] == "hard.q_dyn.violation[0]":
                    item["params"][1]["value"] = 5.0
        requirement(manifest, "hard.q_dyn.violation[0]")["evidence"]["count"] = 1

    # The policy still says 0 s, so structure.bind fails too; the run check is the one under test here.
    _result, checks = run(build(edit=edit))
    assert checks["hard.q_dyn.violation[0].runs"].status == "fail"
    assert "minimum duration" in checks["hard.q_dyn.violation[0].runs"].message


def test_a_shifted_run_boundary_is_caught_by_re_detection():
    def edit(manifest):
        requirement(manifest, "hard.q_dyn.violation[0]")["evidence"]["runs"][0]["end"] += 0.5

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.violation[0].fidelity", fragment="end is")


def test_an_extra_claimed_run_is_caught_by_re_detection():
    def edit(manifest):
        evidence = requirement(manifest, "hard.q_dyn.violation[0]")["evidence"]
        evidence["count"] = 2

    expect_fail(rebuild(build(), manifest=edit), "hard.q_dyn.violation[0].fidelity", fragment="claims 2 runs")


# --- state_transitions -------------------------------------------------------------------------------------


def test_a_wrong_transition_count_fails():
    def edit(manifest):
        requirement(manifest, "hard.mode.state_transitions")["evidence"]["transitions"] = 3

    expect_fail(rebuild(build(), manifest=edit), "hard.mode.state_transitions.transitions",
                fragment="the retained samples show 4")


def test_a_wrong_distinct_count_fails():
    def edit(manifest):
        requirement(manifest, "hard.mode.state_transitions")["evidence"]["distinct_count"] = 9

    expect_fail(rebuild(build(), manifest=edit), "hard.mode.state_transitions.transitions",
                fragment="distinct values")


def test_an_unflagged_transition_fails():
    data = clear_bit(build(), "mode", 5, 1)
    expect_fail(data, "hard.mode.state_transitions.transitions", fragment="is not flagged")


# --- manifest and index must agree -------------------------------------------------------------------------


def test_a_requirement_only_the_manifest_knows_fails():
    def edit(index, _manifest, _policy):
        index["signals"][0]["requirements"] = index["signals"][0]["requirements"][1:]

    expect_fail(build(edit=edit), "hard.q_dyn.global_extrema", fragment="index.json does not")


def test_a_requirement_only_the_index_knows_fails():
    def edit(_index, manifest, _policy):
        manifest["requirements"] = [r for r in manifest["requirements"] if r["id"] != "hard.mode.state_transitions"]

    expect_fail(build(edit=edit), "hard.mode.state_transitions", fragment="the manifest does not")


def test_an_unsupported_operator_is_refused():
    def edit(index, manifest, _policy):
        for entry in index["signals"]:
            for item in entry["requirements"]:
                if item["id"] == "hard.q_dyn.global_extrema":
                    item["op"] = "made_up_operator"
        requirement(manifest, "hard.q_dyn.global_extrema")["op"] = "made_up_operator"

    expect_fail(build(edit=edit), "hard.q_dyn.global_extrema", fragment="not supported by this verifier")


def test_an_unreadable_artifact_reports_only_structure():
    result, checks = run(build()[:-1])
    assert checks["structure.container"].status == "fail"
    assert result.status == "fail"
    assert not any(check.id.startswith("hard.") for check in result.checks)


# --- source checks -----------------------------------------------------------------------------------------


def loaded_run(path=None):
    """Loaded run data in the shape the CLI hands over: signals plus, optionally, the file it came from."""
    signals = {"q_dyn": (T, Q), "mode": (T, MODE)}
    return SimpleNamespace(signals=signals, meta=SimpleNamespace(path=path))


def test_source_recomputes_every_claimed_fact():
    result, checks = run(build(), source=loaded_run())
    assert checks["source.samples"].status == "pass"
    assert checks["hard.q_dyn.global_extrema.max"].basis == "source"
    assert checks["hard.q_dyn.window_extrema.buckets"].basis == "source"
    for check_id in ("hard.q_dyn.threshold_crossing[0].source", "hard.q_dyn.violation[0].source",
                     "hard.mode.state_transitions.source"):
        assert checks[check_id].status == "pass", checks[check_id].message
        assert checks[check_id].basis == "source"
    assert result.status == "pass_with_warnings"  # the digest is 'none', so it cannot be rechecked


def test_source_samples_catch_a_mutated_value():
    def edit(decoded):
        decoded["q_dyn"]["v"][2] = 42.0

    data = rebuild(build(), arrays=edit)
    expect_fail(data, "source.samples", fragment="but the source has", source=loaded_run())


def test_source_samples_catch_a_mutated_timestamp():
    def edit(decoded):
        decoded["q_dyn"]["t"][1] = 2.5

    data = rebuild(build(), arrays=edit)
    expect_fail(data, "source.samples", fragment="retained timestamp", source=loaded_run())


def test_a_signal_missing_from_the_source_fails():
    source = SimpleNamespace(signals={"q_dyn": (T, Q)}, meta=SimpleNamespace(path=None))
    expect_fail(build(), "source.samples", fragment="not in the source", source=source)


def test_a_source_of_a_different_length_fails():
    source = SimpleNamespace(signals={"q_dyn": (T[:-1], Q[:-1]), "mode": (T, MODE)},
                             meta=SimpleNamespace(path=None))
    expect_fail(build(), "source.samples", fragment="the source has 10 samples", source=source)


def test_the_source_settles_an_attested_claim():
    """The artifact alone cannot see a larger value that was dropped; the source can."""
    louder = Q.copy()
    louder[1] = 500.0  # sample 1 is the one sample the artifact does not keep
    source = SimpleNamespace(signals={"q_dyn": (T, louder), "mode": (T, MODE)}, meta=SimpleNamespace(path=None))
    _result, artifact_only = run(build())
    assert artifact_only["hard.q_dyn.global_extrema.max"].status == "pass"
    result, checks = run(build(), source=source)
    assert checks["hard.q_dyn.global_extrema.max"].status == "fail"
    assert "the source's max is sample 1" in checks["hard.q_dyn.global_extrema.max"].message
    assert result.status == "fail"


def test_hold_reconstruction_is_compared_with_the_source():
    def edit(decoded):
        arrays = decoded["mode"]
        position = int(np.flatnonzero(arrays["idx"].astype(np.int64) == 5)[0])
        arrays["v"][position] = 7
        arrays["idx"] = arrays["idx"]  # keep the index untouched

    data = rebuild(build(), arrays=edit)
    _result, checks = run(data, source=loaded_run())
    assert checks["hard.mode.state_transitions.source"].status == "fail"
    assert "hold reconstruction" in checks["hard.mode.state_transitions.source"].message


def test_a_full_digest_that_matches_passes(tmp_path):
    path = tmp_path / "run.bin"
    path.write_bytes(b"a simulation run" * 64)
    digest = hashing.hash_file(path, "full").to_json()
    _result, checks = run(build(digest=digest), source=loaded_run(str(path)))
    check = checks["source.digest"]
    assert check.status == "pass"
    assert "(full)" in check.claimed


def test_a_sampled_digest_is_a_fingerprint_not_a_match(tmp_path):
    path = tmp_path / "run.bin"
    path.write_bytes(b"a simulation run" * 64)
    digest = hashing.hash_file(path, "sampled").to_json()
    result, checks = run(build(digest=digest), source=loaded_run(str(path)))
    check = checks["source.digest"]
    assert check.status == "warn"
    assert "fingerprint, not a sha256 match" in check.message
    assert result.status == "pass_with_warnings"


def test_a_digest_that_differs_fails(tmp_path):
    path = tmp_path / "run.bin"
    path.write_bytes(b"a simulation run" * 64)
    digest = hashing.hash_file(path, "full").to_json()
    path.write_bytes(b"a different run" * 64)
    expect_fail(build(digest=digest), "source.digest", fragment="digests to", source=loaded_run(str(path)))


def test_a_digest_without_the_file_is_not_applicable(tmp_path):
    path = tmp_path / "run.bin"
    path.write_bytes(b"a simulation run" * 64)
    digest = hashing.hash_file(path, "full").to_json()
    _result, checks = run(build(digest=digest), source=loaded_run())  # no path in the loaded run
    assert checks["source.digest"].status == "not_applicable"
    assert "path of the source file" in checks["source.digest"].message


def test_an_arrays_digest_is_rechecked_from_the_arrays():
    digest = hashing.hash_arrays({"t": T, "q_dyn": Q, "mode": MODE}).to_json()
    _result, checks = run(build(digest=digest), source={"t": T, "q_dyn": Q, "mode": MODE})
    assert checks["source.digest"].status == "pass"
    # A mapping of raw arrays carries no clock, so the samples cannot be compared from it.
    assert checks["source.samples"].status == "not_applicable"


def test_a_path_alone_rechecks_the_digest_only(tmp_path):
    path = tmp_path / "run.bin"
    path.write_bytes(b"a simulation run" * 64)
    digest = hashing.hash_file(path, "full").to_json()
    result, checks = run(build(digest=digest), source=path)
    assert checks["source.digest"].status == "pass"
    assert checks["source.samples"].status == "not_applicable"
    assert result.status == "pass_with_warnings"
