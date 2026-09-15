"""Policy validation: every section, every key, defaults and error messages."""

from __future__ import annotations

import pytest

from baslt.errors import PolicyError
from baslt.policy import (
    OP_KINDS,
    OP_SCHEMAS,
    ArtifactSpec,
    Param,
    SignalsSpec,
    load_policy,
    validate_policy,
)
from baslt.policy.validate import IssueCollector, find_location, got
from baslt.units import Quantity


def pol(**sections):
    return {"version": 1, **sections}


def errors(data) -> list[str]:
    with pytest.raises(PolicyError) as info:
        load_policy(data)
    return [issue.render() for issue in info.value.issues]


def one_error(data) -> str:
    found = errors(data)
    assert len(found) == 1, found
    return found[0]


def q(text):
    from baslt.units import parse_quantity

    return parse_quantity(text)


# ----- schema tables ----------------------------------------------------------------------


def test_op_schemas_match_docs_table():
    assert OP_SCHEMAS == {
        "global_extrema": {},
        "local_extrema": {
            "prominence": Param("delta", required=True),
            "separation": Param("duration", default="0 s"),
            "kind": Param("enum", default="both", choices=("max", "min", "both")),
        },
        "window_extrema": {
            "interval": Param("duration", required=True),
            "origin": Param("duration", default="0 s"),
        },
        "threshold_crossing": {
            "value": Param("value", required=True),
            "edge": Param("enum", default="both", choices=("rising", "falling", "both")),
            "hysteresis": Param("delta", default=0),
            "debounce": Param("duration", default="0 s"),
            "tolerance": Param("duration", default="0 s"),
            "interpolate": Param("enum", default="linear", choices=("linear", "none")),
        },
        "violation": {
            "above": Param("value"),
            "below": Param("value"),
            "min_duration": Param("duration", default="0 s"),
        },
        "state_transitions": {},
    }


def test_op_kinds():
    assert OP_KINDS["state_transitions"] == ("discrete",)
    for op in ("local_extrema", "window_extrema", "threshold_crossing", "violation"):
        assert OP_KINDS[op] == ("continuous", "vector")
    assert set(OP_KINDS["global_extrema"]) == {"continuous", "discrete", "vector"}
    assert set(OP_KINDS) == set(OP_SCHEMAS)


# ----- top level --------------------------------------------------------------------------


def test_minimal_policy_defaults():
    policy = load_policy({"version": 1})
    assert policy.version == 1
    assert policy.name == "policy"
    assert policy.artifact == ArtifactSpec()
    assert policy.artifact.max_bytes is None and policy.artifact.codec == "deflate"
    assert policy.artifact.hash is None
    assert policy.artifact.soft_value_dtype == "source"
    assert policy.artifact.on_not_applicable == "warn"
    assert policy.signals == SignalsSpec()
    assert policy.signals.include == ["*"] and policy.signals.exclude == []
    assert policy.signals.time is None and policy.signals.time_unit == "s"
    assert policy.signals.on_non_monotonic == "error"
    assert policy.hard == [] and policy.events == [] and policy.trajectories == []
    assert policy.sync_groups == [] and policy.soft == []
    assert policy.review.thumbnails == [] and policy.review.kpis == []


def test_null_sections_are_absent():
    base = load_policy({"version": 1})
    nulls = load_policy(
        pol(artifact=None, signals=None, hard=None, events=None, trajectories=None, sync_groups=None, soft=None, review=None)
    )
    assert nulls.sha256 == base.sha256


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "version: missing required key 'version' (must be 1)"),
        ({"version": "1"}, "version: expected the integer 1, got '1'"),
        ({"version": True}, "version: expected the integer 1, got a boolean"),
        ({"version": 1.0}, "version: expected the integer 1, got 1.0"),
        ({"version": 2}, "version: unsupported policy version 2; only version 1 is supported"),
        (pol(name=5), "name: expected a non-empty string, got 5"),
        (pol(name=""), "name: expected a non-empty string, got ''"),
        (pol(hardd={}), "hardd: unknown key 'hardd'; did you mean 'hard'?"),
        (
            pol(zzz=1),
            "zzz: unknown key 'zzz'; expected one of version, name, artifact, signals, hard, events, "
            "trajectories, sync_groups, soft, review",
        ),
        ({"version": 1, 3: 4}, "keys must be strings, got 3"),
    ],
)
def test_top_level_errors(data, message):
    assert one_error(data) == message


def test_non_mapping_policy():
    with pytest.raises(PolicyError, match="a policy must be a mapping, got a list"):
        validate_policy([1])
    with pytest.raises(PolicyError, match="the policy is empty"):
        validate_policy(None)


def test_issues_are_collected_together():
    found = errors({"version": 3, "name": 1, "artifact": {"codec": "zip"}, "soft": {}})
    assert found == [
        "version: unsupported policy version 3; only version 1 is supported",
        "name: expected a non-empty string, got 1",
        "artifact.codec: expected one of deflate, store, zstd, got 'zip'",
        "soft: expected a list of rules, got a mapping",
    ]


def test_issue_limit_is_50():
    data = {"version": 1, **{f"unknown_{i}": 1 for i in range(80)}}
    with pytest.raises(PolicyError) as info:
        load_policy(data)
    assert len(info.value.issues) == 50


def test_issue_collector_and_helpers():
    collector = IssueCollector({"a": "f:1:1", "a.b[2]": "f:3:5"}, limit=3)
    collector.add("a.b[2].c", "x")
    collector.add("a.z", "y")
    assert [i.location for i in collector.issues] == ["f:3:5", "f:1:1"]
    with pytest.raises(PolicyError):
        collector.add("q", "1")
    assert find_location({}, "") is None
    assert find_location({"hard": "f:2:1"}, "hard.q[0].x") == "f:2:1"
    assert find_location({"x": "f:1:1"}, "y") is None
    assert got(None) == "null" and got(True) == "a boolean" and got([1]) == "a list"
    assert got({"a": 1}) == "a mapping" and got("s") == "'s'" and got(2.5) == "2.5"
    assert got(object()) == "object"


# ----- artifact ---------------------------------------------------------------------------


def test_artifact_values():
    policy = load_policy(
        pol(artifact={"max_size": "2 MiB", "codec": "zstd", "hash": "sampled", "soft_value_dtype": "float32", "on_not_applicable": "fail"})
    )
    a = policy.artifact
    assert (a.max_size, a.max_bytes, a.codec, a.hash, a.soft_value_dtype, a.on_not_applicable) == (
        "2 MiB",
        2097152,
        "zstd",
        "sampled",
        "float32",
        "fail",
    )
    assert load_policy(pol(artifact={"max_size": 4096})).artifact.max_bytes == 4096
    assert load_policy(pol(artifact={"max_size": "4096"})).artifact.max_bytes == 4096
    assert load_policy(pol(artifact={"hash": "none"})).artifact.hash == "none"


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ([], "artifact: expected a mapping, got a list"),
        ({"max_sise": 1}, "artifact.max_sise: unknown key 'max_sise'; did you mean 'max_size'?"),
        ({"max_size": 0}, "artifact.max_size: must be positive, got 0 bytes"),
        ({"max_size": -5}, "artifact.max_size: byte size must not be negative, got -5"),
        ({"max_size": "2 s"}, "artifact.max_size: '2 s' is a time but a byte size is required"),
        ({"max_size": 1.5}, "artifact.max_size: expected a byte size such as '2 MiB' or an integer number of bytes, got float"),
        ({"max_size": True}, "artifact.max_size: expected a byte size such as '2 MiB' or an integer number of bytes, got a boolean"),
        ({"max_size": "2 mib"}, "artifact.max_size: unknown unit 'mib' in '2 mib'; did you mean 'MiB'?"),
        ({"codec": "deflat"}, "artifact.codec: expected one of deflate, store, zstd, got 'deflat'; did you mean 'deflate'?"),
        ({"hash": "arrays"}, "artifact.hash: expected one of full, sampled, none, got 'arrays'"),
        ({"hash": False}, "artifact.hash: expected one of full, sampled, none, got a boolean"),
        ({"soft_value_dtype": "float16"}, "artifact.soft_value_dtype: expected one of source, float32, got 'float16'; did you mean 'float32'?"),
        ({"on_not_applicable": "error"}, "artifact.on_not_applicable: expected one of warn, fail, got 'error'"),
    ],
)
def test_artifact_errors(artifact, message):
    assert one_error(pol(artifact=artifact)) == message


# ----- signals ----------------------------------------------------------------------------


def test_signals_values():
    policy = load_policy(
        pol(
            signals={
                "time": "t",
                "time_unit": "ms",
                "on_non_monotonic": "sort",
                "include": ["aero/*", "prop/*"],
                "exclude": ["debug/*"],
                "decl": {
                    "q_dyn": {"path": "aero/q", "unit": "Pa"},
                    "mode": {"path": "gnc/mode", "kind": "discrete", "time": "gnc/t"},
                    "plain": None,
                },
            }
        )
    )
    s = policy.signals
    assert (s.time, s.time_unit, s.on_non_monotonic) == ("t", "ms", "sort")
    assert s.include == ["aero/*", "prop/*"] and s.exclude == ["debug/*"]
    assert s.decls["q_dyn"].path == "aero/q" and s.decls["q_dyn"].unit == "Pa" and s.decls["q_dyn"].kind is None
    assert s.decls["mode"].kind == "discrete" and s.decls["mode"].time == "gnc/t"
    assert s.decls["plain"].path is None and s.decls["plain"].alias == "plain"
    assert load_policy(pol(signals={"time_unit": "µs"})).signals.time_unit == "µs"


@pytest.mark.parametrize(
    ("signals", "message"),
    [
        ("t", "signals: expected a mapping, got 't'"),
        ({"tiem": "t"}, "signals.tiem: unknown key 'tiem'; did you mean 'time'?"),
        ({"time": 5}, "signals.time: expected a non-empty string, got 5"),
        ({"time_unit": "sec"}, "signals.time_unit: expected a time unit (s, ms, us, µs, ns, min, h), got 'sec'"),
        ({"time_unit": "mss"}, "signals.time_unit: expected a time unit (s, ms, us, µs, ns, min, h), got 'mss'; did you mean 'ms'?"),
        ({"time_unit": "m"}, "signals.time_unit: expected a time unit (s, ms, us, µs, ns, min, h), got 'm'; did you mean 'ms'?"),
        ({"time_unit": "K"}, "signals.time_unit: expected a time unit (s, ms, us, µs, ns, min, h), got 'K'"),
        ({"time_unit": 1}, "signals.time_unit: expected a time unit (s, ms, us, µs, ns, min, h), got 1"),
        ({"on_non_monotonic": "ignore"}, "signals.on_non_monotonic: expected one of error, sort, got 'ignore'"),
        ({"include": "*"}, "signals.include: expected a list of strings, got '*'"),
        ({"include": ["*", 3]}, "signals.include[1]: expected a non-empty string, got 3"),
        ({"exclude": [""]}, "signals.exclude[0]: expected a non-empty string, got ''"),
        ({"decl": ["q"]}, "signals.decl: expected a mapping, got a list"),
        ({"decl": {"q": "Pa"}}, "signals.decl.q: expected a mapping with path, unit, kind or time, got 'Pa'"),
        ({"decl": {"q": {"units": "Pa"}}}, "signals.decl.q.units: unknown key 'units'; did you mean 'unit'?"),
        ({"decl": {"q": {"unit": ""}}}, "signals.decl.q.unit: expected a non-empty string, got ''"),
        ({"decl": {"q": {"path": 3}}}, "signals.decl.q.path: expected a non-empty string, got 3"),
        ({"decl": {"q": {"kind": "analog"}}}, "signals.decl.q.kind: expected one of continuous, discrete, vector, got 'analog'"),
        ({"decl": {"": {}}}, "signals.decl: names must be non-empty strings, got ''"),
    ],
)
def test_signals_errors(signals, message):
    assert one_error(pol(signals=signals)) == message


# ----- hard -------------------------------------------------------------------------------


def test_hard_defaults_and_ids():
    policy = load_policy(
        pol(
            hard={
                "q": {
                    "global_extrema": {},
                    "local_extrema": {"prominence": "0.5 deg"},
                    "window_extrema": {"interval": "1 s"},
                    "threshold_crossing": [{"value": "65 kPa"}, {"value": 7, "edge": "rising"}],
                    "violation": [{"above": "70 kPa"}],
                },
                "mode": {"state_transitions": None},
            }
        )
    )
    by_id = {req.id: req for req in policy.hard}
    assert list(by_id) == [
        "hard.q.global_extrema",
        "hard.q.local_extrema",
        "hard.q.window_extrema",
        "hard.q.threshold_crossing[0]",
        "hard.q.threshold_crossing[1]",
        "hard.q.violation[0]",
        "hard.mode.state_transitions",
    ]
    assert by_id["hard.q.global_extrema"].params == {}
    assert by_id["hard.q.global_extrema"].severity == "info"
    assert by_id["hard.q.local_extrema"].params == {
        "prominence": Quantity(0.5, "deg", "0.5 deg"),
        "separation": Quantity(0.0, "s", "0 s"),
        "kind": "both",
    }
    assert by_id["hard.q.window_extrema"].params == {
        "interval": Quantity(1.0, "s", "1 s"),
        "origin": Quantity(0.0, "s", "0 s"),
    }
    tc = by_id["hard.q.threshold_crossing[0]"]
    assert tc.signal == "q" and tc.op == "threshold_crossing" and tc.severity == "info"
    assert tc.params == {
        "value": Quantity(65.0, "kPa", "65 kPa"),
        "edge": "both",
        "hysteresis": Quantity(0.0, None, "0"),
        "debounce": Quantity(0.0, "s", "0 s"),
        "tolerance": Quantity(0.0, "s", "0 s"),
        "interpolate": "linear",
    }
    assert by_id["hard.q.threshold_crossing[1]"].params["value"] == Quantity(7.0, None, "7")
    assert by_id["hard.q.threshold_crossing[1]"].params["edge"] == "rising"
    violation = by_id["hard.q.violation[0]"]
    assert violation.severity == "limit"
    assert violation.params == {"above": Quantity(70.0, "kPa", "70 kPa"), "min_duration": Quantity(0.0, "s", "0 s")}
    assert by_id["hard.mode.state_transitions"].params == {}


def test_hard_explicit_parameters():
    policy = load_policy(
        pol(
            hard={
                "aoa": {
                    "local_extrema": {"prominence": 2, "separation": "100 ms", "kind": "max", "severity": "limit"},
                    "window_extrema": {"interval": "2 min", "origin": "-5 s"},
                    "threshold_crossing": {
                        "value": "-7 deg",
                        "edge": "falling",
                        "hysteresis": "1 deg",
                        "debounce": "100ms",
                        "tolerance": "5 ms",
                        "interpolate": "none",
                    },
                    "violation": {"below": "-10 deg", "min_duration": "1 s", "severity": "info"},
                }
            }
        )
    )
    reqs = {req.op: req for req in policy.hard}
    assert reqs["local_extrema"].severity == "limit"
    assert reqs["local_extrema"].params["kind"] == "max"
    assert reqs["window_extrema"].params["origin"] == Quantity(-5.0, "s", "-5 s")
    assert reqs["threshold_crossing"].params["interpolate"] == "none"
    assert reqs["threshold_crossing"].params["debounce"] == Quantity(100.0, "ms", "100ms")
    assert reqs["violation"].severity == "info"
    assert reqs["violation"].params == {"below": Quantity(-10.0, "deg", "-10 deg"), "min_duration": Quantity(1.0, "s", "1 s")}


def test_value_parameters_accept_unknown_units_for_opaque_signals():
    policy = load_policy(pol(hard={"c": {"threshold_crossing": {"value": "12 counts", "hysteresis": "1 counts"}}}))
    assert policy.hard[0].params["value"] == Quantity(12.0, "counts", "12 counts")


@pytest.mark.parametrize(
    ("hard", "message"),
    [
        ([], "hard: expected a mapping, got a list"),
        ({"q": None}, "hard.q: expected a mapping of operator -> spec, got null"),
        ({"q": ["global_extrema"]}, "hard.q: expected a mapping of operator -> spec, got a list"),
        ({"q": {}}, "hard.q: expected at least one operator"),
        ({"q": {1: {}}}, "hard.q: keys must be strings, got 1"),
        ({"q": {"thresold_crossing": {}}}, "hard.q.thresold_crossing: unknown operator 'thresold_crossing'; did you mean 'threshold_crossing'?"),
        (
            {"q": {"foo": {}}},
            "hard.q.foo: unknown operator 'foo'; expected one of global_extrema, local_extrema, window_extrema, "
            "threshold_crossing, violation, state_transitions",
        ),
        ({"q": {"global_extrema": 5}}, "hard.q.global_extrema: expected a mapping or a list of mappings, got 5"),
        ({"q": {"threshold_crossing": []}}, "hard.q.threshold_crossing: expected at least one spec"),
        ({"q": {"threshold_crossing": ["x"]}}, "hard.q.threshold_crossing[0]: expected a mapping, got 'x'"),
        ({"q": {"threshold_crossing": [None]}}, "hard.q.threshold_crossing[0]: missing required parameter 'value'"),
        ({"q": {"local_extrema": {}}}, "hard.q.local_extrema: missing required parameter 'prominence'"),
        ({"q": {"window_extrema": {"interval": None}}}, "hard.q.window_extrema: missing required parameter 'interval'"),
        ({"q": {"threshold_crossing": [{"value": 1, "valeu": 2}]}}, "hard.q.threshold_crossing[0].valeu: unknown key 'valeu'; did you mean 'value'?"),
        ({"q": {"global_extrema": {"value": 1}}}, "hard.q.global_extrema.value: unknown key 'value'; expected one of severity"),
        ({"q": {"threshold_crossing": [{"value": 1, "edge": "up"}]}}, "hard.q.threshold_crossing[0].edge: expected one of rising, falling, both, got 'up'"),
        ({"q": {"threshold_crossing": {"value": 1, "edge": "risign"}}}, "hard.q.threshold_crossing.edge: expected one of rising, falling, both, got 'risign'; did you mean 'rising'?"),
        ({"q": {"threshold_crossing": {"value": 1, "interpolate": True}}}, "hard.q.threshold_crossing.interpolate: expected one of linear, none, got a boolean"),
        ({"q": {"local_extrema": {"prominence": 1, "kind": "peak"}}}, "hard.q.local_extrema.kind: expected one of max, min, both, got 'peak'"),
        ({"q": {"violation": {"above": 1, "below": 0}}}, "hard.q.violation: give exactly one of 'above' or 'below', not both"),
        ({"q": {"violation": {"min_duration": "1 s"}}}, "hard.q.violation: missing required parameter: give one of 'above' or 'below'"),
        ({"q": {"global_extrema": {"severity": "high"}}}, "hard.q.global_extrema.severity: expected one of info, limit, got 'high'"),
        ({"q": {"window_extrema": {"interval": "1 m"}}}, "hard.q.window_extrema.interval: '1 m' is a length but a duration (time) is required"),
        ({"q": {"window_extrema": {"interval": "1 sec"}}}, "hard.q.window_extrema.interval: unknown unit 'sec' in '1 sec'"),
        ({"q": {"window_extrema": {"interval": "0 s"}}}, "hard.q.window_extrema.interval: must be positive, got '0 s'"),
        ({"q": {"window_extrema": {"interval": -1}}}, "hard.q.window_extrema.interval: must be positive, got '-1'"),
        ({"q": {"local_extrema": {"prominence": "-1 Pa"}}}, "hard.q.local_extrema.prominence: must not be negative, got '-1 Pa'"),
        ({"q": {"local_extrema": {"prominence": 1, "separation": "-1 ms"}}}, "hard.q.local_extrema.separation: must not be negative, got '-1 ms'"),
        ({"q": {"threshold_crossing": {"value": 1, "hysteresis": -0.1}}}, "hard.q.threshold_crossing.hysteresis: must not be negative, got '-0.1'"),
        ({"q": {"threshold_crossing": {"value": 1, "debounce": "-2 s"}}}, "hard.q.threshold_crossing.debounce: must not be negative, got '-2 s'"),
        ({"q": {"threshold_crossing": {"value": 1, "tolerance": "5 kPa"}}}, "hard.q.threshold_crossing.tolerance: '5 kPa' is a pressure but a duration (time) is required"),
        ({"q": {"violation": {"above": 1, "min_duration": "-1 s"}}}, "hard.q.violation.min_duration: must not be negative, got '-1 s'"),
        ({"q": {"threshold_crossing": {"value": True}}}, "hard.q.threshold_crossing.value: expected a number or a quantity such as '100 ms', got a boolean"),
        ({"q": {"threshold_crossing": {"value": "high"}}}, "hard.q.threshold_crossing.value: expected a number with an optional unit such as '100 ms', got 'high'"),
        ({"q": {"threshold_crossing": {"value": "2 MiB"}}}, "hard.q.threshold_crossing.value: '2 MiB' is a byte size, which no signal value can be"),
        ({"": {"global_extrema": {}}}, "hard: names must be non-empty strings, got ''"),
    ],
)
def test_hard_errors(hard, message):
    assert one_error(pol(hard=hard)) == message


def test_negative_value_and_origin_allowed():
    policy = load_policy(pol(hard={"q": {"violation": {"below": "-5 Pa"}, "window_extrema": {"interval": 1, "origin": -3}}}))
    assert len(policy.hard) == 2


# ----- events -----------------------------------------------------------------------------


def test_event_full_and_defaults():
    policy = load_policy(
        pol(
            events={
                "MECO": {
                    "when": {"signal": "thrust", "falls_below": "100 N", "debounce": "250 ms", "hysteresis": "5 N"},
                    "occurrence": "last",
                    "expect": 1,
                    "keep": {"before": "5 s", "after": "10 s", "signals": ["thrust", "q_dyn"]},
                    "severity": "limit",
                },
                "LIFTOFF": {"when": {"signal": "thrust", "rises_above": 1}},
                "BOOST": {"when": {"signal": "mode", "equals": "BOOST"}},
                "MODE3": {"when": {"signal": "mode", "equals": 3}},
                "HALF": {"when": {"signal": "mode", "equals": 2.5}, "keep": None},
            }
        )
    )
    events = {e.name: e for e in policy.events}
    meco = events["MECO"]
    assert (meco.signal, meco.trigger, meco.value) == ("thrust", "falls_below", Quantity(100.0, "N", "100 N"))
    assert meco.debounce == Quantity(250.0, "ms", "250 ms") and meco.hysteresis == Quantity(5.0, "N", "5 N")
    assert (meco.occurrence, meco.expect, meco.severity) == ("last", 1, "limit")
    assert meco.before == Quantity(5.0, "s", "5 s") and meco.after == Quantity(10.0, "s", "10 s")
    assert meco.signals == ["thrust", "q_dyn"]
    lift = events["LIFTOFF"]
    assert lift.trigger == "rises_above" and lift.value == Quantity(1.0, None, "1")
    assert lift.hysteresis == Quantity(0.0, None, "0") and lift.debounce == Quantity(0.0, "s", "0 s")
    assert (lift.occurrence, lift.expect, lift.severity, lift.signals) == ("first", None, "info", None)
    assert lift.before == Quantity(0.0, "s", "0 s") and lift.after == Quantity(0.0, "s", "0 s")
    assert events["BOOST"].value == "BOOST" and events["BOOST"].hysteresis is None
    assert events["MODE3"].value == 3 and isinstance(events["MODE3"].value, int)
    assert events["HALF"].value == 2.5


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([], "events: expected a mapping, got a list"),
        ({"E": 5}, "events.E: expected a mapping, got 5"),
        ({"E": {}}, "events.E: missing required key 'when'"),
        ({"E": {"when": "x"}}, "events.E.when: expected a mapping, got 'x'"),
        ({"E": {"when": {"falls_below": 1}}}, "events.E.when: missing required key 'signal'"),
        ({"E": {"when": {"signal": "", "equals": 1}}}, "events.E.when.signal: expected a non-empty string, got ''"),
        ({"E": {"when": {"signal": "s"}}}, "events.E.when: missing trigger: give one of falls_below, rises_above or equals"),
        (
            {"E": {"when": {"signal": "s", "falls_below": 1, "rises_above": 2}}},
            "events.E.when: give exactly one of falls_below, rises_above or equals, got falls_below, rises_above",
        ),
        ({"E": {"when": {"signal": "s", "fall_below": 1, "equals": 1}}}, "events.E.when.fall_below: unknown key 'fall_below'; did you mean 'falls_below'?"),
        ({"E": {"when": {"signal": "s", "equals": True}}}, "events.E.when.equals: expected a number or a label string, got a boolean"),
        ({"E": {"when": {"signal": "s", "equals": ""}}}, "events.E.when.equals: expected a number or a label string, got ''"),
        ({"E": {"when": {"signal": "s", "equals": [1]}}}, "events.E.when.equals: expected a number or a label string, got a list"),
        ({"E": {"when": {"signal": "s", "equals": float("inf")}}}, "events.E.when.equals: expected a finite number or a label string, got inf"),
        ({"E": {"when": {"signal": "s", "equals": 1, "hysteresis": 1}}}, "events.E.when.hysteresis: hysteresis applies only to falls_below and rises_above"),
        ({"E": {"when": {"signal": "s", "equals": 1, "debounce": "1 s"}}}, "events.E.when.debounce: debounce applies only to falls_below and rises_above"),
        ({"E": {"when": {"signal": "s", "falls_below": 1, "hysteresis": -1}}}, "events.E.when.hysteresis: must not be negative, got '-1'"),
        ({"E": {"when": {"signal": "s", "falls_below": 1, "debounce": "1 m"}}}, "events.E.when.debounce: '1 m' is a length but a duration (time) is required"),
        ({"E": {"when": {"signal": "s", "falls_below": False}}}, "events.E.when.falls_below: expected a number or a quantity such as '100 ms', got a boolean"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "occurrence": "every"}}, "events.E.occurrence: expected one of first, last, all, got 'every'"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "expect": -1}}, "events.E.expect: expected a non-negative integer, got -1"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "expect": True}}, "events.E.expect: expected a non-negative integer, got a boolean"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "severity": "warn"}}, "events.E.severity: expected one of info, limit, got 'warn'"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "keep": []}}, "events.E.keep: expected a mapping, got a list"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "keep": {"befor": "1 s"}}}, "events.E.keep.befor: unknown key 'befor'; did you mean 'before'?"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "keep": {"before": "-1 s"}}}, "events.E.keep.before: must not be negative, got '-1 s'"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "keep": {"after": "1 kPa"}}}, "events.E.keep.after: '1 kPa' is a pressure but a duration (time) is required"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "keep": {"signals": "s"}}}, "events.E.keep.signals: expected a list of strings, got 's'"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "keep": {"signals": ["a", "a"]}}}, "events.E.keep.signals[1]: duplicate entry 'a'"),
        ({"E": {"when": {"signal": "s", "equals": 1}, "extra": 1}}, "events.E.extra: unknown key 'extra'; expected one of when, occurrence, expect, keep, severity"),
    ],
)
def test_event_errors(events, message):
    assert errors(pol(events=events)) == [message]


def test_event_value_with_unknown_unit_is_kept():
    policy = load_policy(pol(events={"E": {"when": {"signal": "s", "falls_below": "1 sec"}}}))
    assert policy.events[0].value == Quantity(1.0, "sec", "1 sec")


# ----- trajectories -----------------------------------------------------------------------


def test_trajectory_forms():
    policy = load_policy(
        pol(
            trajectories={
                "a": {"position": "pos", "max_position_error": "1 m", "max_time_error": "20 ms", "linked": ["aoa"]},
                "b": {"position": ["x", "y", "z"], "max_position_error": 0.5},
            }
        )
    )
    a, b = policy.trajectories
    assert a.position == "pos" and a.max_position_error == Quantity(1.0, "m", "1 m")
    assert a.max_time_error == Quantity(20.0, "ms", "20 ms") and a.linked == ["aoa"]
    assert b.position == ["x", "y", "z"] and b.max_position_error == Quantity(0.5, None, "0.5")
    assert b.max_time_error is None and b.linked == []


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (5, "trajectories.T: expected a mapping, got 5"),
        ({"max_position_error": "1 m"}, "trajectories.T: missing required key 'position'"),
        ({"position": ["x", "y"], "max_position_error": "1 m"}, "trajectories.T.position: expected exactly three signals [x, y, z], got 2"),
        ({"position": ["x", "y", "z", "w"], "max_position_error": "1 m"}, "trajectories.T.position: expected exactly three signals [x, y, z], got 4"),
        ({"position": ["x", "x", "z"], "max_position_error": "1 m"}, "trajectories.T.position[1]: duplicate entry 'x'"),
        ({"position": 5, "max_position_error": "1 m"}, "trajectories.T.position: expected a signal name or a list of three signal names, got 5"),
        ({"position": "p"}, "trajectories.T: missing required key 'max_position_error'"),
        ({"position": "p", "max_position_error": "1 s"}, "trajectories.T.max_position_error: '1 s' is a time but a length is required"),
        ({"position": "p", "max_position_error": "-1 m"}, "trajectories.T.max_position_error: must not be negative, got '-1 m'"),
        ({"position": "p", "max_position_error": "1 m", "max_time_error": "1 m"}, "trajectories.T.max_time_error: '1 m' is a length but a duration (time) is required"),
        ({"position": "p", "max_position_error": "1 m", "linked": "aoa"}, "trajectories.T.linked: expected a list of strings, got 'aoa'"),
        ({"position": "p", "max_position_error": "1 m", "eps": 1}, "trajectories.T.eps: unknown key 'eps'; expected one of position, max_position_error, max_time_error, linked"),
    ],
)
def test_trajectory_errors(spec, message):
    assert one_error(pol(trajectories={"T": spec})) == message


# ----- sync groups ------------------------------------------------------------------------


def test_sync_group():
    policy = load_policy(pol(sync_groups={"g": {"members": ["a", "b", "c"]}}))
    assert policy.sync_groups[0].name == "g" and policy.sync_groups[0].members == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (["a", "b"], "sync_groups.G: expected a mapping with members, got a list"),
        ({}, "sync_groups.G: missing required key 'members'"),
        ({"members": "a"}, "sync_groups.G.members: expected a list of strings, got 'a'"),
        ({"members": ["a"]}, "sync_groups.G.members: a sync group needs at least two members, got 1"),
        ({"members": ["a", "a"]}, "sync_groups.G.members[1]: duplicate entry 'a'"),
        ({"members": ["a", "b"], "member": 1}, "sync_groups.G.member: unknown key 'member'; did you mean 'members'?"),
    ],
)
def test_sync_group_errors(spec, message):
    assert one_error(pol(sync_groups={"G": spec})) == message


# ----- soft -------------------------------------------------------------------------------


def test_soft_rules():
    policy = load_policy(
        pol(soft=[{"match": "q_dyn", "priority": "high"}, {"match": "*", "priority": "none", "max_points": 0}, {"match": "x"}])
    )
    rules = [(r.match, r.priority, r.max_points) for r in policy.soft]
    assert rules == [("q_dyn", "high", None), ("*", "none", 0), ("x", "medium", None)]


@pytest.mark.parametrize(
    ("soft", "message"),
    [
        ({"match": "*"}, "soft: expected a list of rules, got a mapping"),
        (["*"], "soft[0]: expected a mapping with match, priority and max_points, got '*'"),
        ([{"priority": "low"}], "soft[0]: missing required key 'match'"),
        ([{"match": ""}], "soft[0].match: expected a non-empty string, got ''"),
        ([{"match": "*", "priority": "urgent"}], "soft[0].priority: expected one of high, medium, low, none, got 'urgent'"),
        ([{"match": "*", "priority": "hihg"}], "soft[0].priority: expected one of high, medium, low, none, got 'hihg'; did you mean 'high'?"),
        ([{"match": "*", "max_points": -1}], "soft[0].max_points: expected a non-negative integer, got -1"),
        ([{"match": "*", "max_points": 10.5}], "soft[0].max_points: expected a non-negative integer, got 10.5"),
        ([{"match": "*", "max_points": True}], "soft[0].max_points: expected a non-negative integer, got a boolean"),
        ([{"match": "*", "weight": 1}], "soft[0].weight: unknown key 'weight'; expected one of match, priority, max_points"),
    ],
)
def test_soft_errors(soft, message):
    assert one_error(pol(soft=soft)) == message


# ----- review -----------------------------------------------------------------------------


def test_review_defaults_from_hard():
    hard = {name: {"global_extrema": {}} for name in ("a", "b", "c", "d", "e")}
    hard["a"]["violation"] = {"above": 1}
    policy = load_policy(pol(hard=hard))
    assert policy.review.thumbnails == ["a", "b", "c", "d"]
    assert policy.review.kpis == []
    explicit = load_policy(pol(hard=hard, review={"thumbnails": ["e"], "kpis": ["a", "b"]}))
    assert explicit.review.thumbnails == ["e"] and explicit.review.kpis == ["a", "b"]
    partial = load_policy(pol(hard=hard, review={"kpis": ["a"]}))
    assert partial.review.thumbnails == ["a", "b", "c", "d"]


def test_default_thumbnails_are_order_independent():
    # The default is part of the canonical form, so it is sorted rather than taken in the order the
    # hard signals happen to appear in an unordered YAML or JSON mapping.
    policy = load_policy(pol(hard={name: {"global_extrema": {}} for name in ("e", "d", "c", "b", "a")}))
    assert policy.review.thumbnails == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    ("review", "message"),
    [
        ([], "review: expected a mapping, got a list"),
        ({"thumbnails": ["a", "b", "c", "d", "e"]}, "review.thumbnails: at most 4 entries are allowed, got 5"),
        ({"thumbnails": "a"}, "review.thumbnails: expected a list of strings, got 'a'"),
        ({"kpis": ["a", "a"]}, "review.kpis[1]: duplicate entry 'a'"),
        ({"thumbnail": ["a"]}, "review.thumbnail: unknown key 'thumbnail'; did you mean 'thumbnails'?"),
    ],
)
def test_review_errors(review, message):
    assert one_error(pol(review=review)) == message


def test_validate_policy_sets_source_fields():
    policy = validate_policy({"version": 1}, source_format="json", source_text="{}", locations={"version": "f:1:1"})
    assert policy.source_format == "json" and policy.source_text == "{}"
    assert policy.locations == {"version": "f:1:1"}
    assert len(policy.sha256) == 64
