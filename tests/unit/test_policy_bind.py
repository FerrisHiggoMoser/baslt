"""Binding policies to source signals: names, selection, kinds, units, soft weights and budget."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from baslt.errors import PolicyError, UsageError
from baslt.policy import BoundPolicy, bind_policy, load_policy
from baslt.signals import SignalInfo

DOCS = Path(__file__).resolve().parents[2] / "docs" / "policy.md"


def doc_example() -> str:
    text = DOCS.read_text(encoding="utf-8")
    return re.search(r"## Complete example\s*```yaml\n(.*?)```", text, re.S).group(1)


def info(name, *, shape=(100,), dtype="<f8", unit=None, kind="continuous", n=None, time_ref="t", path=None):
    return SignalInfo(
        name=name,
        path=path if path is not None else "/" + name,
        shape=tuple(shape),
        dtype=dtype,
        unit=unit,
        kind=kind,
        n=n if n is not None else shape[0],
        time_ref=time_ref,
    )


def flight_infos():
    return [
        info("aero/q"),
        info("aero/alpha"),
        info("prop/thrust"),
        info("nav/position", shape=(100, 3), kind="vector"),
        info("gnc/mode", dtype="<i4", kind="discrete"),
        info("aux/temp", unit="K"),
    ]


def bind(data, infos=None, **kwargs) -> BoundPolicy:
    policy = data if not isinstance(data, dict) else load_policy(data)
    return bind_policy(policy, flight_infos() if infos is None else infos, **kwargs)


def bind_errors(data, infos=None, **kwargs) -> list[str]:
    with pytest.raises(PolicyError) as exc:
        bind(data, infos, **kwargs)
    return [issue.render() for issue in exc.value.issues]


def one_bind_error(data, infos=None) -> str:
    found = bind_errors(data, infos)
    assert len(found) == 1, found
    return found[0]


def pol(**sections):
    return {"version": 1, **sections}


# ----- full example -----------------------------------------------------------------------


def test_docs_example_binds():
    bound = bind(load_policy(doc_example(), format="yaml"))
    assert bound.included == ["aero/q", "aero/alpha", "prop/thrust", "nav/position", "gnc/mode", "aux/temp"]
    assert bound.aliases == {
        "q_dyn": "aero/q",
        "aoa": "aero/alpha",
        "thrust": "prop/thrust",
        "position": "nav/position",
        "flight_mode": "gnc/mode",
    }
    reqs = {r.id: r for r in bound.reqs}
    assert reqs["hard.q_dyn.global_extrema"].signal == "aero/q"
    assert reqs["hard.q_dyn.global_extrema"].params == {}
    assert reqs["hard.q_dyn.threshold_crossing[0]"].params == {
        "value": 65000.0,
        "edge": "both",
        "hysteresis": 1000.0,
        "debounce": pytest.approx(0.1),
        "tolerance": pytest.approx(0.005),
        "interpolate": "linear",
    }
    assert reqs["hard.q_dyn.violation[0]"].params == {"above": 70000.0, "min_duration": pytest.approx(0.1)}
    assert reqs["hard.q_dyn.violation[0]"].severity == "limit"
    assert reqs["hard.aoa.local_extrema"].params == {"prominence": 0.5, "separation": pytest.approx(0.1), "kind": "both"}
    assert reqs["hard.aoa.window_extrema"].params == {"interval": 1.0, "origin": 0.0}
    assert reqs["hard.aoa.threshold_crossing[0]"].params["value"] == -7.0
    assert reqs["hard.aoa.threshold_crossing[1]"].params["value"] == 7.0
    assert reqs["hard.flight_mode.state_transitions"].signal == "gnc/mode"
    assert [r.id for r in bound.reqs_by_signal["aero/alpha"]] == [
        "hard.aoa.local_extrema",
        "hard.aoa.window_extrema",
        "hard.aoa.threshold_crossing[0]",
        "hard.aoa.threshold_crossing[1]",
    ]
    assert bound.reqs_by_signal["aux/temp"] == []

    (meco,) = bound.events
    assert (meco.name, meco.signal, meco.trigger, meco.value) == ("MECO", "prop/thrust", "falls_below", 100.0)
    assert (meco.hysteresis, meco.debounce, meco.occurrence, meco.expect) == (0.0, 0.25, "first", 1)
    assert (meco.before, meco.after, meco.severity) == (5.0, 10.0, "info")
    assert meco.signals == ["prop/thrust", "aero/q", "aero/alpha", "gnc/mode"]

    (ascent,) = bound.trajectories
    assert ascent.position == ["nav/position"]
    assert (ascent.max_position_error, ascent.max_time_error, ascent.unit) == (1.0, pytest.approx(0.02), "m")
    assert ascent.linked == ["aero/alpha"]
    assert [(g.name, g.members) for g in bound.sync_groups] == [("flight_dynamics", ["aero/q", "aero/alpha"])]

    assert bound.soft == {
        "aero/q": (4, None),
        "aero/alpha": (1, 4000),
        "prop/thrust": (1, 4000),
        "nav/position": (1, 4000),
        "gnc/mode": (1, 4000),
        "aux/temp": (1, 4000),
    }
    assert (bound.max_bytes, bound.budget_source) == (2097152, "policy")
    assert bound.thumbnails == ["aero/q", "aero/alpha"] and bound.kpis == ["aero/q"]
    assert bound.units["aero/q"] == "Pa" and bound.units["aux/temp"] == "K"
    assert bound.kinds["gnc/mode"] == "discrete" and bound.kinds["nav/position"] == "vector"
    assert bound.load_options == {
        "time_hints": {},
        "global_time": "t",
        "time_scale": 1.0,
        "on_non_monotonic": "error",
        "units": {"aero/q": "Pa", "aero/alpha": "deg", "prop/thrust": "N", "nav/position": "m"},
        "kinds": {"gnc/mode": "discrete"},
    }


# ----- names ------------------------------------------------------------------------------


def test_name_resolution_forms():
    infos = [info("aero/q", unit="Pa"), info("aero/alpha"), info("x", path="/raw/x_data")]
    data = pol(
        hard={
            "/aero/q": {"global_extrema": {}},
            "alpha": {"global_extrema": {}},
            "raw/x_data": {"global_extrema": {}},
            "aero/q": {"violation": {"above": 1}},
        }
    )
    bound = bind(data, infos)
    assert [r.signal for r in bound.reqs] == ["aero/q", "aero/alpha", "x", "aero/q"]


def test_alias_takes_precedence_over_canonical_name():
    infos = [info("q"), info("aero/q2")]
    bound = bind(pol(signals={"decl": {"q": {"path": "aero/q2"}}}, hard={"q": {"global_extrema": {}}}), infos)
    assert bound.reqs[0].signal == "aero/q2"
    assert bound.aliases == {"q": "aero/q2"}


def test_decl_without_path_names_the_signal_itself():
    bound = bind(pol(signals={"decl": {"thrust": {"unit": "kN"}}}, hard={"thrust": {"violation": {"above": "5000 N"}}}))
    assert bound.aliases == {"thrust": "prop/thrust"}
    assert bound.reqs[0].params["above"] == pytest.approx(5.0)


def test_ambiguous_leaf_lists_candidates():
    infos = [info("sim/q"), info("aero/q")]
    assert one_bind_error(pol(hard={"q": {"global_extrema": {}}}), infos) == (
        "hard.q: ambiguous signal 'q'; it matches aero/q, sim/q. Use the full name or declare an alias in signals.decl"
    )


def test_unknown_signal_suggestion_matches_docs():
    data = load_policy(doc_example().replace("signal: thrust,", "signal: thrust_n,"), format="yaml")
    found = bind_errors(data)
    assert len(found) == 1
    assert found[0].startswith("events.MECO.when.signal: unknown signal 'thrust_n'; did you mean 'thrust'? (<string>:")


def test_unknown_signal_without_suggestion_and_reported_once():
    found = bind_errors(pol(hard={"zzzz": {"global_extrema": {}}}, sync_groups={"g": {"members": ["zzzz", "aero/q"]}}))
    assert found == ["hard.zzzz: unknown signal 'zzzz'"]


def test_dimension_mismatch_message_matches_docs():
    text = doc_example().replace("- {value: 65 kPa, edge: both", "- {value: 7 deg, edge: both")
    found = bind_errors(load_policy(text, format="yaml"))
    assert found == ["hard.q_dyn.threshold_crossing[0].value: '7 deg' is an angle but signal q_dyn is a pressure (Pa) (<string>:28:10)"]


def test_decl_errors():
    assert one_bind_error(pol(signals={"decl": {"q_dyn": {"path": "aero/qq"}}})) == (
        "signals.decl.q_dyn.path: unknown signal 'aero/qq'; did you mean 'aero/q'?"
    )
    assert one_bind_error(pol(signals={"decl": {"nothing_here": {"unit": "Pa"}}})) == (
        "signals.decl.nothing_here: unknown signal 'nothing_here'"
    )
    assert one_bind_error(pol(signals={"decl": {"a": {"path": "aero/q"}, "b": {"path": "/aero/q"}}})) == (
        "signals.decl.b: signal 'aero/q' is already declared by 'a'"
    )


def test_failed_alias_is_not_reported_again():
    data = pol(signals={"decl": {"q_dyn": {"path": "aero/nope"}}}, hard={"q_dyn": {"global_extrema": {}}})
    found = bind_errors(data)
    assert len(found) == 1 and found[0].startswith("signals.decl.q_dyn.path: unknown signal 'aero/nope'")


def test_decl_kind_shape_checks():
    infos = [info("x"), info("p", shape=(100, 3), kind="vector"), info("m", dtype="<U8", kind="discrete")]
    assert one_bind_error(pol(signals={"decl": {"x": {"kind": "vector"}}}), infos) == (
        "signals.decl.x.kind: signal 'x' has shape (100,); kind vector needs shape (n, k)"
    )
    assert one_bind_error(pol(signals={"decl": {"p": {"kind": "continuous"}}}), infos) == (
        "signals.decl.p.kind: signal 'p' has 3 components; only kind vector fits"
    )
    assert one_bind_error(pol(signals={"decl": {"m": {"kind": "continuous"}}}), infos) == (
        "signals.decl.m.kind: signal 'm' holds strings; only kind discrete fits"
    )
    bound = bind(pol(signals={"decl": {"x": {"kind": "discrete"}}}), infos)
    assert bound.kinds["x"] == "discrete"


# ----- selection --------------------------------------------------------------------------


def test_include_globs_and_referenced_signals():
    data = pol(signals={"include": ["aero/*"]}, hard={"prop/thrust": {"global_extrema": {}}})
    bound = bind(data)
    assert bound.included == ["aero/q", "aero/alpha", "prop/thrust"]
    assert set(bound.soft) == set(bound.included)
    assert set(bound.reqs_by_signal) == set(bound.included)


def test_include_matches_canonical_names_only_and_review_references():
    # docs/policy.md scopes include/exclude to "glob patterns over canonical signal names"; only
    # soft rules match aliases as well. 'q_dyn' is an alias, so it selects nothing.
    data = pol(
        signals={"include": ["q_dyn"], "decl": {"q_dyn": {"path": "aero/q"}}},
        review={"thumbnails": ["gnc/mode"], "kpis": ["temp", "aux/temp"]},
    )
    bound = bind(data)
    assert bound.included == ["gnc/mode", "aux/temp"]
    assert bound.thumbnails == ["gnc/mode"] and bound.kpis == ["aux/temp"]
    canonical = pol(signals={"include": ["aero/q"], "decl": {"q_dyn": {"path": "aero/q"}}})
    assert bind(canonical).included == ["aero/q"]


def test_exclude_globs_over_canonical_names():
    data = pol(signals={"exclude": ["aero/*", "gnc/mode"]})
    assert bind(data).included == ["prop/thrust", "nav/position", "aux/temp"]


def test_exclude_ignores_aliases():
    # 'mode' names gnc/mode only through signals.decl, so it excludes nothing.
    data = pol(signals={"exclude": ["mode"], "decl": {"mode": {"path": "gnc/mode"}}})
    assert bind(data).included == ["aero/q", "aero/alpha", "prop/thrust", "nav/position", "gnc/mode", "aux/temp"]


def test_every_reference_kind_includes_signals():
    data = pol(
        signals={"include": []},
        hard={"aero/q": {"global_extrema": {}}},
        events={"E": {"when": {"signal": "gnc/mode", "equals": 2}, "keep": {"signals": ["aux/temp"]}}},
        trajectories={"T": {"position": "nav/position", "max_position_error": 1, "linked": ["aero/alpha"]}},
        sync_groups={"G": {"members": ["aero/q", "prop/thrust"]}},
    )
    assert bind(data).included == ["aero/q", "aero/alpha", "prop/thrust", "nav/position", "gnc/mode", "aux/temp"]


def test_excluded_but_referenced_is_an_error():
    found = bind_errors(
        pol(
            signals={"exclude": ["aero/*"], "decl": {"q_dyn": {"path": "aero/q"}}},
            hard={"q_dyn": {"global_extrema": {}}, "aero/alpha": {"global_extrema": {}}},
            review={"kpis": ["q_dyn"]},
        )
    )
    assert found == [
        "hard.q_dyn: signal q_dyn (aero/q) is excluded by signals.exclude but referenced here",
        "hard.aero/alpha: signal aero/alpha is excluded by signals.exclude but referenced here",
    ]


def test_alias_pattern_does_not_exclude_its_signal():
    # 'thr*' matches the alias only, never the canonical name prop/thrust, so nothing is dropped.
    data = pol(
        signals={"exclude": ["thr*"], "decl": {"thrust": {"path": "prop/thrust"}}},
        sync_groups={"g": {"members": ["prop/thrust", "aero/q"]}},
    )
    assert "prop/thrust" in bind(data).included
    # A pattern over the canonical name does exclude it, and referencing it is then an error.
    canonical = pol(
        signals={"exclude": ["prop/*"], "decl": {"thrust": {"path": "prop/thrust"}}},
        sync_groups={"g": {"members": ["prop/thrust", "aero/q"]}},
    )
    assert bind_errors(canonical) == [
        "sync_groups.g.members[0]: signal prop/thrust is excluded by signals.exclude but referenced here"
    ]


# ----- kinds ------------------------------------------------------------------------------


def test_kind_mismatches():
    found = bind_errors(
        pol(
            hard={
                "gnc/mode": {"threshold_crossing": [{"value": 1}, {"value": 2}], "local_extrema": {"prominence": 1}},
                "aero/q": {"state_transitions": {}},
            },
            events={"E": {"when": {"signal": "aero/alpha", "equals": 1}}},
        )
    )
    assert found == [
        "hard.gnc/mode.threshold_crossing: threshold_crossing needs a continuous or vector signal, but gnc/mode is discrete",
        "hard.gnc/mode.local_extrema: local_extrema needs a continuous or vector signal, but gnc/mode is discrete",
        "hard.aero/q.state_transitions: state_transitions needs a discrete signal, but aero/q is continuous",
        "events.E.when.equals: equals needs a discrete signal, but aero/alpha is continuous",
    ]


def test_allowed_kinds():
    data = pol(
        hard={
            "nav/position": {"threshold_crossing": {"value": "1 km"}, "window_extrema": {"interval": 1}, "violation": {"below": 0}},
            "gnc/mode": {"global_extrema": {}, "state_transitions": {}},
        },
        events={"E": {"when": {"signal": "gnc/mode", "rises_above": 2}}},
    )
    infos = flight_infos()
    infos[3].unit = "m"
    bound = bind(data, infos)
    assert bound.reqs[0].params["value"] == 1000.0
    assert len(bound.reqs) == 5


def test_decl_kind_override_enables_op():
    infos = [info("mode", dtype="<i4", kind="continuous")]
    with pytest.raises(PolicyError):
        bind(pol(hard={"mode": {"state_transitions": {}}}), infos)
    bound = bind(pol(signals={"decl": {"mode": {"kind": "discrete"}}}, hard={"mode": {"state_transitions": {}}}), infos)
    assert bound.reqs[0].op == "state_transitions"
    assert bound.load_options["kinds"] == {"mode": "discrete"}


# ----- units ------------------------------------------------------------------------------


def test_absolute_versus_delta_temperature():
    infos = [info("temp_k", unit="K"), info("temp_f", unit="degF"), info("temp_c", unit="degC")]
    data = pol(
        hard={
            "temp_k": {"threshold_crossing": {"value": "20 degC", "hysteresis": "2 degC"}},
            "temp_f": {"threshold_crossing": {"value": "100 degC", "hysteresis": "1 degC"}},
            "temp_c": {"violation": {"above": "300 K"}, "local_extrema": {"prominence": "9 degF"}},
        }
    )
    reqs = {r.id: r.params for r in bind(data, infos).reqs}
    assert reqs["hard.temp_k.threshold_crossing"]["value"] == pytest.approx(293.15)
    assert reqs["hard.temp_k.threshold_crossing"]["hysteresis"] == pytest.approx(2.0)
    assert reqs["hard.temp_f.threshold_crossing"]["value"] == pytest.approx(212.0)
    assert reqs["hard.temp_f.threshold_crossing"]["hysteresis"] == pytest.approx(1.8)
    assert reqs["hard.temp_c.violation"]["above"] == pytest.approx(26.85)
    assert reqs["hard.temp_c.local_extrema"]["prominence"] == pytest.approx(5.0)


def test_bare_numbers_are_in_the_signal_unit():
    infos = [info("p", unit="kPa"), info("u")]
    data = pol(hard={"p": {"threshold_crossing": {"value": 65, "hysteresis": 0.5}}, "u": {"violation": {"above": 3}}})
    reqs = bind(data, infos).reqs
    assert reqs[0].params["value"] == 65.0 and reqs[0].params["hysteresis"] == 0.5
    assert reqs[1].params["above"] == 3.0


def test_missing_unit_rejects_quantities():
    infos = [info("u")]
    assert one_bind_error(pol(hard={"u": {"threshold_crossing": {"value": "65 kPa"}}}), infos) == (
        "hard.u.threshold_crossing.value: '65 kPa' has a unit but signal u has no unit; "
        "write a bare number or declare its unit in signals.decl"
    )
    # A typo or a nonsense unit is reported as such, with its suggestion, rather than as the
    # signal having no unit.
    assert one_bind_error(pol(hard={"u": {"threshold_crossing": {"value": "65 kpa"}}}), infos) == (
        "hard.u.threshold_crossing.value: unknown unit 'kpa' in '65 kpa'; did you mean 'kPa'?"
    )
    assert one_bind_error(pol(hard={"u": {"violation": {"above": "65 zzzzzz"}}}), infos) == (
        "hard.u.violation.above: unknown unit 'zzzzzz' in '65 zzzzzz'"
    )


def test_decl_unit_overrides_source_unit():
    infos = [info("p", unit="Pa")]
    bound = bind(pol(signals={"decl": {"p": {"unit": "kPa"}}}, hard={"p": {"threshold_crossing": {"value": "65 kPa", "hysteresis": "500 Pa"}}}), infos)
    assert bound.reqs[0].params["value"] == 65.0
    assert bound.reqs[0].params["hysteresis"] == pytest.approx(0.5)
    assert bound.units == {"p": "kPa"}
    assert bound.load_options["units"] == {"p": "kPa"}


def test_opaque_units():
    infos = [info("c", unit="counts"), info("k", unit="kpa")]
    bound = bind(pol(hard={"c": {"threshold_crossing": {"value": "12 counts", "hysteresis": 2}}}), infos)
    assert bound.reqs[0].params == {
        "value": 12.0,
        "edge": "both",
        "hysteresis": 2.0,
        "debounce": 0.0,
        "tolerance": 0.0,
        "interpolate": "linear",
    }
    assert one_bind_error(pol(hard={"c": {"violation": {"above": "5 kPa"}}}), infos) == (
        "hard.c.violation.above: '5 kPa' does not match the unit of signal c: 'counts' is not a known unit, "
        "so values must be bare numbers or use exactly 'counts'"
    )
    assert one_bind_error(pol(hard={"c": {"violation": {"above": "5 cnt"}}}), infos).startswith(
        "hard.c.violation.above: '5 cnt' does not match the unit of signal c"
    )
    assert one_bind_error(pol(hard={"k": {"violation": {"above": "5 kPa"}}}), infos) == (
        "hard.k.violation.above: '5 kPa' does not match the unit of signal k: 'kpa' is not a known unit, "
        "so values must be bare numbers or use exactly 'kpa' (did you mean to declare unit 'kPa'?)"
    )
    assert bind(pol(hard={"k": {"violation": {"above": "5 kpa"}}}), infos).reqs[0].params["above"] == 5.0


def test_unknown_policy_unit_on_known_signal():
    infos = [info("p", unit="Pa")]
    assert one_bind_error(pol(hard={"p": {"violation": {"above": "5 kpa"}}}), infos) == (
        "hard.p.violation.above: unknown unit 'kpa' in '5 kpa'; did you mean 'kPa'?"
    )


def test_durations_become_seconds():
    bound = bind(pol(hard={"aero/q": {"window_extrema": {"interval": "2 min", "origin": "-500 ms"}}}))
    assert bound.reqs[0].params == {"interval": 120.0, "origin": -0.5}


def test_vector_values_convert_per_signal_unit():
    infos = [info("v", shape=(10, 2), kind="vector", unit="m/s")]
    bound = bind(pol(hard={"v": {"violation": {"above": "36 km/h"}}}), infos)
    assert bound.reqs[0].params["above"] == pytest.approx(10.0)


def test_several_unit_issues_are_collected():
    infos = [info("p", unit="Pa")]
    found = bind_errors(pol(hard={"p": {"threshold_crossing": [{"value": "1 m"}, {"value": "1 s", "hysteresis": "1 deg"}]}}), infos)
    assert found == [
        "hard.p.threshold_crossing[0].value: '1 m' is a length but signal p is a pressure (Pa)",
        "hard.p.threshold_crossing[1].value: '1 s' is a time but signal p is a pressure (Pa)",
        "hard.p.threshold_crossing[1].hysteresis: '1 deg' is an angle but signal p is a pressure (Pa)",
    ]


def test_bind_issue_limit():
    data = pol(hard={f"missing_{i}": {"global_extrema": {}} for i in range(70)})
    with pytest.raises(PolicyError) as exc:
        bind(data)
    assert len(exc.value.issues) == 50


# ----- events -----------------------------------------------------------------------------


def test_event_conversions_and_defaults():
    infos = flight_infos()
    infos[2].unit = "N"
    data = pol(
        signals={"include": ["prop/*", "aero/q"], "decl": {"thrust": {"path": "prop/thrust"}}},
        events={
            "A": {
                "when": {"signal": "prop/thrust", "falls_below": "0.1 kN", "hysteresis": "10 N", "debounce": "250 ms"},
                "keep": {"before": "1 min", "after": 2},
            },
            "B": {"when": {"signal": "gnc/mode", "equals": "BOOST"}, "occurrence": "all", "severity": "limit"},
            "C": {"when": {"signal": "gnc/mode", "equals": 3}, "keep": {"signals": ["thrust", "prop/thrust", "q"]}},
        },
    )
    bound = bind(data, infos)
    a, b, c = bound.events
    assert (a.value, a.hysteresis, a.debounce, a.before, a.after) == (pytest.approx(100.0), 10.0, 0.25, 60.0, 2.0)
    assert a.signals == ["aero/q", "prop/thrust", "gnc/mode"] == bound.included
    assert (b.value, b.hysteresis, b.debounce, b.occurrence, b.severity) == ("BOOST", 0.0, 0.0, "all", "limit")
    assert c.value == 3 and isinstance(c.value, int)
    assert c.signals == ["prop/thrust", "aero/q"]


def test_event_unit_errors():
    infos = [info("f", unit="N")]
    found = bind_errors(pol(events={"E": {"when": {"signal": "f", "rises_above": "5 Pa", "hysteresis": "1 m"}}}), infos)
    assert found == [
        "events.E.when.rises_above: '5 Pa' is a pressure but signal f is a force (N)",
        "events.E.when.hysteresis: '1 m' is a length but signal f is a force (N)",
    ]


# ----- trajectories -----------------------------------------------------------------------


def test_trajectory_vector_form():
    infos = [info("pos", shape=(50, 3), kind="vector", unit="km"), info("aoa")]
    data = pol(trajectories={"T": {"position": "pos", "max_position_error": "10 m", "max_time_error": "20 ms", "linked": ["aoa"]}})
    (tr,) = bind(data, infos).trajectories
    assert tr.position == ["pos"] and tr.unit == "km"
    assert tr.max_position_error == pytest.approx(0.01)
    assert tr.max_time_error == pytest.approx(0.02)
    assert tr.linked == ["aoa"]


def test_trajectory_three_scalars():
    infos = [info("x", unit="m"), info("y", unit="m"), info("z", unit="m"), info("other/z", unit="m")]
    data = pol(
        signals={"decl": {"up": {"path": "z"}}},
        trajectories={"T": {"position": ["x", "y", "up"], "max_position_error": "1 ft"}},
    )
    (tr,) = bind(data, infos).trajectories
    assert tr.position == ["x", "y", "z"]
    assert tr.max_position_error == pytest.approx(0.3048)
    assert tr.max_time_error is None and tr.unit == "m"


def test_trajectory_unitless_position():
    infos = [info("pos", shape=(50, 3), kind="vector")]
    (tr,) = bind(pol(trajectories={"T": {"position": "pos", "max_position_error": 2}}), infos).trajectories
    assert tr.max_position_error == 2.0 and tr.unit is None
    assert one_bind_error(pol(trajectories={"T": {"position": "pos", "max_position_error": "2 m"}}), infos) == (
        "trajectories.T.max_position_error: '2 m' has a unit but position signal pos has no unit; "
        "write a bare number or declare its unit in signals.decl"
    )


@pytest.mark.parametrize(
    ("infos", "position", "message"),
    [
        (
            [info("pos", shape=(50, 2), kind="vector", unit="m")],
            "pos",
            "trajectories.T.position: position signal pos must be a vector signal with shape (n, 3), got vector with shape (50, 2)",
        ),
        (
            [info("pos", unit="m")],
            "pos",
            "trajectories.T.position: position signal pos must be a vector signal with shape (n, 3), got continuous with shape (100,)",
        ),
        (
            [info("pos", shape=(50, 3), kind="vector", unit="deg")],
            "pos",
            "trajectories.T.position: position signal pos is an angle (deg) but a length is required",
        ),
        (
            [info("x", unit="m"), info("y", unit="m"), info("z", dtype="<i4", kind="discrete", unit="m")],
            ["x", "y", "z"],
            "trajectories.T.position[2]: position component z must be a continuous scalar signal, got discrete",
        ),
        (
            [info("x", unit="m"), info("y", unit="m"), info("z", shape=(50,), unit="m")],
            ["x", "y", "z"],
            "trajectories.T.position: position components must share one clock, but they have 100, 100, 50 samples",
        ),
        (
            [info("x", unit="m"), info("y", unit="m"), info("z", unit="m", time_ref="t2")],
            ["x", "y", "z"],
            "trajectories.T.position: position components must share one clock, but they use t, t, t2",
        ),
        (
            [info("x", unit="m"), info("y", unit="m"), info("z", unit="km")],
            ["x", "y", "z"],
            "trajectories.T.position: position components have different units: 'm', 'm', 'km'",
        ),
        (
            [info("x", unit="deg"), info("y", unit="deg"), info("z", unit="deg")],
            ["x", "y", "z"],
            "trajectories.T.position: position [x, y, z] is an angle (deg) but a length is required",
        ),
    ],
)
def test_trajectory_errors(infos, position, message):
    data = pol(trajectories={"T": {"position": position, "max_position_error": "1 m"}})
    assert one_bind_error(data, infos) == message


def test_trajectory_components_resolving_to_one_signal():
    infos = [info("x", unit="m"), info("y", unit="m")]
    data = pol(signals={"decl": {"a": {"path": "x"}}}, trajectories={"T": {"position": ["x", "a", "y"], "max_position_error": "1 m"}})
    assert one_bind_error(data, infos) == (
        "trajectories.T.position: position components must be three different signals, got x, x, y"
    )


def test_trajectory_unknown_time_ref_is_not_an_error():
    infos = [info("x", unit="m", time_ref=None), info("y", unit="m"), info("z", unit="m", time_ref=None)]
    data = pol(trajectories={"T": {"position": ["x", "y", "z"], "max_position_error": "1 m"}})
    assert bind(data, infos).trajectories[0].position == ["x", "y", "z"]
    mixed = [info("x", unit="m", time_ref=None), info("y", unit="m"), info("z", unit="m", time_ref="t")]
    assert bind(data, mixed).trajectories[0].position == ["x", "y", "z"]


def test_trajectory_unknown_clock_does_not_hide_a_mismatch():
    # contracts.md: "All position components share one clock." An unknown third time_ref must not
    # switch off the check for the two that are known and different.
    infos = [info("x", unit="m", time_ref="ta"), info("y", unit="m", time_ref="tb"), info("z", unit="m", time_ref=None)]
    data = pol(trajectories={"T": {"position": ["x", "y", "z"], "max_position_error": "1 m"}})
    assert one_bind_error(data, infos) == (
        "trajectories.T.position: position components must share one clock, but they use ta, tb, unknown"
    )


def test_trajectory_decl_time_overrides_clock():
    infos = [info("x", unit="m"), info("y", unit="m"), info("z", unit="m", time_ref="t2")]
    data = pol(
        signals={"decl": {"z": {"time": "t"}}},
        trajectories={"T": {"position": ["x", "y", "z"], "max_position_error": "1 m"}},
    )
    assert bind(data, infos).trajectories[0].position == ["x", "y", "z"]


def test_trajectory_linked_unknown():
    infos = [info("pos", shape=(50, 3), kind="vector", unit="m")]
    data = pol(trajectories={"T": {"position": "pos", "max_position_error": "1 m", "linked": ["nope"]}})
    assert one_bind_error(data, infos) == "trajectories.T.linked[0]: unknown signal 'nope'"


# ----- sync groups ------------------------------------------------------------------------


def test_sync_group_members_same_signal():
    data = pol(signals={"decl": {"q_dyn": {"path": "aero/q"}}}, sync_groups={"G": {"members": ["q_dyn", "aero/q"]}})
    assert one_bind_error(data) == "sync_groups.G.members[1]: members 'q_dyn' and 'aero/q' are the same signal (aero/q)"


# ----- soft -------------------------------------------------------------------------------


def test_soft_first_matching_rule_wins():
    data = pol(
        signals={"decl": {"q_dyn": {"path": "aero/q"}}},
        soft=[
            {"match": "aero/*", "priority": "low"},
            {"match": "q_dyn", "priority": "high"},
            {"match": "prop/*", "priority": "none", "max_points": 10},
            {"match": "*", "priority": "high", "max_points": 99},
        ],
    )
    soft = bind(data).soft
    assert soft["aero/q"] == (1, None)
    assert soft["aero/alpha"] == (1, None)
    assert soft["prop/thrust"] == (0, 10)
    assert soft["gnc/mode"] == (4, 99)


def test_soft_alias_match_and_default_medium():
    data = pol(signals={"decl": {"q_dyn": {"path": "aero/q"}}}, soft=[{"match": "q_dyn", "priority": "high"}, {"match": "Aero/*", "priority": "low"}])
    soft = bind(data).soft
    assert soft["aero/q"] == (4, None)
    assert soft["aero/alpha"] == (2, None)  # globs are case-sensitive
    assert bind(pol()).soft == {name: (2, None) for name in bind(pol()).included}


# ----- budget and load options ------------------------------------------------------------


def test_max_bytes_sources():
    assert (bind(pol()).max_bytes, bind(pol()).budget_source) == (None, "none")
    with_policy = pol(artifact={"max_size": "1 KiB"})
    assert (bind(with_policy).max_bytes, bind(with_policy).budget_source) == (1024, "policy")
    over = bind(with_policy, max_bytes_override=5000)
    assert (over.max_bytes, over.budget_source) == (5000, "cli")
    assert bind(pol(), max_bytes_override=77).budget_source == "cli"


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2 MiB"])
def test_invalid_max_bytes_override(bad):
    with pytest.raises(UsageError, match="positive number of bytes"):
        bind(pol(), max_bytes_override=bad)


def test_load_options_from_declarations():
    infos = [info("aero/q"), info("gnc/mode", dtype="<i4", kind="continuous"), info("clock/tt")]
    data = pol(
        signals={
            "time": "/sim/t",
            "time_unit": "ms",
            "on_non_monotonic": "sort",
            "decl": {
                "q": {"path": "aero/q", "unit": "Pa", "time": "/aero/t"},
                "mode": {"path": "gnc/mode", "kind": "discrete", "time": "tt"},
            },
        }
    )
    options = bind(data, infos).load_options
    assert options == {
        "time_hints": {"aero/q": "aero/t", "gnc/mode": "clock/tt"},
        "global_time": "sim/t",
        "time_scale": 1e-3,
        "on_non_monotonic": "sort",
        "units": {"aero/q": "Pa"},
        "kinds": {"gnc/mode": "discrete"},
    }
    assert bind(pol(signals={"time_unit": "min"}), infos).load_options["time_scale"] == 60.0
    assert bind(pol(), infos).load_options["global_time"] is None


def test_yaml_locations_in_bind_errors():
    text = "version: 1\nhard:\n  nope:\n    global_extrema: {}\n"
    with pytest.raises(PolicyError) as exc:
        bind(load_policy(text, format="yaml"))
    assert exc.value.issues[0].location == "<string>:3:3"


def test_time_reference_through_alias():
    infos = [info("aero/q"), info("clock/t")]
    data = pol(signals={"time": "clk", "decl": {"clk": {"path": "clock/t"}, "q": {"path": "aero/q", "time": "clk"}}})
    options = bind(data, infos).load_options
    assert options["global_time"] == "clock/t"
    assert options["time_hints"] == {"aero/q": "clock/t"}


def test_reference_errors_in_event_keep_and_trajectory_positions():
    infos = [info("f", unit="N"), info("pos", shape=(50, 3), kind="vector", unit="m"), info("x", unit="m")]
    found = bind_errors(
        pol(
            events={"E": {"when": {"signal": "f", "falls_below": 1}, "keep": {"signals": ["nope_a"]}}},
            trajectories={
                "A": {"position": "nope_b", "max_position_error": 1},
                "B": {"position": ["x", "nope_c", "nope_d"], "max_position_error": 1},
            },
        ),
        infos,
    )
    assert found == [
        "events.E.keep.signals[0]: unknown signal 'nope_a'",
        "trajectories.A.position: unknown signal 'nope_b'",
        "trajectories.B.position[1]: unknown signal 'nope_c'",
        "trajectories.B.position[2]: unknown signal 'nope_d'",
    ]


def test_bind_does_not_mutate_policy():
    policy = load_policy(doc_example(), format="yaml")
    before = (policy.sha256, [e.signals for e in policy.events])
    bind(policy)
    bind(policy, max_bytes_override=10)
    assert (policy.sha256, [e.signals for e in policy.events]) == before
