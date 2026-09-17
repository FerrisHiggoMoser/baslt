"""Binding expressions to a run: name resolution, labels, units and the errors a person sees."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.reqs.expr import UNKNOWN, ExprError
from reference.reqs_runs import context

pytestmark = pytest.mark.minimal

T = np.linspace(0.0, 10.0, 11)
SIGNALS = {
    "aero__q": (T, T * 1000.0, "Pa"),
    "aero__alpha": (T, np.sin(T), "deg"),
    "nav__alpha": (T, np.cos(T), "rad"),
    "prop__thrust": (T, T * 10.0, "kN"),
    "nav__position": (T, np.column_stack([T, 2 * T, 3 * T]), "m", "vector"),
    "gnc__mode": (T, (T > 5).astype(np.int64)),
    "phase": (T, np.where(T > 3, "BURN", "IDLE")),
    "raw": (T, T),
    "odd": (T, T, "furlong"),
}
MAPPING = {
    "signals": {
        "q": {"path": "aero/q"},
        "qk": {"path": "aero/q", "unit": "kPa"},
        "mode": {"path": "gnc/mode", "kind": "discrete", "labels": {0: "IDLE", 1: "BURN"}},
        "alt": {"expr": "`nav/position`.z", "unit": "m"},
        "ratio": {"expr": "q / 2 kPa"},
        "loop_a": {"expr": "loop_b + 1"},
        "loop_b": {"expr": "loop_a + 1"},
        "mystery": {"expr": "q * q"},
        "declared": {"expr": "q * q", "unit": "kPa"},
    },
    "conditions": {"burning": "mode == 'BURN'", "high_q": "q > 5 kPa", "not_a_condition": "q + 1",
                   "both": "burning and high_q"},
    "events": {"MECO": {"signal": "thrust", "falls_below": "10 kN"}},
    "curves": {"qmax": {"points": [[0, 60], [5, 70]], "x_unit": "1", "y_unit": "kPa", "x": "ratio"}},
}


@pytest.fixture(scope="module")
def ctx():
    return context(SIGNALS, MAPPING)


def bound(ctx, text, role="check"):
    return ctx.binder.bind(text, role=role)


def test_names_resolve_in_order(ctx):
    assert bound(ctx, "q").signals == ["aero/q"]  # an alias
    assert bound(ctx, "`nav/alpha`").signals == ["nav/alpha"]  # a path
    assert bound(ctx, "thrust").signals == ["prop/thrust"]  # a unique leaf
    assert bound(ctx, "raw").signals == ["raw"]  # an exact name
    assert bound(ctx, "t").root.op == "tvar" and bound(ctx, "t").type.unit == "s"
    both = bound(ctx, "both")
    assert both.conditions == {"both", "burning", "high_q"} and both.signals == ["gnc/mode", "aero/q"]
    assert bound(ctx, "alt").signals == ["nav/position"] and bound(ctx, "alt").type.components == 1


@pytest.mark.parametrize(("text", "message"), [
    ("alpha", "ambiguous signal 'alpha'; it matches aero/alpha, nav/alpha"),
    ("thrustt", "unknown name 'thrustt'; did you mean 'thrust'?"),
    ("burnin", "unknown name 'burnin'; did you mean 'burning'?"),
    ("`aero/nope`", "unknown signal 'aero/nope'"),
    ("loop_a", "loop: loop_a -> loop_b -> loop_a"),
    ("not_a_condition", "is not a condition"),
    ("mode == 'COAST'", "'COAST' is not a label of this state; labels are IDLE, BURN"),
    ("mode == 'BRUN'", "did you mean 'BURN'"),
    ("mode > 'BURN'", "states can only be compared with == or !="),
    ("q == 'BURN'", "a text can only be compared with a state"),
    ("`gnc/mode` == 'BURN'", "a text can only be compared with a state"),
    ("q > 5 s", "a pressure (Pa) and a time (s) cannot be compared"),
    ("q + 1 deg", "a pressure (Pa) and an angle (deg) cannot be compared or added"),
    ("raw > 5 kPa", "5 kPa has a unit but the other side has none"),
    ("odd > 5 m", "are different units and at least one is unknown"),
    ("mystery > 5 kPa", "one side has a unit the language cannot work out"),
    ("position > 1", "compares single values"),
    ("q and burning", "and needs a condition"),
    ("not q", "not needs a condition"),
    ("-burning", "a sign needs a number"),
    ("mean(burning)", "mean() needs a number"),
    ("mean(`nav/position`)", "needs a single-valued signal"),
    ("mean(5)", "needs a signal"),
    ("duration(q)", "duration() needs a condition"),
    ("after('LOS')", "unknown event 'LOS'; define it under events:"),
    ("after('MECO#sometimes')", "after '#' write first, last, all"),
    ("after(q > 1)", "after() needs an event name"),
    ("after('MECO', q)", "takes a fixed duration"),
    ("after('MECO', 3 kPa)", "takes a duration, not 3 kPa"),
    ("at('MECO', 5)", "second argument"),
    ("curve(qmaxx, 1)", "unknown curve 'qmaxx'; did you mean 'qmax'?"),
    ("curve(q + 1, 1)", "takes the curve's name first"),
    ("sqrt(q)", "sqrt() needs a value without a unit, but it is in Pa"),
    ("sin(q)", "sin() needs an angle"),
    ("movmean(q, 0 s)", "positive window"),
    ("movmean(q, q)", "takes a fixed duration"),
    ("as_unit(q, 'furlong')", "unknown unit 'furlong'"),
    ("as_unit(q, 's')", "cannot be declared as s"),
    ("`nav/position`.x + `nav/position`[5]", "component 5 does not exist; the vector has 3"),
    ("q.x", "needs a vector signal"),
    ("mode in ('BURN', 'NOPE')", "'NOPE' is not a label"),
    ("q in (1 s, 2 s)", "cannot be compared"),
    ("(q > 1) > (q > 2)", "conditions can only be compared with == or !="),
])
def test_errors(ctx, text, message):
    with pytest.raises(ExprError) as caught:
        bound(ctx, text)
    assert message in str(caught.value)


def test_units_are_worked_out(ctx):
    cases = {
        "q": "Pa", "qk": "kPa", "q + 2 kPa": "Pa", "2 kPa + q": "Pa", "q * 2": "Pa", "2 * q": "Pa",
        "q / 2": "Pa", "q / 2 kPa": "1", "ratio": "1", "q * q": UNKNOWN, "declared": "kPa", "2 / q": UNKNOWN,
        "q ** 2": UNKNOWN, "q ** 1": "Pa", "deriv(`aero/alpha`)": "deg/s", "deriv(q)": "Pa/s",
        "deriv(raw)": None, "deriv(`odd`)": UNKNOWN, "integral(thrust)": UNKNOWN, "abs(q)": "Pa",
        "max(q, 3 kPa)": "Pa", "norm(`nav/position`)": "m", "since('MECO')": "s", "time('MECO')": "s",
        "count('MECO')": None, "duration(burning)": "s", "at('MECO', q)": "Pa", "curve(qmax)": "kPa",
        "sin(`aero/alpha`)": None, "as_unit(q * q, 'Pa')": "Pa", "as_unit(qk, 'Pa')": "Pa",
        "movmean(q, 200 ms)": "Pa", "where(burning, q, 0)": "Pa", "5 %": None,
        "clip(q, 0, 1 kPa)": "Pa", "hypot(q, qk)": "Pa", "mean(q)": "Pa", "integral(mystery)": UNKNOWN,
    }
    for text, unit in cases.items():
        assert bound(ctx, text).type.unit == unit, text


def test_quantities_are_converted_into_the_other_side(ctx):
    node = bound(ctx, "q > 5 kPa").root
    assert node.args[1].op == "num" and node.args[1].value == 5000.0
    node = bound(ctx, "qk < 700 Pa").root
    assert node.args[1].value == pytest.approx(0.7)
    node = bound(ctx, "q > qk").root
    assert node.args[1].op == "scale" and node.args[1].value == 1000.0
    node = bound(ctx, "2 kPa < q").root
    assert node.args[0].op == "num" and node.args[0].value == 2000.0
    node = bound(ctx, "sin(`aero/alpha`)").root
    assert node.args[0].op == "scale"  # degrees to radians


def test_shapes_and_types(ctx):
    assert bound(ctx, "q > 5").type.shape == "series" and bound(ctx, "q > 5").type.dtype == "bool"
    assert bound(ctx, "max(q) > 5").type.shape == "scalar"
    assert bound(ctx, "max(q)").type.shape == "scalar" and bound(ctx, "max(q, 5)").type.shape == "series"
    assert bound(ctx, "time('MECO') - 2 s").type.shape == "scalar"
    assert bound(ctx, "mode").type.dtype == "enum" and bound(ctx, "mode").type.labels == ("IDLE", "BURN")
    assert bound(ctx, "phase").type.dtype == "enum" and bound(ctx, "phase").type.labels == ("BURN", "IDLE")
    assert bound(ctx, "phase == 'BURN'").type.dtype == "bool"
    assert bound(ctx, "`nav/position`").type.components == 3
    assert bound(ctx, "`nav/position` * 2").type.components == 3


def test_bare_words_next_to_states_are_labels(ctx):
    result = bound(ctx, "mode == BURN")
    assert result.root.args[1].op == "label" and "BURN was read as the label 'BURN'" in result.notes[0]
    result = bound(ctx, "mode in (IDLE, BURN)")
    assert [arg.op for arg in result.root.args[1:]] == ["label", "label"]


def test_applies_to_sees_only_parameters(ctx):
    params = ctx.binder.bind("payload == 'heavy' and mass > 1000", role="applies_to")
    assert params.params == {"payload", "mass"} and params.signals == []
    with pytest.raises(ExprError, match="applies_to looks at run parameters"):
        ctx.binder.bind("`aero/q` > 1", role="applies_to")
    with pytest.raises(ExprError, match="cannot be used in applies_to"):
        ctx.binder.bind("mean(q) > 1", role="applies_to")


def test_parameter_names_can_be_checked():
    ctx = context({"a": (T, T)}, {"params": {"units": {"mass": "kg"}}})
    ctx.binder.param_names = {"payload", "mass"}
    assert ctx.binder.bind("param.mass > 1 t").root.args[1].value == 1000.0  # converted into the parameter unit
    with pytest.raises(ExprError, match="unknown run parameter 'payloda'; did you mean 'payload'?"):
        ctx.binder.bind("param.payloda == 'x'")
