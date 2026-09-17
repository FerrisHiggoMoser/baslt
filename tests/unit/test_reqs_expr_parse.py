"""Reading requirement expressions: the pre-lexer, the whitelist and the limits of the language."""

from __future__ import annotations

import pytest

from baslt.reqs.expr import ExprError, lex, parse

pytestmark = pytest.mark.minimal


@pytest.mark.parametrize(("text", "python"), [
    ("q > 60 kPa", 'q > _q(60, "kPa")'),
    ("q>60kPa", 'q>_q(60, "kPa")'),
    ("q > 60[kPa]", 'q > _q(60, "kPa")'),
    ("abs(alpha) >= 7 deg", 'abs(alpha) >= _q(7, "deg")'),
    ("`nav/position`.z", '_sig("nav/position").z'),
    ("`a b/c'd`", '_sig("a b/c\'d")'),
    ("speed / 340 m/s", 'speed / _q(340, "m/s")'),
    ("9.81 m/s^2 * t", '_q(9.81, "m/s^2") * t'),
    ("2 min", '_q(2, "min")'),
    ("200 ms", '_q(200, "ms")'),
    ("10 m*2", '_q(10, "m")*2'),
    ("1e3 Pa", '_q(1e3, "Pa")'),
    (".5 s", '_q(.5, "s")'),
    ("5 %", '_q(5, "%")'),
    ("5 in (1, 2)", "5 in (1, 2)"),
    ("x ≥ 5 AND y ≠ 3 || !z", "x >= 5  and  y != 3  or   not z"),
    ("a <> b", "a != b"),
    ("q = 5", "q == 5"),
    ("a^2 − b × c ÷ d · e", "a**2 - b * c / d * e"),
    ("mode == 'COAST'", 'mode == "COAST"'),
    ('x == "a\'b"', 'x == "a\'b"'),
    ("q2 > 3", "q2 > 3"),
    ("TRUE and False", "True  and  False"),
    ("q[2]", "q[2]"),
])
def test_lexing(text, python):
    assert lex(text) == python


@pytest.mark.parametrize(("text", "shape"), [
    ("q > 60 kPa", "cmp"),
    ("0.8 <= mach <= 1.2", "and"),
    ("x if y > 1 else z", "if"),
    ("mode in ('A', 'B')", "in"),
    ("mode not in ('A',)", "notin"),
    ("-q", "neg"),
    ("+q", "pos"),
    ("not a", "not"),
    ("max(q) - min(q)", "sub"),
    ("table(qmax, mach)", "call"),
    ("param.payload", "param"),
    ("`p`[1]", "comp"),
    ("2 ** 3", "pow"),
])
def test_parsing(text, shape):
    assert parse(text).op == shape


def test_chained_comparisons_become_pairs():
    node = parse("0.8 <= mach <= 1.2")
    assert [part.value for part in node.args] == ["<=", "<="]
    assert node.args[0].args[1] == node.args[1].args[0]


@pytest.mark.parametrize(("text", "message"), [
    ("", "empty"),
    ("   ", "empty"),
    ("2 mode", "invalid syntax"),
    ("import os", "invalid syntax"),
    ("lambda: 1", "unexpected character ':'"),
    ("x.__class__", "`.__class__` is not allowed"),
    ("q.real", "`.real` is not allowed"),
    ("open('f')", "unknown function open()"),
    ("maxx(q)", "unknown function maxx(); did you mean max()?"),
    ("abs(q, r)", "abs() takes 1 arguments, got 2"),
    ("between('a')", "between() takes 2 arguments, got 1"),
    ("q[1:2]", "unexpected character ':'"),
    ("q[x]", "whole-number component index"),
    ("[1, 2]", "only belongs after `in`"),
    ("mode in 'A'", "needs a list of values"),
    ("mode in (q + 1,)", "may only hold values"),
    ("q % 2", "unexpected character '%'"),
    ("q # comment", "unexpected character '#'"),
    ("'open", "not closed"),
    ("`open", "not closed"),
    ("``", "empty signal path"),
    ("q > 5[furlong]", "unknown unit 'furlong'"),
    ("q > 5[kPa", "not closed"),
    ("q is None", "None is not allowed"),
    ("q is q", "the comparison in 'q is q' is not allowed"),
    ("f(x)(y)", "only plain function calls"),
    ("(q, r)", "only belongs after `in`"),
    ("{1: 2}", "unexpected character '{'"),
    ("q > b'x'", "is not allowed in an expression"),
    ("1 if", "invalid syntax"),
])
def test_rejections(text, message):
    with pytest.raises(ExprError, match=message.replace("(", r"\(").replace(")", r"\)").replace("?", r"\?")
                       .replace("[", r"\[")):
        parse(text)


def test_size_limits():
    with pytest.raises(ExprError, match="longer than 2000 characters"):
        parse("q + " * 600 + "q")
    with pytest.raises(ExprError, match="more than 300 parts"):
        parse(" and ".join(f"q > {k}" for k in range(120)))
    with pytest.raises(ExprError, match="nested more than 40 levels"):
        parse("abs(" * 45 + "q" + ")" * 45)


def test_random_text_only_ever_raises_expr_errors():
    import random

    rng = random.Random(1)
    alphabet = "qxy012.,()[]`'\"+-*/<>=!&|^ %#:;{}abcminkPa\\\n\t≤≥±·"
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        try:
            parse(text)
        except ExprError:
            pass
