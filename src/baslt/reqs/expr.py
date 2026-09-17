"""The expression language of requirement checks: reading, name resolution, types and units.

An expression is written the way engineers write it and read in three steps:

1. `lex` rewrites the text into Python syntax: `` `aero/q` `` becomes a signal reference, `60 kPa` and `60[kPa]`
   become quantities, `≤ ≥ ≠ − × ÷ ^ AND OR NOT && || !` and a lone `=` become their Python spelling.
2. `parse` reads it with `ast.parse` and keeps only a fixed set of node types (arithmetic, comparisons,
   and/or/not, conditional expressions, calls to the functions below, names, constants, integer indexing,
   `.x/.y/.z` and `param.<name>`). Nothing is evaluated by Python; the size and nesting are capped.
3. `Binder.bind` resolves every name (reserved names, aliases and named conditions, then signals), checks types
   and works out units, so that a limit written in kPa can be compared with a signal in Pa.

Evaluation on a run lives in `evaluate.py`; this module does not need numpy.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..units import Quantity
from .units_ext import UnitsError, UnitTable

__all__ = [
    "AGGREGATES",
    "EVENT_FUNCTIONS",
    "FUNCTIONS",
    "UNKNOWN",
    "B",
    "Binder",
    "BoundExpr",
    "ExprError",
    "N",
    "TypeInfo",
    "lex",
    "parse",
]

UNKNOWN = "?"  # a unit the language cannot work out; declare it with as_unit() or an alias unit
MAX_CHARS = 2000
MAX_NODES = 300
MAX_DEPTH = 40

ELEMENTWISE = {"abs": (1, 1), "sign": (1, 1), "clip": (3, 3), "where": (3, 3), "hypot": (2, 2), "norm": (1, 1),
               "sqrt": (1, 1), "exp": (1, 1), "log": (1, 1), "sin": (1, 1), "cos": (1, 1), "tan": (1, 1),
               "deriv": (1, 1), "movmean": (2, 2), "as_unit": (2, 2), "curve": (1, 2)}
AGGREGATES = {"initial": (1, 1), "final": (1, 1), "mean": (1, 1), "rms": (1, 1), "integral": (1, 1),
              "duration": (1, 1)}
MINMAX = {"min": (1, 32), "max": (1, 32)}
EVENT_FUNCTIONS = {"after": (1, 2), "before": (1, 2), "between": (2, 2), "during": (2, 2), "since": (1, 1),
                   "time": (1, 1), "count": (1, 1), "at": (2, 2)}
FUNCTIONS = {**ELEMENTWISE, **AGGREGATES, **MINMAX, **EVENT_FUNCTIONS}
RESERVED = {"t", "param", "true", "false", "True", "False", "and", "or", "not", "in"}
COMPONENTS = {"x": 0, "y": 1, "z": 2}

_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNIT_RUN_RE = re.compile(r"[A-Za-z°µ%/^*·²³0-9]+")
_WORDS = {"and": " and ", "or": " or ", "not": " not ", "true": "True", "false": "False"}


class ExprError(ValueError):
    """An expression that cannot be read, resolved or evaluated; the message says why and where."""


# ------------------------------------------------------------------------------------------------------------
# Reading


def lex(text: str, units: UnitTable | None = None) -> str:
    """The expression in Python syntax, with signal references and quantities as calls to `_sig` and `_q`."""
    units = units or UnitTable()
    if len(text) > MAX_CHARS:
        raise ExprError(f"the expression is longer than {MAX_CHARS} characters")
    out: list[str] = []
    i = 0
    n = len(text)
    previous = ""  # the last significant character copied

    def emit(chunk: str) -> None:
        nonlocal previous
        out.append(chunk)
        stripped = chunk.strip()
        if stripped:
            previous = stripped[-1]

    while i < n:
        ch = text[i]
        if ch.isspace():
            out.append(" ")
            i += 1
            continue
        if ch in "'\"":
            end = text.find(ch, i + 1)
            if end < 0:
                raise ExprError(f"a text value opened at column {i + 1} is not closed")
            emit(json.dumps(text[i + 1:end], ensure_ascii=False))
            i = end + 1
            continue
        if ch == "`":
            end = text.find("`", i + 1)
            if end < 0:
                raise ExprError(f"a signal path opened at column {i + 1} is not closed")
            path = text[i + 1:end].strip()
            if not path:
                raise ExprError(f"an empty signal path at column {i + 1}")
            emit(f"_sig({json.dumps(path, ensure_ascii=False)})")
            i = end + 1
            continue
        number = _NUMBER_RE.match(text, i) if (ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit())) \
            else None
        if number is not None and not (previous.isalnum() or previous == "_"):
            value = number.group(0)
            j = number.end()
            if j < n and text[j] == "[":
                end = text.find("]", j)
                if end < 0:
                    raise ExprError(f"a unit opened at column {j + 1} is not closed")
                unit = text[j + 1:end].strip()
                if not units.known(unit) and unit not in ("1", "%"):
                    raise ExprError(units.unknown(unit))
                emit(f"_q({value}, {json.dumps(unit)})")
                i = end + 1
                continue
            k = j
            while k < n and text[k] == " ":
                k += 1
            unit, after = _unit_at(text, k, units)
            if unit is not None:
                emit(f"_q({value}, {json.dumps(unit)})")
                i = after
                continue
            emit(value)
            i = j
            continue
        ident = _IDENT_RE.match(text, i)
        if ident is not None:
            word = ident.group(0)
            emit(_WORDS.get(word.lower(), word) if word.lower() in _WORDS else word)
            i = ident.end()
            continue
        two = text[i:i + 2]
        if two in ("<=", ">=", "==", "!="):
            emit(two)
            i += 2
            continue
        if two == "<>":
            emit("!=")
            i += 2
            continue
        if two == "&&":
            emit(" and ")
            i += 2
            continue
        if two == "||":
            emit(" or ")
            i += 2
            continue
        if two == "**":
            emit("**")
            i += 2
            continue
        replacements = {"≤": "<=", "≥": ">=", "≠": "!=", "−": "-", "×": "*", "÷": "/", "·": "*", "^": "**"}
        if ch in replacements:
            emit(replacements[ch])
            i += 1
            continue
        if ch == "=":
            emit("==")
            i += 1
            continue
        if ch == "!":
            emit(" not ")
            i += 1
            continue
        if ch in "+-*/<>(),.[]":
            emit(ch)
            i += 1
            continue
        raise ExprError(f"unexpected character {ch!r} at column {i + 1}")
    return "".join(out)


def _unit_at(text: str, k: int, units: UnitTable) -> tuple[str | None, int]:
    """The longest known unit starting at `k` that is not followed by a letter or a call, and where it ends."""
    if k < len(text) and text[k] == "%":
        return "%", k + 1
    run = _UNIT_RUN_RE.match(text, k)
    if run is None:
        return None, k
    token = run.group(0)
    for size in range(len(token), 0, -1):
        candidate = token[:size]
        if not units.known(candidate):
            continue
        rest = token[size:]
        if rest and (rest[0].isalnum() or rest[0] in "°µ_"):
            continue
        end = k + size
        probe = end
        while probe < len(text) and text[probe] == " ":
            probe += 1
        if probe < len(text) and text[probe] == "(":
            return None, k  # `5 in (...)` is a membership test, not five inches
        if probe < len(text) and (text[probe].isalpha() or text[probe] == "_") and probe == end:
            continue
        return candidate, end
    return None, k


@dataclass(frozen=True)
class N:
    """A parsed (not yet bound) expression node."""

    op: str
    args: tuple = ()
    value: object = None

    def key(self) -> str:
        inner = ",".join(a.key() if isinstance(a, N) else repr(a) for a in self.args)
        return f"{self.op}({inner}|{self.value!r})"


_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
_CMPOPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}


def parse(text: str, units: UnitTable | None = None) -> N:
    """Read an expression into nodes; raises ExprError for anything outside the language."""
    source = lex(text, units)
    try:
        tree = ast.parse(source.strip() or "_empty_", mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"cannot read {text!r}: {exc.msg}") from None
    except (RecursionError, MemoryError):
        raise ExprError(f"{text!r} is nested too deeply") from None
    if source.strip() == "":
        raise ExprError("the expression is empty")
    counter = [0]
    return _convert(tree.body, text, counter, 0)


def _convert(node: ast.AST, text: str, counter: list[int], depth: int) -> N:
    counter[0] += 1
    if counter[0] > MAX_NODES:
        raise ExprError(f"the expression has more than {MAX_NODES} parts")
    if depth > MAX_DEPTH:
        raise ExprError(f"the expression is nested more than {MAX_DEPTH} levels deep")

    def sub(child: ast.AST) -> N:
        return _convert(child, text, counter, depth + 1)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return N("bool", value=value)
        if isinstance(value, (int, float)):
            return N("num", value=float(value))
        if isinstance(value, str):
            return N("str", value=value)
        raise ExprError(f"{value!r} is not allowed in an expression")
    if isinstance(node, ast.Name):
        if node.id == "_empty_":
            raise ExprError("the expression is empty")
        if node.id in ("True", "False"):
            return N("bool", value=node.id == "True")
        return N("name", value=node.id)
    if isinstance(node, ast.UnaryOp):
        op = {ast.USub: "neg", ast.UAdd: "pos", ast.Not: "not"}.get(type(node.op))
        if op is None:
            raise ExprError(f"the operator in {ast.unparse(node)!r} is not allowed")
        return N(op, (sub(node.operand),))
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExprError(f"the operator in {ast.unparse(node)!r} is not allowed")
        return N(op, (sub(node.left), sub(node.right)))
    if isinstance(node, ast.BoolOp):
        return N("and" if isinstance(node.op, ast.And) else "or", tuple(sub(v) for v in node.values))
    if isinstance(node, ast.Compare):
        operands = [sub(node.left)]
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(comparator, (ast.Tuple, ast.List)):
                    raise ExprError("`in` needs a list of values in parentheses, such as mode in ('A', 'B')")
                items = tuple(sub(e) for e in comparator.elts)
                if any(item.op not in ("num", "str", "qty", "bool", "name") for item in items):
                    raise ExprError("a list may only hold values, such as ('A', 'B') or (1, 2)")
                operands.append(N("list", items))
            else:
                operands.append(sub(comparator))
        parts = []
        for left, op, right in zip(operands, node.ops, operands[1:]):
            if isinstance(op, (ast.In, ast.NotIn)):
                parts.append(N("in" if isinstance(op, ast.In) else "notin", (left, right)))
                continue
            symbol = _CMPOPS.get(type(op))
            if symbol is None:
                raise ExprError(f"the comparison in {ast.unparse(node)!r} is not allowed")
            parts.append(N("cmp", (left, right), symbol))
        return parts[0] if len(parts) == 1 else N("and", tuple(parts))
    if isinstance(node, (ast.Tuple, ast.List)):
        raise ExprError("a list of values only belongs after `in`, as in mode in ('A', 'B')")
    if isinstance(node, ast.IfExp):
        return N("if", (sub(node.test), sub(node.body), sub(node.orelse)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ExprError(f"only plain function calls are allowed, not {ast.unparse(node.func)!r}")
        name = node.func.id
        if name == "_q":
            value, unit = node.args
            return N("qty", value=(float(value.value), unit.value))
        if name == "_sig":
            return N("path", value=node.args[0].value)
        lowered = name.lower()
        if lowered not in FUNCTIONS and lowered != "table":
            close = difflib.get_close_matches(lowered, list(FUNCTIONS), n=1)
            hint = f"; did you mean {close[0]}()?" if close else ""
            raise ExprError(f"unknown function {name}(){hint}")
        lowered = "curve" if lowered == "table" else lowered
        low, high = FUNCTIONS[lowered]
        if not low <= len(node.args) <= high:
            count = str(low) if low == high else f"{low} to {high}"
            raise ExprError(f"{lowered}() takes {count} arguments, got {len(node.args)}")
        return N("call", tuple(sub(a) for a in node.args), lowered)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "param":
            return N("param", value=node.attr)
        if node.attr in COMPONENTS:
            return N("comp", (sub(node.value),), COMPONENTS[node.attr])
        raise ExprError(f"`.{node.attr}` is not allowed; use .x, .y, .z or param.<name>")
    if isinstance(node, ast.Subscript):
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int) and not isinstance(index.value, bool):
            return N("comp", (sub(node.value),), index.value)
        raise ExprError("only a whole-number component index is allowed, such as `nav/position`[2]")
    raise ExprError(f"{type(node).__name__} is not allowed in an expression")


# ------------------------------------------------------------------------------------------------------------
# Types and binding


@dataclass(frozen=True)
class TypeInfo:
    shape: str  # "series" or "scalar"
    dtype: str  # "num", "bool", "enum", "str" or "any" (a run parameter, known when the run is read)
    unit: str | None = None
    discrete: bool = False
    components: int = 1
    labels: tuple[str, ...] | None = None  # enum labels in code order, when known before loading

    @property
    def series(self) -> bool:
        return self.shape == "series"


@dataclass
class B:
    """A bound node: an operation with typed arguments. `value` holds constants, names and settings."""

    op: str
    args: tuple
    value: object
    type: TypeInfo
    key: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            inner = ",".join(a.key if isinstance(a, B) else repr(a) for a in self.args)
            self.key = f"{self.op}[{inner}|{self.value!r}|{self.type.unit}]"


@dataclass
class BoundExpr:
    root: B
    text: str
    signals: list[str] = field(default_factory=list)  # canonical names, in order of first use
    events: set[str] = field(default_factory=set)
    params: set[str] = field(default_factory=set)
    conditions: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    @property
    def type(self) -> TypeInfo:
        return self.root.type


NUM_SERIES = TypeInfo("series", "num")
BOOL_SERIES = TypeInfo("series", "bool", discrete=True)
BOOL_SCALAR = TypeInfo("scalar", "bool")


def _num(shape: str, unit: str | None, *, discrete: bool = False, components: int = 1) -> TypeInfo:
    return TypeInfo(shape, "num", unit, discrete, components)


class Binder:
    """Resolves the names of expressions against one source's signals and a mapping."""

    def __init__(self, index, infos: Mapping, config, *, param_names: set[str] | None = None,
                 param_units: Mapping[str, str] | None = None, labels: Mapping[str, Sequence[str]] | None = None
                 ) -> None:
        self.index = index
        self.infos = infos
        self.labels = dict(labels or {})
        self.config = config
        self.units: UnitTable = config.units
        self.param_names = param_names
        self.param_units = dict(param_units or {})
        self._stack: list[str] = []
        self._aliases: dict[str, BoundExpr] = {}
        self._conditions: dict[str, BoundExpr] = {}

    # ----- entry point ------------------------------------------------------------------------------------

    def bind(self, text: str, *, role: str = "check") -> BoundExpr:
        """Bind `text`. `role` is check, when, limit, applies_to or event (applies_to only sees parameters)."""
        try:
            tree = parse(text, self.units)
        except ExprError:
            raise
        except UnitsError as exc:
            raise ExprError(str(exc)) from None
        out = BoundExpr(root=None, text=text)  # type: ignore[arg-type]
        out.root = self._bind(tree, out, role)
        return out

    # ----- helpers ----------------------------------------------------------------------------------------

    def _error(self, message: str) -> ExprError:
        return ExprError(message)

    def _signal_type(self, name: str, alias=None) -> TypeInfo:
        info = self.infos[name]
        kind = (alias.kind if alias is not None and alias.kind else None) or info.kind
        unit = alias.unit if alias is not None and alias.unit else info.unit
        components = int(info.shape[1]) if len(info.shape) == 2 else 1
        labels = None
        dtype = "num"
        if alias is not None and alias.labels:
            top = max(alias.labels)
            labels = tuple(alias.labels.get(code, f"<{code}>") for code in range(top + 1))
            dtype = "enum"
        elif self.labels.get(name):
            labels = tuple(self.labels[name])
            dtype = "enum"
        elif str(info.dtype).lstrip("<>|=")[:1] in ("U", "S", "O") or info.dtype == "str":
            dtype = "enum"
        if kind == "discrete" and dtype == "num" and str(info.dtype).lstrip("<>|=")[:1] in ("i", "u", "b"):
            dtype = "num"
        return TypeInfo("series", dtype, unit, kind == "discrete" or dtype == "enum", components, labels)

    def _signal(self, name: str, out: BoundExpr, alias=None) -> B:
        if name not in out.signals:
            out.signals.append(name)
        return B("sig", (), (name, alias.name if alias is not None else None), self._signal_type(name, alias))

    def _resolve(self, name: str, out: BoundExpr, role: str) -> B:
        config = self.config
        if role == "applies_to":
            return self._param(name, out)
        if name == "t":
            return B("tvar", (), None, _num("series", "s"))
        alias = config.signals.get(name)
        if alias is not None:
            return self._alias(alias, out, role)
        if name in config.conditions:
            return self._condition(name, out, role)
        try:
            canonical = self.index.lookup(name)
        except LookupError as exc:
            message = str(exc)
            if message.startswith("unknown signal"):
                pool = [*config.signals, *config.conditions, *self.index.candidates()]
                close = difflib.get_close_matches(name, pool, n=1, cutoff=0.6)
                hint = f"; did you mean {close[0]!r}?" if close else ""
                raise self._error(f"unknown name {name!r}{hint}") from None
            raise self._error(message) from None
        return self._signal(canonical, out)

    def _param(self, name: str, out: BoundExpr) -> B:
        if self.param_names is not None and name not in self.param_names:
            close = difflib.get_close_matches(name, sorted(self.param_names), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise self._error(f"unknown run parameter {name!r}{hint}")
        out.params.add(name)
        return B("param", (), name, TypeInfo("scalar", "any", self.param_units.get(name)))

    def _alias(self, alias, out: BoundExpr, role: str) -> B:
        if alias.path is not None:
            try:
                canonical = self.index.lookup(alias.path)
            except LookupError as exc:
                raise self._error(f"alias {alias.name!r}: {exc}") from None
            return self._signal(canonical, out, alias)
        if alias.name in self._stack:
            chain = " -> ".join([*self._stack[self._stack.index(alias.name):], alias.name])
            raise self._error(f"the definitions refer to each other in a loop: {chain}")
        self._stack.append(alias.name)
        try:
            inner = self.bind(alias.expr, role=role)
        except ExprError as exc:
            raise self._error(f"alias {alias.name!r}: {exc}") from None
        finally:
            self._stack.pop()
        self._merge(out, inner)
        node = inner.root
        if alias.unit:
            node = self._declare(node, alias.unit, f"alias {alias.name!r}")
        return B("alias", (node,), alias.name, node.type)

    def _condition(self, name: str, out: BoundExpr, role: str) -> B:
        if name in self._stack:
            chain = " -> ".join([*self._stack[self._stack.index(name):], name])
            raise self._error(f"the definitions refer to each other in a loop: {chain}")
        self._stack.append(name)
        try:
            inner = self.bind(self.config.conditions[name], role=role)
        except ExprError as exc:
            raise self._error(f"condition {name!r}: {exc}") from None
        finally:
            self._stack.pop()
        if inner.type.dtype != "bool":
            raise self._error(f"condition {name!r} is not a condition (it has no comparison)")
        self._merge(out, inner)
        out.conditions.add(name)
        return B("cond", (inner.root,), name, inner.type)

    @staticmethod
    def _merge(out: BoundExpr, inner: BoundExpr) -> None:
        for name in inner.signals:
            if name not in out.signals:
                out.signals.append(name)
        out.events |= inner.events
        out.params |= inner.params
        out.conditions |= inner.conditions
        out.notes.extend(inner.notes)

    def _declare(self, node: B, unit: str, where: str) -> B:
        current = node.type.unit
        if current in (None, UNKNOWN, unit):
            return B("unit", (node,), unit, TypeInfo(node.type.shape, node.type.dtype, unit, node.type.discrete,
                                                      node.type.components, node.type.labels))
        source, target = self.units.lookup(current), self.units.lookup(unit)
        if source is None or target is None or source.dimension != target.dimension:
            raise self._error(f"{where}: the values are in {current}, which cannot be declared as {unit}")
        return self._convert(node, unit)

    def _convert(self, node: B, unit: str) -> B:
        """`node` expressed in `unit` (same dimension, or a bare number that simply takes the unit)."""
        current = node.type.unit
        if current == unit or current is None:
            if node.op in ("num",) and current is None:
                return B("num", (), node.value, TypeInfo(node.type.shape, "num", unit))
            return node
        if self.units.is_affine(current) or self.units.is_affine(unit):
            if node.op == "qty":
                value, _ = node.value
                converted = self.units.convert(Quantity(value, current, f"{value} {current}"), unit, delta=False)
                return B("num", (), converted, TypeInfo(node.type.shape, "num", unit))
            raise self._error(f"values in {current} cannot be converted to {unit} inside an expression")
        factor = self.units.factor(current, unit)
        if node.op == "qty":
            value, _ = node.value
            return B("num", (), value * factor, TypeInfo(node.type.shape, "num", unit))
        return B("scale", (node,), factor, TypeInfo(node.type.shape, node.type.dtype, unit, node.type.discrete,
                                                     node.type.components, node.type.labels))

    def _reconcile(self, left: B, right: B, what: str) -> tuple[B, B, str | None]:
        """Two numeric operands in one unit: bare numbers take the other side's unit, known units convert."""
        a, b = left.type.unit, right.type.unit
        if a == b:
            return left, right, a
        literal = ("num",)
        if b is None and right.op in literal:
            return left, self._convert(right, a) if a not in (None, UNKNOWN) else right, a
        if a is None and left.op in literal:
            return self._convert(left, b) if b not in (None, UNKNOWN) else left, right, b
        if UNKNOWN in (a, b):
            other = b if a == UNKNOWN else a
            if other is None:
                return left, right, UNKNOWN
            raise self._error(f"{what}: one side has a unit the language cannot work out; declare it with "
                              f"as_unit(..., \"{other}\") or an alias unit")
        if a is None or b is None:
            known = a if a is not None else b
            known_side = left if a is not None else right
            if known_side.op == "qty":
                raise self._error(f"{what}: {known_side.value[0]:g} {known} has a unit but the other side has "
                                  "none; write a bare number or declare the signal's unit in its alias")
            # two signals, one without a unit: the unitless one is taken to be in the other's unit
            return left, right, known
        ua, ub = self.units.lookup(a), self.units.lookup(b)
        if ua is None or ub is None:
            raise self._error(f"{what}: {a!r} and {b!r} are different units and at least one is unknown")
        if ua.dimension != ub.dimension:
            raise self._error(f"{what}: {self.units.phrase(ua.dimension)} ({a}) and "
                              f"{self.units.phrase(ub.dimension)} ({b}) cannot be compared or added")
        if right.op == "qty" or right.op == "num":
            return left, self._convert(right, a), a
        if left.op == "qty":
            return self._convert(left, b), right, b
        return left, self._convert(right, a), a

    @staticmethod
    def _shape(*nodes: B) -> str:
        return "series" if any(n.type.series for n in nodes) else "scalar"

    def _need(self, node: B, dtypes: Sequence[str], what: str) -> None:
        if node.type.dtype not in dtypes and node.type.dtype != "any":
            names = {"num": "a number", "bool": "a condition", "enum": "a state", "str": "a text"}
            wanted = " or ".join(names[d] for d in dtypes)
            raise self._error(f"{what} needs {wanted}, but got {names.get(node.type.dtype, node.type.dtype)}")

    def _scalar_series(self, node: B, what: str) -> None:
        if node.type.components != 1:
            raise self._error(f"{what} needs a single-valued signal; use norm() or .x/.y/.z")

    # ----- the walk ---------------------------------------------------------------------------------------

    def _bind(self, node: N, out: BoundExpr, role: str) -> B:
        op = node.op
        if op == "num":
            return B("num", (), node.value, _num("scalar", None))
        if op == "bool":
            return B("bool", (), node.value, BOOL_SCALAR)
        if op == "str":
            return B("str", (), node.value, TypeInfo("scalar", "str"))
        if op == "qty":
            value, unit = node.value
            if unit == "%":
                return B("num", (), value / 100.0, _num("scalar", None))
            return B("qty", (), (value, unit), _num("scalar", unit))
        if op == "name":
            return self._resolve(node.value, out, role)
        if op == "path":
            if role == "applies_to":
                raise self._error("applies_to looks at run parameters, not signals")
            try:
                canonical = self.index.lookup(node.value)
            except LookupError as exc:
                raise self._error(str(exc)) from None
            return self._signal(canonical, out)
        if op == "param":
            return self._param(node.value, out)
        if op in ("neg", "pos"):
            inner = self._bind(node.args[0], out, role)
            self._need(inner, ("num",), "a sign")
            return inner if op == "pos" else B("neg", (inner,), None, inner.type)
        if op == "not":
            inner = self._bind(node.args[0], out, role)
            self._need(inner, ("bool",), "not")
            return B("not", (inner,), None, inner.type)
        if op in ("and", "or"):
            parts = tuple(self._bind(a, out, role) for a in node.args)
            for part in parts:
                self._need(part, ("bool",), op)
            shape = self._shape(*parts)
            return B(op, parts, None, BOOL_SERIES if shape == "series" else BOOL_SCALAR)
        if op in ("add", "sub", "mul", "div", "pow"):
            return self._arith(op, node, out, role)
        if op == "cmp":
            return self._compare(node, out, role)
        if op in ("in", "notin"):
            return self._membership(node, out, role)
        if op == "if":
            test = self._bind(node.args[0], out, role)
            self._need(test, ("bool",), "a conditional expression")
            return self._where(test, self._bind(node.args[1], out, role), self._bind(node.args[2], out, role))
        if op == "comp":
            inner = self._bind(node.args[0], out, role)
            index = node.value
            if inner.type.components < 2:
                raise self._error("a component (.x, .y, .z or [i]) needs a vector signal")
            if not 0 <= index < inner.type.components:
                raise self._error(f"component {index} does not exist; the vector has {inner.type.components}")
            t = inner.type
            return B("comp", (inner,), index, TypeInfo(t.shape, t.dtype, t.unit, t.discrete, 1, t.labels))
        if op == "list":
            raise self._error("a list of values only belongs after `in`")
        if op == "call":
            return self._call(node, out, role)
        raise self._error(f"cannot use {op!r} here")

    def _arith(self, op: str, node: N, out: BoundExpr, role: str) -> B:
        left = self._bind(node.args[0], out, role)
        right = self._bind(node.args[1], out, role)
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}[op]
        for side in (left, right):
            self._need(side, ("num",), f"'{symbol}'")
        components = max(left.type.components, right.type.components)
        if left.type.components > 1 and right.type.components > 1 and left.type.components != right.type.components:
            raise self._error(f"'{symbol}' between vectors of different sizes")
        shape = self._shape(left, right)
        discrete = left.type.discrete and right.type.discrete
        a, b = left.type.unit, right.type.unit
        if op in ("add", "sub"):
            for u in (a, b):
                if self.units.is_affine(u) and a != b:
                    raise self._error(f"'{symbol}' with {u} values needs both sides in {u}")
            left, right, unit = self._reconcile(left, right, f"'{symbol}'")
            return B(op, (left, right), None, _num(shape, unit, discrete=discrete, components=components))
        if op == "pow":
            unit = a if (right.op == "num" and right.value == 1.0) else (None if a is None else UNKNOWN)
            return B(op, (left, right), None, _num(shape, unit, discrete=discrete, components=components))
        # multiply and divide
        left, right = self._qty_to_number(left), self._qty_to_number(right)
        a, b = left.type.unit, right.type.unit
        if b is None:
            unit = a
        elif a is None and op == "mul":
            unit = b
        elif a is None:
            unit = None if b is None else UNKNOWN
        elif op == "div" and UNKNOWN not in (a, b):
            ua, ub = self.units.lookup(a), self.units.lookup(b)
            if ua is not None and ub is not None and ua.dimension == ub.dimension:
                right = self._convert(right, a) if right.op != "num" else right
                unit = "1"
            else:
                unit = UNKNOWN
        else:
            unit = UNKNOWN
        return B(op, (left, right), None, _num(shape, unit, discrete=discrete, components=components))

    def _qty_to_number(self, node: B) -> B:
        """A dimensionless quantity (`1`) multiplies like a number."""
        if node.op == "qty" and self.units.dimension(node.value[1]) == "dimensionless":
            value, unit = node.value
            factor = self.units.factor(unit, "1") if self.units.known("1") else 1.0
            return B("num", (), value * factor, _num("scalar", None))
        return node

    def _labels_value(self, enum: B, value: B, where: str, out: BoundExpr) -> B:
        """A text (or bare word) compared with a state: the label's code."""
        if value.op == "str":
            return B("label", (), value.value, TypeInfo("scalar", "str"))
        raise self._error(f"{where}: compare a state with its label in quotes")

    def _compare(self, node: N, out: BoundExpr, role: str) -> B:
        symbol = node.value
        raw_left, raw_right = node.args
        left = self._bind_side(raw_left, raw_right, out, role)
        right = self._bind_side(raw_right, raw_left, out, role)
        where = f"'{symbol}'"
        shape = self._shape(left, right)
        kinds = {left.type.dtype, right.type.dtype}
        if left.type.components > 1 or right.type.components > 1:
            raise self._error(f"{where} compares single values; use norm() or .x/.y/.z on vectors")
        result = BOOL_SERIES if shape == "series" else BOOL_SCALAR
        if "str" in kinds and "any" not in kinds:
            if kinds == {"str"}:
                return B("cmp", (left, right), symbol, result)
            if "enum" not in kinds:
                raise self._error(f"{where}: a text can only be compared with a state (a signal with labels) or "
                                  "a run parameter; give the signal labels in its alias")
            enum, other = (left, right) if left.type.dtype == "enum" else (right, left)
            if symbol not in ("==", "!="):
                raise self._error(f"{where}: states can only be compared with == or !=")
            self._check_label(enum, other.value)
            label = B("label", (), other.value, TypeInfo("scalar", "str"))
            return B("cmp", (enum, label) if enum is left else (label, enum), symbol, result)
        if "enum" in kinds:
            if not kinds <= {"enum", "num"}:
                raise self._error(f"{where}: cannot compare a state with {' or '.join(sorted(kinds - {'enum'}))}")
            return B("cmp", (left, right), symbol, result)
        if kinds == {"bool"}:
            if symbol not in ("==", "!="):
                raise self._error(f"{where}: conditions can only be compared with == or !=")
            return B("cmp", (left, right), symbol, result)
        if "any" in kinds:
            other = right if left.type.dtype == "any" else left
            param = left if other is right else right
            if other.op == "qty" and param.type.unit is not None:
                other = self._convert(other, param.type.unit)
                left, right = (param, other) if param is left else (other, param)
            return B("cmp", (left, right), symbol, result)
        for side in (left, right):
            self._need(side, ("num",), where)
        left, right, _ = self._reconcile(left, right, where)
        return B("cmp", (left, right), symbol, result)

    def _bind_side(self, node: N, other: N, out: BoundExpr, role: str) -> B:
        """A bare word next to a state is its label, when it names nothing else."""
        if node.op == "name" and role != "applies_to":
            name = node.value
            known = (name in self.config.signals or name in self.config.conditions or name == "t")
            if not known:
                try:
                    self.index.lookup(name)
                    known = True
                except LookupError:
                    known = False
            if not known:
                try:
                    partner = self._bind(other, BoundExpr(root=None, text=""), role)  # type: ignore[arg-type]
                except ExprError:
                    partner = None
                if partner is not None and partner.type.dtype == "enum":
                    out.notes.append(f"{name} was read as the label '{name}'; write it in quotes")
                    return B("str", (), name, TypeInfo("scalar", "str"))
        return self._bind(node, out, role)

    def _check_label(self, enum: B, label: str) -> None:
        labels = enum.type.labels
        if labels is not None and label not in labels:
            close = difflib.get_close_matches(label, list(labels), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else f"; labels are {', '.join(labels)}"
            raise self._error(f"{label!r} is not a label of this state{hint}")

    def _membership(self, node: N, out: BoundExpr, role: str) -> B:
        subject = self._bind(node.args[0], out, role)
        values_node = node.args[1]
        values: list[B] = []
        for item in values_node.args:
            if item.op == "name" and subject.type.dtype == "enum":
                values.append(B("str", (), item.value, TypeInfo("scalar", "str")))
                out.notes.append(f"{item.value} was read as the label '{item.value}'; write it in quotes")
            else:
                values.append(self._bind(item, out, role))
        if subject.type.components > 1:
            raise self._error("`in` compares single values")
        converted = []
        for value in values:
            if value.op == "str":
                if subject.type.dtype not in ("enum", "any", "str"):
                    raise self._error(f"{value.value!r} is a text, but the subject is not a state")
                if subject.type.dtype == "enum":
                    self._check_label(subject, value.value)
                converted.append(B("label", (), value.value, TypeInfo("scalar", "str")))
            elif value.op in ("num", "qty", "bool"):
                if value.op == "qty":
                    _, value, _ = self._reconcile(subject, value, "`in`")
                converted.append(value)
            else:
                raise self._error("`in` needs constant values")
        shape = subject.type.shape
        return B(node.op, (subject, *converted), None, BOOL_SERIES if shape == "series" else BOOL_SCALAR)

    def _where(self, test: B, a: B, b: B) -> B:
        if a.type.dtype == "bool" and b.type.dtype == "bool":
            shape = self._shape(test, a, b)
            return B("where", (test, a, b), None, BOOL_SERIES if shape == "series" else BOOL_SCALAR)
        for side in (a, b):
            self._need(side, ("num",), "where()")
        a, b, unit = self._reconcile(a, b, "where()")
        shape = self._shape(test, a, b)
        return B("where", (test, a, b), None, _num(shape, unit, components=max(a.type.components,
                                                                               b.type.components)))

    def _event_name(self, node: N, out: BoundExpr, fn: str) -> str:
        if node.op == "str":
            text = node.value
        elif node.op == "name":
            text = node.value
        else:
            raise self._error(f"{fn}() needs an event name, such as {fn}('MECO')")
        name, _, occurrence = text.partition("#")
        name = name.strip()
        if name not in self.config.events:
            close = difflib.get_close_matches(name, list(self.config.events), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else "; define it under events:"
            raise self._error(f"unknown event {name!r}{hint}")
        occurrence = occurrence.strip().lower()
        if occurrence and occurrence not in ("first", "last", "all") and not occurrence.isdigit():
            raise self._error(f"{text!r}: after '#' write first, last, all or a trigger number")
        out.events.add(name)
        return text.strip()

    def _duration(self, node: N, out: BoundExpr, role: str, fn: str) -> float:
        bound = self._bind(node, out, role)
        if bound.type.shape != "scalar" or bound.op not in ("num", "qty", "neg"):
            raise self._error(f"{fn}() takes a fixed duration, such as 2 s")
        if bound.op == "neg":
            inner = bound.args[0]
            return -self._duration_value(inner, fn)
        return self._duration_value(bound, fn)

    def _duration_value(self, bound: B, fn: str) -> float:
        if bound.op == "num":
            return float(bound.value)
        value, unit = bound.value
        if self.units.dimension(unit) != "time":
            raise self._error(f"{fn}() takes a duration, not {value:g} {unit}")
        return value * self.units.factor(unit, "s")

    def _call(self, node: N, out: BoundExpr, role: str) -> B:
        fn = node.value
        raw = node.args
        if role == "applies_to" and fn not in ("abs", "min", "max", "clip", "sqrt", "exp", "log"):
            raise self._error(f"{fn}() cannot be used in applies_to, which looks at run parameters")

        if fn in EVENT_FUNCTIONS:
            if fn in ("after", "before", "during"):
                event = self._event_name(raw[0], out, fn)
                delay = self._duration(raw[1], out, role, fn) if len(raw) > 1 else 0.0
                return B(fn, (), (event, delay), BOOL_SERIES)
            if fn == "between":
                first = self._event_name(raw[0], out, fn)
                second = self._event_name(raw[1], out, fn)
                return B(fn, (), (first, second), BOOL_SERIES)
            if fn == "since":
                return B(fn, (), self._event_name(raw[0], out, fn), _num("series", "s"))
            if fn == "time":
                return B(fn, (), self._event_name(raw[0], out, fn), _num("scalar", "s"))
            if fn == "count":
                return B(fn, (), self._event_name(raw[0], out, fn), _num("scalar", None))
            if fn == "at":
                event = self._event_name(raw[0], out, fn)
                inner = self._bind(raw[1], out, role)
                self._need(inner, ("num", "enum"), "at()")
                self._scalar_series(inner, "at()")
                if not inner.type.series:
                    raise self._error("at() takes a signal or a series expression as its second argument")
                t = inner.type
                return B(fn, (inner,), event, TypeInfo("scalar", t.dtype, t.unit, t.discrete, 1, t.labels))

        if fn == "curve":
            return self._curve(raw, out, role)

        if fn in ("min", "max") and len(raw) == 1 or fn in AGGREGATES:
            inner = self._bind(raw[0], out, role)
            if fn == "duration":
                self._need(inner, ("bool",), "duration()")
                if not inner.type.series:
                    raise self._error("duration() needs a condition on signals")
                return B("agg", (inner,), fn, _num("scalar", "s"))
            self._need(inner, ("num",), f"{fn}()")
            self._scalar_series(inner, f"{fn}()")
            if not inner.type.series:
                raise self._error(f"{fn}() of a single value is that value; it needs a signal")
            unit = inner.type.unit
            if fn == "integral" and unit not in (None, UNKNOWN):
                unit = self.units.integral_unit(unit) or UNKNOWN
            return B("agg", (inner,), fn, TypeInfo("scalar", "num", unit, inner.type.discrete))

        args = tuple(self._bind(a, out, role) for a in raw) if fn not in ("as_unit", "movmean") else ()
        if fn in ("min", "max"):
            for arg in args:
                self._need(arg, ("num",), f"{fn}()")
            nodes = list(args)
            unit = nodes[0].type.unit
            for k in range(1, len(nodes)):
                nodes[0], nodes[k], unit = self._reconcile(nodes[0], nodes[k], f"{fn}()")
            shape = self._shape(*nodes)
            components = max(n.type.components for n in nodes)
            return B(fn, tuple(nodes), None, _num(shape, unit, components=components))
        if fn in ("abs", "sign"):
            (x,) = args
            self._need(x, ("num",), f"{fn}()")
            unit = x.type.unit if fn == "abs" else None
            return B(fn, (x,), None, _num(x.type.shape, unit, discrete=x.type.discrete,
                                          components=x.type.components))
        if fn == "clip":
            x, lo, hi = args
            for arg in args:
                self._need(arg, ("num",), "clip()")
            x, lo, unit = self._reconcile(x, lo, "clip()")
            x, hi, unit = self._reconcile(x, hi, "clip()")
            return B(fn, (x, lo, hi), None, _num(self._shape(x, lo, hi), unit, components=x.type.components))
        if fn == "where":
            test = args[0]
            self._need(test, ("bool",), "where()")
            return self._where(test, args[1], args[2])
        if fn == "hypot":
            a, b = args
            for arg in args:
                self._need(arg, ("num",), "hypot()")
            a, b, unit = self._reconcile(a, b, "hypot()")
            return B(fn, (a, b), None, _num(self._shape(a, b), unit))
        if fn == "norm":
            (x,) = args
            self._need(x, ("num",), "norm()")
            return B(fn, (x,), None, _num(x.type.shape, x.type.unit))
        if fn in ("sqrt", "exp", "log"):
            (x,) = args
            self._need(x, ("num",), f"{fn}()")
            unit = x.type.unit
            if unit not in (None, UNKNOWN, "1") and self.units.dimension(unit) != "dimensionless":
                raise self._error(f"{fn}() needs a value without a unit, but it is in {unit}")
            return B(fn, (x,), None, _num(x.type.shape, None if unit != UNKNOWN else UNKNOWN,
                                          components=x.type.components))
        if fn in ("sin", "cos", "tan"):
            (x,) = args
            self._need(x, ("num",), f"{fn}()")
            unit = x.type.unit
            if unit not in (None, UNKNOWN) and self.units.dimension(unit) != "angle":
                raise self._error(f"{fn}() needs an angle, but the values are in {unit}")
            if unit not in (None, UNKNOWN, "rad"):
                x = self._convert(x, "rad")
            return B(fn, (x,), None, _num(x.type.shape, None, components=x.type.components))
        if fn == "deriv":
            (x,) = args
            self._need(x, ("num",), "deriv()")
            if not x.type.series:
                raise self._error("deriv() needs a signal")
            unit = x.type.unit
            if unit not in (None, UNKNOWN):
                unit = self.units.derivative_unit(unit) or UNKNOWN
            elif unit is None:
                unit = None
            return B(fn, (x,), None, _num("series", unit, components=x.type.components))
        if fn == "movmean":
            x = self._bind(raw[0], out, role)
            self._need(x, ("num",), "movmean()")
            if not x.type.series:
                raise self._error("movmean() needs a signal")
            window = self._duration(raw[1], out, role, "movmean")
            if window <= 0:
                raise self._error("movmean() needs a positive window, such as 200 ms")
            t = x.type
            return B(fn, (x,), window, _num("series", t.unit, components=t.components))
        if fn == "as_unit":
            x = self._bind(raw[0], out, role)
            unit_node = raw[1]
            if unit_node.op not in ("str", "name"):
                raise self._error('as_unit() takes the unit as text, such as as_unit(x, "m/s^2")')
            unit = str(unit_node.value).strip()
            if not self.units.known(unit):
                raise self._error(self.units.unknown(unit))
            return self._declare(x, unit, "as_unit()")
        raise self._error(f"{fn}() cannot be used here")

    def _curve(self, raw: tuple, out: BoundExpr, role: str) -> B:
        first = raw[0]
        if first.op not in ("str", "name"):
            raise self._error("curve() takes the curve's name first, such as curve(q_max, mach)")
        name = str(first.value)
        curve = self.config.curves.get(name)
        if curve is None:
            close = difflib.get_close_matches(name, list(self.config.curves), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else "; define it under curves:"
            raise self._error(f"unknown curve {name!r}{hint}")
        if len(raw) > 1:
            arg = self._bind(raw[1], out, role)
        elif curve.x:
            arg = self.bind(curve.x, role=role)
            self._merge(out, arg)
            arg = arg.root
        else:
            raise self._error(f"curve {name!r} has no default argument; write curve({name}, <signal>)")
        self._need(arg, ("num",), "curve()")
        self._scalar_series(arg, "curve()")
        if curve.x_unit and arg.type.unit not in (None, UNKNOWN):
            try:
                arg = self._convert(arg, curve.x_unit)
            except (ExprError, UnitsError) as exc:
                raise self._error(f"curve {name!r}: {exc}") from None
        elif curve.x_unit and arg.type.unit == UNKNOWN and self.units.dimension(curve.x_unit) != "dimensionless":
            raise self._error(f"curve {name!r}: the argument's unit is unknown; declare it with as_unit()")
        return B("curve", (arg,), name, _num(arg.type.shape, curve.y_unit))
