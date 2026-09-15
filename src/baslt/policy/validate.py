"""Structural and semantic validation of a parsed policy.

Every section and key of docs/policy.md is checked. Problems are collected (up to 50) with
their paths and, when known, file locations, and raised together as one PolicyError. The result
is a `Policy` with all defaults filled in, its canonical form and its hash.
"""

from __future__ import annotations

import difflib
import math
import numbers
from collections.abc import Mapping, Sequence

from ..errors import Issue, PolicyError
from ..units import (
    UNITS,
    Quantity,
    describe_dimension,
    lookup_unit,
    parse_bytes,
    parse_quantity,
    to_seconds,
)
from .canonical import canonical_dict, canonical_sha256
from .schema import (
    DEFAULT_SOFT_PRIORITY,
    MAX_THUMBNAILS,
    OP_SCHEMAS,
    SEVERITIES,
    SOFT_WEIGHTS,
    ArtifactSpec,
    EventSpec,
    HardReq,
    Param,
    Policy,
    ReviewSpec,
    SignalDecl,
    SignalsSpec,
    SoftRule,
    SyncGroupSpec,
    TrajectorySpec,
    default_severity,
)

MAX_ISSUES = 50

TOP_KEYS = ("version", "name", "artifact", "signals", "hard", "events", "trajectories", "sync_groups", "soft", "review")
ARTIFACT_KEYS = ("max_size", "codec", "hash", "soft_value_dtype", "on_not_applicable")
SIGNALS_KEYS = ("time", "time_unit", "on_non_monotonic", "include", "exclude", "decl")
DECL_KEYS = ("path", "unit", "kind", "time")
EVENT_KEYS = ("when", "occurrence", "expect", "keep", "severity")
WHEN_KEYS = ("signal", "falls_below", "rises_above", "equals", "hysteresis", "debounce")
TRIGGERS = ("falls_below", "rises_above", "equals")
KEEP_KEYS = ("before", "after", "signals")
TRAJECTORY_KEYS = ("position", "max_position_error", "max_time_error", "linked")
SYNC_KEYS = ("members",)
SOFT_KEYS = ("match", "priority", "max_points")
REVIEW_KEYS = ("thumbnails", "kpis")

CODECS = ("deflate", "store", "zstd")
HASH_MODES = ("full", "sampled", "none")
SOFT_VALUE_DTYPES = ("source", "float32")
ON_NOT_APPLICABLE = ("warn", "fail")
ON_NON_MONOTONIC = ("error", "sort")
SIGNAL_KINDS = ("continuous", "discrete", "vector")
OCCURRENCES = ("first", "last", "all")

# Sign constraints for hard parameters: deltas and durations are non-negative unless listed.
_SIGNED_PARAMS = {("window_extrema", "origin")}
_POSITIVE_PARAMS = {("window_extrema", "interval")}


class _Invalid:
    def __repr__(self) -> str:
        return "<invalid>"


INVALID = _Invalid()


def join_path(parent: str, key: object) -> str:
    return f"{parent}.{key}" if parent else str(key)


def item_path(parent: str, index: int) -> str:
    return f"{parent}[{index}]"


def find_location(locations: Mapping[str, str], path: str) -> str | None:
    """Location of `path` or of its nearest ancestor that has one."""
    while path:
        found = locations.get(path)
        if found is not None:
            return found
        if path.endswith("]") and "[" in path:
            path = path[: path.rindex("[")]
        elif "." in path:
            path = path[: path.rindex(".")]
        else:
            return None
    return None


def suggest(word: str, choices: Sequence[str]) -> str | None:
    close = difflib.get_close_matches(word, list(choices), n=1, cutoff=0.6)
    return close[0] if close else None


def got(value: object) -> str:
    """Short description of an offending value for messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, numbers.Real):
        return repr(value)
    if isinstance(value, Mapping):
        return "a mapping"
    if isinstance(value, (list, tuple)):
        return "a list"
    return type(value).__name__


class IssueCollector:
    """Collects issues and raises one PolicyError once `limit` issues are reached."""

    def __init__(self, locations: Mapping[str, str] | None = None, limit: int = MAX_ISSUES) -> None:
        self.locations = dict(locations or {})
        self.limit = limit
        self.issues: list[Issue] = []

    def add(self, path: str, message: str) -> None:
        self.issues.append(Issue(path=path, message=message, location=find_location(self.locations, path)))
        if len(self.issues) >= self.limit:
            raise PolicyError(self.issues)

    def absorb(self, error: PolicyError) -> None:
        for issue in error.issues:
            self.add(issue.path, issue.message)

    def raise_if_any(self) -> None:
        if self.issues:
            raise PolicyError(self.issues)


def _is_list(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _is_int(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


class _Validator:
    def __init__(self, locations: Mapping[str, str]) -> None:
        self.issues = IssueCollector(locations)

    # ----- generic helpers -------------------------------------------------------------

    def check_keys(self, mapping: Mapping, allowed: Sequence[str], path: str) -> None:
        for key in mapping:
            if not isinstance(key, str):
                self.issues.add(path, f"keys must be strings, got {got(key)}")
                continue
            if key not in allowed:
                hint = suggest(key, allowed)
                if hint is not None:
                    message = f"unknown key {key!r}; did you mean {hint!r}?"
                else:
                    message = f"unknown key {key!r}; expected one of {', '.join(allowed)}"
                self.issues.add(join_path(path, key), message)

    def section(self, data: Mapping, key: str, path: str) -> Mapping | None:
        """A mapping value; null or absent gives an empty mapping, a wrong type gives None."""
        value = data.get(key)
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            self.issues.add(path, f"expected a mapping, got {got(value)}")
            return None
        return value

    def string_keys(self, mapping: Mapping, path: str) -> list[tuple[str, object]]:
        items = []
        for key, value in mapping.items():
            if not isinstance(key, str) or not key:
                self.issues.add(path, f"names must be non-empty strings, got {got(key)}")
                continue
            items.append((key, value))
        return items

    def string(self, mapping: Mapping, key: str, path: str, *, required: bool = False) -> str | None | _Invalid:
        value = mapping.get(key)
        kpath = join_path(path, key)
        if value is None:
            if required:
                self.issues.add(path, f"missing required key {key!r}")
                return INVALID
            return None
        if not isinstance(value, str) or not value:
            self.issues.add(kpath, f"expected a non-empty string, got {got(value)}")
            return INVALID
        return value

    def enum(self, mapping: Mapping, key: str, path: str, choices: Sequence[str], default: str | None) -> str | None:
        value = mapping.get(key)
        if value is None:
            return default
        checked = self.enum_value(value, join_path(path, key), choices)
        return default if checked is INVALID else checked  # type: ignore[return-value]

    def enum_value(self, value: object, path: str, choices: Sequence[str]) -> str | _Invalid:
        if isinstance(value, str) and value in choices:
            return value
        message = f"expected one of {', '.join(choices)}, got {got(value)}"
        if isinstance(value, str):
            hint = suggest(value, choices)
            if hint is not None:
                message += f"; did you mean {hint!r}?"
        self.issues.add(path, message)
        return INVALID

    def string_list(
        self,
        mapping: Mapping,
        key: str,
        path: str,
        *,
        default: list[str] | None,
        unique: bool = True,
        max_items: int | None = None,
    ) -> list[str] | None:
        value = mapping.get(key)
        kpath = join_path(path, key)
        if value is None:
            return None if default is None else list(default)
        if not _is_list(value):
            self.issues.add(kpath, f"expected a list of strings, got {got(value)}")
            return None if default is None else list(default)
        out: list[str] = []
        for i, item in enumerate(value):
            if not isinstance(item, str) or not item:
                self.issues.add(item_path(kpath, i), f"expected a non-empty string, got {got(item)}")
                continue
            if unique and item in out:
                self.issues.add(item_path(kpath, i), f"duplicate entry {item!r}")
                continue
            out.append(item)
        if max_items is not None and len(value) > max_items:
            self.issues.add(kpath, f"at most {max_items} entries are allowed, got {len(value)}")
        return out

    def non_negative_int(self, mapping: Mapping, key: str, path: str) -> int | None:
        value = mapping.get(key)
        if value is None:
            return None
        if not _is_int(value) or value < 0:
            self.issues.add(join_path(path, key), f"expected a non-negative integer, got {got(value)}")
            return None
        return int(value)

    def quantity(
        self,
        raw: object,
        path: str,
        kind: str,
        *,
        sign: str = "non_negative",
    ) -> Quantity | _Invalid:
        """Parse a value, delta or duration quantity. `sign` is "any", "non_negative" or "positive"."""
        try:
            q = parse_quantity(raw, path=path, allow_unknown=kind != "duration")  # type: ignore[arg-type]
            if kind == "duration":
                to_seconds(q, path=path)
        except PolicyError as exc:
            self.issues.absorb(exc)
            return INVALID
        if kind == "value":
            unit = lookup_unit(q.unit)
            if unit is not None and unit.dimension == "bytes":
                self.issues.add(path, f"{q.text!r} is a byte size, which no signal value can be")
                return INVALID
            return q
        if sign == "positive" and not q.value > 0:
            self.issues.add(path, f"must be positive, got {q.text!r}")
            return INVALID
        if sign == "non_negative" and q.value < 0:
            self.issues.add(path, f"must not be negative, got {q.text!r}")
            return INVALID
        return q

    def default_quantity(self, default: object) -> Quantity:
        return parse_quantity(default)  # type: ignore[arg-type]

    # ----- sections -----------------------------------------------------------------------

    def run(self, data: object) -> Policy:
        if not isinstance(data, Mapping):
            if data is None:
                raise PolicyError([Issue("", "the policy is empty; at least 'version: 1' is required")])
            raise PolicyError([Issue("", f"a policy must be a mapping, got {got(data)}")])
        self.check_keys(data, TOP_KEYS, "")

        version = data.get("version")
        if version is None:
            self.issues.add("version", "missing required key 'version' (must be 1)")
        elif not _is_int(version):
            self.issues.add("version", f"expected the integer 1, got {got(version)}")
        elif version != 1:
            self.issues.add("version", f"unsupported policy version {version}; only version 1 is supported")

        name = self.string(data, "name", "")
        if not isinstance(name, str):
            name = "policy"

        artifact = self.artifact(data)
        signals = self.signals(data)
        hard = self.hard(data)
        events = self.events(data)
        trajectories = self.trajectories(data)
        sync_groups = self.sync_groups(data)
        soft = self.soft(data)
        review = self.review(data, hard)
        self.issues.raise_if_any()

        policy = Policy(
            version=1,
            name=name,
            artifact=artifact,
            signals=signals,
            hard=hard,
            events=events,
            trajectories=trajectories,
            sync_groups=sync_groups,
            soft=soft,
            review=review,
            canonical={},
            sha256="",
            source_format="dict",
            source_text=None,
            locations=dict(self.issues.locations),
        )
        return policy

    def artifact(self, data: Mapping) -> ArtifactSpec:
        spec = ArtifactSpec()
        m = self.section(data, "artifact", "artifact")
        if not m:
            return spec
        self.check_keys(m, ARTIFACT_KEYS, "artifact")
        raw = m.get("max_size")
        if raw is not None:
            path = "artifact.max_size"
            try:
                count = parse_bytes(raw, path=path)  # type: ignore[arg-type]
            except PolicyError as exc:
                self.issues.absorb(exc)
            else:
                if count == 0:
                    self.issues.add(path, "must be positive, got 0 bytes")
                else:
                    # A byte quantity keeps its written form ('2 MiB'); a plain number is kept as
                    # the integer count, so 2097152 and '2097152' canonicalize alike.
                    written = isinstance(raw, str) and parse_quantity(raw, path=path).unit is not None
                    spec.max_size = raw.strip() if written else count  # type: ignore[union-attr]
                    spec.max_bytes = count
        spec.codec = self.enum(m, "codec", "artifact", CODECS, "deflate")  # type: ignore[assignment]
        spec.hash = self.enum(m, "hash", "artifact", HASH_MODES, None)
        spec.soft_value_dtype = self.enum(m, "soft_value_dtype", "artifact", SOFT_VALUE_DTYPES, "source")  # type: ignore[assignment]
        spec.on_not_applicable = self.enum(m, "on_not_applicable", "artifact", ON_NOT_APPLICABLE, "warn")  # type: ignore[assignment]
        return spec

    def signals(self, data: Mapping) -> SignalsSpec:
        spec = SignalsSpec()
        m = self.section(data, "signals", "signals")
        if not m:
            return spec
        self.check_keys(m, SIGNALS_KEYS, "signals")
        time = self.string(m, "time", "signals")
        spec.time = time if isinstance(time, str) else None

        raw_unit = m.get("time_unit")
        if raw_unit is not None:
            unit = lookup_unit(raw_unit) if isinstance(raw_unit, str) else None
            if unit is None or unit.dimension != "time":
                time_units = [u.symbol for u in UNITS.values() if u.dimension == "time"]
                message = f"expected a time unit ({', '.join(time_units)}), got {got(raw_unit)}"
                if isinstance(raw_unit, str):
                    hint = suggest(raw_unit, time_units)
                    if hint is not None:
                        message += f"; did you mean {hint!r}?"
                self.issues.add("signals.time_unit", message)
            else:
                spec.time_unit = raw_unit
        spec.on_non_monotonic = self.enum(m, "on_non_monotonic", "signals", ON_NON_MONOTONIC, "error")  # type: ignore[assignment]
        spec.include = self.string_list(m, "include", "signals", default=["*"], unique=False)  # type: ignore[assignment]
        spec.exclude = self.string_list(m, "exclude", "signals", default=[], unique=False)  # type: ignore[assignment]

        decl = self.section(m, "decl", "signals.decl")
        for alias, raw in self.string_keys(decl or {}, "signals.decl"):
            path = join_path("signals.decl", alias)
            if raw is None:
                raw = {}
            if not isinstance(raw, Mapping):
                self.issues.add(path, f"expected a mapping with path, unit, kind or time, got {got(raw)}")
                continue
            self.check_keys(raw, DECL_KEYS, path)
            entry = SignalDecl(alias=alias)
            for key in ("path", "unit", "time"):
                value = self.string(raw, key, path)
                setattr(entry, key, value if isinstance(value, str) else None)
            entry.kind = self.enum(raw, "kind", path, SIGNAL_KINDS, None)
            spec.decls[alias] = entry
        return spec

    def hard(self, data: Mapping) -> list[HardReq]:
        reqs: list[HardReq] = []
        m = self.section(data, "hard", "hard")
        for signal, ops in self.string_keys(m or {}, "hard"):
            sig_path = join_path("hard", signal)
            if not isinstance(ops, Mapping):
                self.issues.add(sig_path, f"expected a mapping of operator -> spec, got {got(ops)}")
                continue
            if not ops:
                self.issues.add(sig_path, "expected at least one operator")
                continue
            for op, spec in ops.items():
                if not isinstance(op, str):
                    self.issues.add(sig_path, f"keys must be strings, got {got(op)}")
                    continue
                op_path = join_path(sig_path, op)
                if op not in OP_SCHEMAS:
                    hint = suggest(op, list(OP_SCHEMAS))
                    if hint is not None:
                        message = f"unknown operator {op!r}; did you mean {hint!r}?"
                    else:
                        message = f"unknown operator {op!r}; expected one of {', '.join(OP_SCHEMAS)}"
                    self.issues.add(op_path, message)
                    continue
                if spec is None or isinstance(spec, Mapping):
                    req = self.hard_spec(spec or {}, signal, op, op_path)
                    if req is not None:
                        reqs.append(req)
                elif _is_list(spec):
                    if not spec:
                        self.issues.add(op_path, "expected at least one spec")
                    for i, item in enumerate(spec):
                        ipath = item_path(op_path, i)
                        if item is not None and not isinstance(item, Mapping):
                            self.issues.add(ipath, f"expected a mapping, got {got(item)}")
                            continue
                        req = self.hard_spec(item or {}, signal, op, ipath)
                        if req is not None:
                            reqs.append(req)
                else:
                    self.issues.add(op_path, f"expected a mapping or a list of mappings, got {got(spec)}")
        return reqs

    def hard_spec(self, spec: Mapping, signal: str, op: str, path: str) -> HardReq | None:
        schema = OP_SCHEMAS[op]
        self.check_keys(spec, (*schema, "severity"), path)
        before = len(self.issues.issues)
        params: dict[str, object] = {}
        for name, param in schema.items():
            raw = spec.get(name)
            if raw is None:
                if param.required:
                    self.issues.add(path, f"missing required parameter {name!r}")
                elif param.default is not None:
                    params[name] = self.default_param(param)
                continue
            value = self.param(raw, param, op, name, join_path(path, name))
            if value is not INVALID:
                params[name] = value
        if op == "violation":
            given = [key for key in ("above", "below") if spec.get(key) is not None]
            if len(given) == 2:
                self.issues.add(path, "give exactly one of 'above' or 'below', not both")
            elif not given:
                self.issues.add(path, "missing required parameter: give one of 'above' or 'below'")
        severity = self.enum(spec, "severity", path, SEVERITIES, default_severity(op))
        if len(self.issues.issues) != before:
            return None
        return HardReq(id=path, signal=signal, op=op, params=params, severity=severity or default_severity(op))

    def default_param(self, param: Param) -> object:
        if param.kind in ("value", "delta", "duration"):
            return self.default_quantity(param.default)
        return param.default

    def param(self, raw: object, param: Param, op: str, name: str, path: str) -> object:
        if param.kind == "enum":
            return self.enum_value(raw, path, param.choices)
        if param.kind == "int":
            if not _is_int(raw):
                self.issues.add(path, f"expected an integer, got {got(raw)}")
                return INVALID
            return int(raw)  # type: ignore[call-overload]
        if (op, name) in _SIGNED_PARAMS:
            sign = "any"
        elif (op, name) in _POSITIVE_PARAMS:
            sign = "positive"
        else:
            sign = "non_negative"
        return self.quantity(raw, path, param.kind, sign=sign)

    def events(self, data: Mapping) -> list[EventSpec]:
        out: list[EventSpec] = []
        m = self.section(data, "events", "events")
        for name, spec in self.string_keys(m or {}, "events"):
            path = join_path("events", name)
            if not isinstance(spec, Mapping):
                self.issues.add(path, f"expected a mapping, got {got(spec)}")
                continue
            before = len(self.issues.issues)
            self.check_keys(spec, EVENT_KEYS, path)
            event = self.event(name, spec, path)
            if event is not None and len(self.issues.issues) == before:
                out.append(event)
        return out

    def event(self, name: str, spec: Mapping, path: str) -> EventSpec | None:
        when_path = join_path(path, "when")
        when = spec.get("when")
        if when is None:
            self.issues.add(path, "missing required key 'when'")
            return None
        if not isinstance(when, Mapping):
            self.issues.add(when_path, f"expected a mapping, got {got(when)}")
            return None
        self.check_keys(when, WHEN_KEYS, when_path)
        signal = self.string(when, "signal", when_path, required=True)
        triggers = [key for key in TRIGGERS if when.get(key) is not None]
        if not triggers:
            self.issues.add(when_path, "missing trigger: give one of falls_below, rises_above or equals")
            return None
        if len(triggers) > 1:
            self.issues.add(when_path, f"give exactly one of falls_below, rises_above or equals, got {', '.join(triggers)}")
            return None
        trigger = triggers[0]
        raw = when[trigger]
        tpath = join_path(when_path, trigger)
        hysteresis: Quantity | None | _Invalid = None
        debounce: Quantity | None | _Invalid = None
        if trigger == "equals":
            value: object = raw
            if isinstance(raw, bool) or not isinstance(raw, (str, numbers.Real)) or raw == "":
                self.issues.add(tpath, f"expected a number or a label string, got {got(raw)}")
            elif isinstance(raw, numbers.Integral):
                value = int(raw)
            elif isinstance(raw, numbers.Real):
                if not math.isfinite(float(raw)):
                    self.issues.add(tpath, f"expected a finite number or a label string, got {got(raw)}")
                value = float(raw)
            for key in ("hysteresis", "debounce"):
                if when.get(key) is not None:
                    self.issues.add(join_path(when_path, key), f"{key} applies only to falls_below and rises_above")
        else:
            value = self.quantity(raw, tpath, "value")
            raw_h = when.get("hysteresis")
            raw_d = when.get("debounce")
            hysteresis = (
                parse_quantity(0) if raw_h is None else self.quantity(raw_h, join_path(when_path, "hysteresis"), "delta")
            )
            debounce = (
                parse_quantity("0 s")
                if raw_d is None
                else self.quantity(raw_d, join_path(when_path, "debounce"), "duration")
            )

        occurrence = self.enum(spec, "occurrence", path, OCCURRENCES, "first")
        expect = self.non_negative_int(spec, "expect", path)
        severity = self.enum(spec, "severity", path, SEVERITIES, "info")

        before_q: Quantity | _Invalid = parse_quantity("0 s")
        after_q: Quantity | _Invalid = parse_quantity("0 s")
        keep_signals: list[str] | None = None
        keep_path = join_path(path, "keep")
        keep = self.section(spec, "keep", keep_path)
        if keep:
            self.check_keys(keep, KEEP_KEYS, keep_path)
            if keep.get("before") is not None:
                before_q = self.quantity(keep["before"], join_path(keep_path, "before"), "duration")
            if keep.get("after") is not None:
                after_q = self.quantity(keep["after"], join_path(keep_path, "after"), "duration")
            keep_signals = self.string_list(keep, "signals", keep_path, default=None)

        if not isinstance(signal, str) or INVALID in (value, hysteresis, debounce, before_q, after_q):
            return None
        return EventSpec(
            name=name,
            signal=signal,
            trigger=trigger,
            value=value,  # type: ignore[arg-type]
            hysteresis=hysteresis,  # type: ignore[arg-type]
            debounce=debounce,  # type: ignore[arg-type]
            occurrence=occurrence or "first",
            expect=expect,
            before=before_q,  # type: ignore[arg-type]
            after=after_q,  # type: ignore[arg-type]
            signals=keep_signals,
            severity=severity or "info",
        )

    def trajectories(self, data: Mapping) -> list[TrajectorySpec]:
        out: list[TrajectorySpec] = []
        m = self.section(data, "trajectories", "trajectories")
        for name, spec in self.string_keys(m or {}, "trajectories"):
            path = join_path("trajectories", name)
            if not isinstance(spec, Mapping):
                self.issues.add(path, f"expected a mapping, got {got(spec)}")
                continue
            before = len(self.issues.issues)
            self.check_keys(spec, TRAJECTORY_KEYS, path)

            position = spec.get("position")
            ppath = join_path(path, "position")
            pos_value: str | list[str] | _Invalid = INVALID
            if position is None:
                self.issues.add(path, "missing required key 'position'")
            elif isinstance(position, str) and position:
                pos_value = position
            elif _is_list(position):
                names = self.string_list(spec, "position", path, default=None)
                if len(position) != 3:
                    self.issues.add(ppath, f"expected exactly three signals [x, y, z], got {len(position)}")
                elif names is not None and len(names) == 3:
                    pos_value = names
            else:
                self.issues.add(ppath, f"expected a signal name or a list of three signal names, got {got(position)}")

            eps: Quantity | _Invalid = INVALID
            raw_eps = spec.get("max_position_error")
            epath = join_path(path, "max_position_error")
            if raw_eps is None:
                self.issues.add(path, "missing required key 'max_position_error'")
            else:
                eps = self.quantity(raw_eps, epath, "delta")
                unit = lookup_unit(eps.unit) if isinstance(eps, Quantity) else None
                if unit is not None and unit.dimension != "length":
                    self.issues.add(
                        epath, f"{eps.text!r} is {describe_dimension(unit.dimension)} but a length is required"  # type: ignore[union-attr]
                    )
                    eps = INVALID

            max_time: Quantity | None | _Invalid = None
            if spec.get("max_time_error") is not None:
                max_time = self.quantity(spec["max_time_error"], join_path(path, "max_time_error"), "duration")
            linked = self.string_list(spec, "linked", path, default=[])

            if len(self.issues.issues) == before:
                out.append(
                    TrajectorySpec(
                        name=name,
                        position=pos_value,  # type: ignore[arg-type]
                        max_position_error=eps,  # type: ignore[arg-type]
                        max_time_error=max_time,  # type: ignore[arg-type]
                        linked=linked or [],
                    )
                )
        return out

    def sync_groups(self, data: Mapping) -> list[SyncGroupSpec]:
        out: list[SyncGroupSpec] = []
        m = self.section(data, "sync_groups", "sync_groups")
        for name, spec in self.string_keys(m or {}, "sync_groups"):
            path = join_path("sync_groups", name)
            if not isinstance(spec, Mapping):
                self.issues.add(path, f"expected a mapping with members, got {got(spec)}")
                continue
            before = len(self.issues.issues)
            self.check_keys(spec, SYNC_KEYS, path)
            if spec.get("members") is None:
                self.issues.add(path, "missing required key 'members'")
                continue
            members = self.string_list(spec, "members", path, default=None)
            if members is not None and _is_list(spec["members"]) and len(spec["members"]) < 2:
                self.issues.add(join_path(path, "members"), f"a sync group needs at least two members, got {len(spec['members'])}")
            if len(self.issues.issues) == before and members is not None:
                out.append(SyncGroupSpec(name=name, members=members))
        return out

    def soft(self, data: Mapping) -> list[SoftRule]:
        rules: list[SoftRule] = []
        raw = data.get("soft")
        if raw is None:
            return rules
        if not _is_list(raw):
            self.issues.add("soft", f"expected a list of rules, got {got(raw)}")
            return rules
        for i, item in enumerate(raw):
            path = item_path("soft", i)
            if not isinstance(item, Mapping):
                self.issues.add(path, f"expected a mapping with match, priority and max_points, got {got(item)}")
                continue
            before = len(self.issues.issues)
            self.check_keys(item, SOFT_KEYS, path)
            match = self.string(item, "match", path, required=True)
            priority = self.enum(item, "priority", path, tuple(SOFT_WEIGHTS), DEFAULT_SOFT_PRIORITY)
            max_points = self.non_negative_int(item, "max_points", path)
            if len(self.issues.issues) == before and isinstance(match, str):
                rules.append(SoftRule(match=match, priority=priority or DEFAULT_SOFT_PRIORITY, max_points=max_points))
        return rules

    def review(self, data: Mapping, hard: list[HardReq]) -> ReviewSpec:
        spec = ReviewSpec()
        # The default is sorted, not taken in the order the hard signals happen to appear: YAML and
        # JSON mappings are unordered, and this default is part of the canonical form and the hash.
        spec.thumbnails = sorted({req.signal for req in hard})[:MAX_THUMBNAILS]
        m = self.section(data, "review", "review")
        if not m:
            return spec
        self.check_keys(m, REVIEW_KEYS, "review")
        spec.thumbnails = self.string_list(  # type: ignore[assignment]
            m, "thumbnails", "review", default=spec.thumbnails, max_items=MAX_THUMBNAILS
        )
        spec.kpis = self.string_list(m, "kpis", "review", default=[])  # type: ignore[assignment]
        return spec


def validate_policy(
    data: object,
    *,
    source_format: str = "dict",
    source_text: str | None = None,
    locations: Mapping[str, str] | None = None,
) -> Policy:
    """Validate parsed policy data and return a Policy with defaults, canonical form and hash."""
    validator = _Validator(locations or {})
    policy = validator.run(data)
    policy.source_format = source_format
    policy.source_text = source_text
    policy.canonical = canonical_dict(policy)
    policy.sha256 = canonical_sha256(policy.canonical)
    return policy
