"""A compact, verified copy of a checked run: a policy made from the requirements, compiled with baslt.

The derived policy keeps the signals the requirements read and protects what they check, so the artifact can be
reviewed and verified later without the full run:

    a limit on a signal              global extrema, and a violation for each constant limit (its tolerance as
                                     the minimum duration)
    a duration of `signal > value`   threshold crossings at the value
    max(), min() of a signal         global extrema
    a state signal                   state transitions
    events on one signal             the same events, with a second of data around them

Checks on derived signals (expressions) keep their input signals but add no protected facts.
"""

from __future__ import annotations

import glob
from pathlib import Path

from ..errors import BasltError, Issue
from ..units import lookup_unit
from .bind import BoundRequirement, bind_requirements, open_run_source
from .check import _unwrap

__all__ = ["archive_run", "derived_policy"]

EVENT_KEEP = {"before": "1 s", "after": "1 s"}


def _plain_signal(node):
    """(canonical name, alias name) when `node` reads one signal as it is, else None."""
    node = _unwrap(node)
    if node.op == "sig":
        return node.value
    return None


class _Builder:
    def __init__(self, config, infos, index) -> None:
        self.config = config
        self.infos = {info.name: info for info in infos}
        self.index = index
        self.hard: dict[str, dict] = {}
        self.decl: dict[str, dict] = {}
        self.names: dict[str, str] = {}  # canonical name -> the name the policy uses for it
        self.aliases: dict[str, str] = {}  # canonical name -> the first mapping alias that reads it as it is
        for name, spec in config.signals.items():
            if spec.path is None:
                continue
            try:
                self.aliases.setdefault(index.lookup(spec.path), name)
            except LookupError:
                continue

    def name_for(self, canonical: str, alias: str | None) -> str:
        found = self.names.get(canonical)
        if found is not None:
            return found
        alias = alias or self.aliases.get(canonical)
        spec = self.config.signals.get(alias) if alias else None
        if spec is not None and spec.path is not None and alias not in self.infos:
            entry: dict = {"path": canonical}
            if spec.unit and lookup_unit(spec.unit) is not None:
                entry["unit"] = spec.unit
            if spec.kind:
                entry["kind"] = spec.kind
            self.decl[alias] = entry
            self.names[canonical] = alias
            return alias
        self.names[canonical] = canonical
        return canonical

    def add(self, name: str, op: str, spec: dict | None = None) -> None:
        ops = self.hard.setdefault(name, {})
        if spec is None:
            ops.setdefault(op, {})
            return
        items = ops.setdefault(op, [])
        if spec not in items:
            items.append(spec)

    def values_ok(self, alias: str | None, unit: str | None) -> bool:
        """Whether a bare number in the check's unit means the same in the policy (the signal's declared unit)."""
        if unit in (None, "?"):
            return unit is None
        spec = self.config.signals.get(alias) if alias else None
        return spec is None or not spec.unit or lookup_unit(spec.unit) is not None

    def discrete(self, canonical: str) -> bool:
        info = self.infos.get(canonical)
        return info is not None and info.kind == "discrete"

    def requirement(self, item: BoundRequirement) -> None:
        if not item.ok or item.check is None:
            return
        root = _unwrap(item.check.root)
        if item.kind == "bound":
            plain = _plain_signal(root)
            if plain is None or item.check.type.components != 1:
                return
            name = self.name_for(*plain)
            if self.discrete(plain[0]):
                self.add(name, "state_transitions")
                return
            self.add(name, "global_extrema")
            if not self.values_ok(plain[1], item.unit):
                return
            for case in item.cases:
                limit = case.limit
                if limit is None:
                    continue
                for side, key in ((limit.upper, "above"), (limit.lower, "below")):
                    if side is None or side.node.op != "num":
                        continue
                    spec = {key: float(side.node.value)}
                    if case.tol_seconds > 0:
                        spec["min_duration"] = f"{case.tol_seconds!r} s"
                    self.add(name, "violation", spec)
        elif item.kind == "duration" and root.op == "cmp" and root.value in ("<", "<=", ">", ">="):
            plain = _plain_signal(root.args[0])
            level = root.args[1]
            if plain is None or level.op != "num" or self.discrete(plain[0]):
                return
            if self.values_ok(plain[1], root.args[0].type.unit):
                self.add(self.name_for(*plain), "threshold_crossing", {"value": float(level.value)})
        elif item.kind == "value" and root.op == "agg" and root.value in ("max", "min"):
            plain = _plain_signal(root.args[0])
            if plain is not None and not self.discrete(plain[0]):
                self.add(self.name_for(*plain), "global_extrema")
        for name in item.signals():
            if self.discrete(name):
                self.add(self.name_for(name, None), "state_transitions")

    def events(self, index) -> dict:
        out = {}
        for name, defn in self.config.events.items():
            if defn.condition is None or defn.signal is None:
                continue
            alias = self.config.signals.get(defn.signal)
            try:
                if alias is not None and alias.path is not None:
                    canonical = index.lookup(alias.path)
                elif alias is None:
                    canonical = index.lookup(defn.signal.strip("`"))
                else:
                    continue
            except LookupError:
                continue
            when: dict = {"signal": self.name_for(canonical, alias.name if alias else None)}
            value = defn.value
            if isinstance(value, str):
                parsed = _core_quantity(value)
                if parsed is None:
                    continue
                value = parsed
            when[defn.condition] = value
            for key in ("hysteresis", "debounce"):
                text = getattr(defn, key)
                if text:
                    if _core_quantity(text) is None:
                        break
                    when[key] = text
            else:
                out[_policy_name(name)] = {"when": when, "occurrence": "all", "keep": dict(EVENT_KEEP)}
        return out


def _core_quantity(text: str):
    """`text` when the policy's unit table reads it, a number for a bare number, else None."""
    from ..units import parse_quantity

    try:
        quantity = parse_quantity(text)
    except (BasltError, ValueError):
        return None
    if quantity.unit is None:
        return quantity.value
    return text


def _policy_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name) or "event"


def derived_policy(reqset, bound, infos, *, max_size=None) -> dict:
    """The policy that keeps what `bound` (the requirements bound to this source) checks."""
    from ..policy import SignalIndex

    config = reqset.config
    index = SignalIndex(list(infos))
    builder = _Builder(config, infos, index)
    for item in bound.requirements:
        builder.requirement(item)
    events = builder.events(index)
    wanted = list(dict.fromkeys(bound.signals))
    for name in wanted:
        builder.name_for(name, None)
    signals: dict = {"include": [glob.escape(name) for name in wanted] or ["-"], "exclude": []}
    if config.time.signal:
        signals["time"] = config.time.signal
    if config.time.unit:
        signals["time_unit"] = config.time.unit
    if config.time.on_non_monotonic != "error":
        signals["on_non_monotonic"] = config.time.on_non_monotonic
    if builder.decl:
        signals["decl"] = builder.decl
    policy: dict = {"version": 1, "name": _policy_name(config.name or reqset.table.path.stem),
                    "signals": signals, "artifact": {"on_not_applicable": "warn"}}
    budget = max_size if max_size is not None else config.archive_max_size
    if budget is not None:
        policy["artifact"]["max_size"] = budget
    if builder.hard:
        policy["hard"] = builder.hard
    if events:
        policy["events"] = events
    return policy


def archive_run(source, reqset, run, path, *, max_size=None, hash: str = "sampled") -> tuple[Path | None, list]:
    """Compile `source` to `path` with the derived policy; problems come back as issues, not exceptions."""
    from ..api import compile

    issues: list[Issue] = []
    try:
        adapter, _ = open_run_source(source, reqset.config)
        infos = adapter.list_signals()
        bound = bind_requirements(reqset, infos, param_names=set(run.params) if run.params else None)
        policy = derived_policy(reqset, bound, infos, max_size=max_size)
        result = compile(source, policy, output=path, hash="none" if hash == "none" else "full")
    except BasltError as exc:
        issues.append(Issue(path="archive", message=f"the run could not be archived: {exc}",
                            location=Path(str(path)).name))
        return None, issues
    if result.status != "pass":
        issues.append(Issue(path="archive", message=f"the archive was written with status {result.status}",
                            location=Path(str(path)).name))
    return Path(path), issues
