"""Canonical form of a validated policy and its SHA-256.

The canonical form fills every default and keeps quantities as their original strings, so the
same policy written as YAML, JSON or a dict, with or without explicit defaults, hashes the same.
"""

from __future__ import annotations

import hashlib
import json
import re

from ..units import Quantity
from .schema import Policy

_INDEX_RE = re.compile(r"\[(\d+)\]$")


def _number(value: float) -> str:
    """One spelling for a bare number: 1e3, 1000 and 1000.0 all canonicalize as "1000"."""
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def _q(value: object) -> object:
    """A quantity as canonical text.

    A quantity with a unit keeps the string the policy used ("65 kPa"), but a bare number is
    normalized: YAML, JSON and dict input spell the same number differently (YAML 1.1 reads `1e3`
    as text, JSON as the number 1000.0), and those forms must hash alike.
    """
    if isinstance(value, Quantity):
        return value.text if value.unit is not None else _number(value.value)
    return value


def canonical_json(obj: object) -> str:
    """JSON with sorted keys, compact separators, UTF-8 characters unescaped and no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_sha256(obj: object) -> str:
    """SHA-256 hex digest of `canonical_json(obj)` encoded as UTF-8."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def canonical_dict(policy: Policy) -> dict:
    """Build the canonical dict of a validated policy."""
    artifact = policy.artifact
    signals = policy.signals

    hard: dict[str, dict[str, object]] = {}
    for req in policy.hard:
        spec = {name: _q(value) for name, value in req.params.items()}
        spec["severity"] = req.severity
        ops = hard.setdefault(req.signal, {})
        if _INDEX_RE.search(req.id):
            ops.setdefault(req.op, [])
            ops[req.op].append(spec)  # type: ignore[union-attr]
        else:
            ops[req.op] = spec

    events: dict[str, object] = {}
    for ev in policy.events:
        when: dict[str, object] = {"signal": ev.signal, ev.trigger: _q(ev.value)}
        if ev.trigger != "equals":
            when["hysteresis"] = _q(ev.hysteresis)
            when["debounce"] = _q(ev.debounce)
        events[ev.name] = {
            "when": when,
            "occurrence": ev.occurrence,
            "expect": ev.expect,
            "keep": {
                "before": _q(ev.before),
                "after": _q(ev.after),
                "signals": None if ev.signals is None else list(ev.signals),
            },
            "severity": ev.severity,
        }

    trajectories = {
        tr.name: {
            "position": tr.position if isinstance(tr.position, str) else list(tr.position),
            "max_position_error": _q(tr.max_position_error),
            "max_time_error": _q(tr.max_time_error),
            "linked": list(tr.linked),
        }
        for tr in policy.trajectories
    }

    return {
        "version": policy.version,
        "name": policy.name,
        "artifact": {
            "max_size": artifact.max_size,
            "codec": artifact.codec,
            "hash": artifact.hash,
            "soft_value_dtype": artifact.soft_value_dtype,
            "on_not_applicable": artifact.on_not_applicable,
        },
        "signals": {
            "time": signals.time,
            "time_unit": signals.time_unit,
            "on_non_monotonic": signals.on_non_monotonic,
            "include": list(signals.include),
            "exclude": list(signals.exclude),
            "decl": {
                # An omitted path defaults to the alias itself, filled in here so that both
                # spellings of the same declaration hash alike.
                alias: {"path": d.path if d.path is not None else alias, "unit": d.unit, "kind": d.kind, "time": d.time}
                for alias, d in signals.decls.items()
            },
        },
        "hard": hard,
        "events": events,
        "trajectories": trajectories,
        "sync_groups": {g.name: {"members": list(g.members)} for g in policy.sync_groups},
        "soft": [{"match": r.match, "priority": r.priority, "max_points": r.max_points} for r in policy.soft],
        "review": {"thumbnails": list(policy.review.thumbnails), "kpis": list(policy.review.kpis)},
    }
