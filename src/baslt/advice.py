"""Advice for `baslt explain`: what to change when the hard requirements do not fit the byte budget.

A compile that cannot fit its budget writes an infeasible-budget report (`<run>.error.json`) listing the minimum
size and every requirement's sample count and standalone bytes. That report is cheap to produce and is all this
module needs for its first answer: the smallest budget that works, and the largest requirements with the
relaxation that usually helps each operator.

Given the source and the policy as well, the relaxations are measured instead of guessed. Each one changes a single
bound requirement or event (a longer window, a larger prominence, hysteresis and debounce for a chattering
threshold, a shorter event window, or demoting the requirement to the soft layer) and the smallest artifact the
changed requirements allow is sized the way the compiler sizes it. The artifact also embeds the policy text, which
the edit itself changes by a few bytes, so a change only counts as fitting with POLICY_EDIT_MARGIN bytes to spare.
The best change per requirement is then applied greedily until the run fits or the changes run out. Nothing is
written: the suggestions describe edits to make in the policy file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from .errors import UsageError

__all__ = ["explain", "render", "suggested_budget"]

REPORT_KIND = "infeasible_budget"
BUDGET_STEP = 64 * 1024  # suggested budgets are whole multiples of 64 KiB
TOP_REQUIREMENTS = 8
MAX_COMBINED = 6
SHOWN_RELAXATIONS = 12
POLICY_EDIT_MARGIN = 64  # bytes left for the policy text an edit adds to the artifact
FACTORS = (2.0, 4.0)
RANGE_FRACTION = 0.01  # hysteresis suggested for a chattering threshold: 1% of the signal's range
STEP_MULTIPLE = 10  # debounce and minimum durations suggested from the median sample spacing

ADVICE: dict[str, str] = {
    "global_extrema": "keeps only a few samples; demoting it to soft saves little",
    "window_extrema": "use a longer interval: every window keeps its own maximum and minimum",
    "local_extrema": "raise the prominence or add a separation so small wiggles stop counting as peaks",
    "threshold_crossing": "add hysteresis or debounce: a noisy signal near the level crosses it many times",
    "violation": "raise min_duration so short excursions stop counting as violations",
    "state_transitions": "exclude the signal or merge states that toggle rapidly",
    "event": "shorten the event window (before/after) or narrow its signal list",
}


# ---------------------------------------------------------------------------------------------------------------
# Report-only suggestions


def _load_report(report: object) -> dict:
    if isinstance(report, Mapping):
        data = dict(report)
    else:
        path = Path(report)  # type: ignore[arg-type]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise UsageError(f"no error report at {path}") from None
        except (OSError, ValueError) as exc:
            raise UsageError(f"{path} is not a readable JSON error report: {exc}") from None
    if not isinstance(data, dict) or data.get("kind") != REPORT_KIND:
        found = data.get("kind") if isinstance(data, dict) else type(data).__name__
        raise UsageError(
            f"baslt explain needs an infeasible-budget report (kind {REPORT_KIND!r}), got {found!r}; "
            "only a compile that did not fit its budget writes one"
        )
    for key in ("max_bytes", "minimum_bytes"):
        if not isinstance(data.get(key), int) or isinstance(data.get(key), bool):
            raise UsageError(f"the error report has no integer {key!r}")
    return data


def suggested_budget(minimum_bytes: int) -> int:
    """The smallest multiple of 64 KiB that holds `minimum_bytes`."""
    return -(-int(minimum_bytes) // BUDGET_STEP) * BUDGET_STEP


def _size_text(count: int) -> str:
    """A budget as a policy would write it: whole MiB when possible, else KiB."""
    kib = count // 1024
    return f"{kib // 1024} MiB" if kib % 1024 == 0 else f"{kib} KiB"


def _report_suggestions(data: Mapping) -> list[dict]:
    budget = suggested_budget(data["minimum_bytes"])
    suggestions = [{
        "kind": "budget",
        "target": "artifact.max_size",
        "change": f"raise the budget to {_size_text(budget)}",
        "max_size": _size_text(budget),
        "max_bytes": budget,
    }]
    ranked = sorted(
        (entry for entry in data.get("requirements", []) or []
         if isinstance(entry, dict) and int(entry.get("samples", 0) or 0) > 0),
        key=lambda entry: (-int(entry.get("standalone_bytes", 0) or 0), str(entry.get("id"))),
    )
    for entry in ranked[:TOP_REQUIREMENTS]:
        suggestions.append({
            "kind": "requirement",
            "target": str(entry.get("id")),
            "change": ADVICE.get(str(entry.get("op")), "relax or demote this requirement"),
            "samples": int(entry.get("samples", 0) or 0),
            "standalone_bytes": int(entry.get("standalone_bytes", 0) or 0),
        })
    return suggestions


# ---------------------------------------------------------------------------------------------------------------
# Measured relaxations


def _number(value: float, unit: str | None) -> str:
    text = f"{value:.6g}"
    return f"{text} {unit}" if unit else text


def _median_step(t: np.ndarray) -> float:
    steps = np.diff(t)
    steps = steps[steps > 0]
    return float(np.median(steps)) if steps.size else 0.0


def _value_range(v: np.ndarray) -> float:
    values = np.asarray(v, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.max() - finite.min()) if finite.size else 0.0


def _grow(params: Mapping, key: str, unit: str | None, floor: float = 0.0) -> Iterator[tuple[str, dict]]:
    """Parameter sets with `key` multiplied by each factor, or raised to `floor` when it is zero."""
    current = float(params.get(key, 0.0) or 0.0)
    seen: set[float] = set()
    for factor in FACTORS:
        value = current * factor if current > 0 else floor * factor / FACTORS[0]
        if value <= current or value in seen:
            continue
        seen.add(value)
        yield f"{key} {_number(current, unit)} -> {_number(value, unit)}", {**params, key: value}


def _requirement_variants(req, sig, unit: str | None) -> Iterator[tuple[str, dict | None]]:
    """(description, new params) for each relaxation of one requirement; None params drops it."""
    params = dict(req.params)
    step = _median_step(sig.t) * STEP_MULTIPLE
    if req.op == "window_extrema":
        yield from _grow(params, "interval", "s")
    elif req.op == "local_extrema":
        yield from _grow(params, "prominence", unit)
        if not params.get("separation"):
            yield from _grow(params, "separation", "s", step)
    elif req.op == "threshold_crossing":
        yield from _grow(params, "hysteresis", unit, _value_range(sig.v) * RANGE_FRACTION)
        yield from _grow(params, "debounce", "s", step)
    elif req.op == "violation":
        yield from _grow(params, "min_duration", "s", step)
    yield "demote to soft", None


def _event_variants(event) -> Iterator[tuple[str, object | None]]:
    for factor in FACTORS:
        before, after = event.before / factor, event.after / factor
        if before < event.before or after < event.after:
            yield (f"window before {_number(event.before, 's')} -> {_number(before, 's')}, "
                   f"after {_number(event.after, 's')} -> {_number(after, 's')}"), replace(
                       event, before=before, after=after)
    yield "remove the event", None


def _dropped_policy(policy, reqs, events):
    """`policy` with the requirements and events these lists no longer hold.

    The artifact embeds the policy, so a drop shrinks it by that requirement's whole canonical stanza -- far more
    than POLICY_EDIT_MARGIN -- and measuring the drop against the original policy would claim a minimum the edited
    policy never needs. The source text is left alone: how the edit rewrites the user's own file is unknowable, and
    keeping it counts bytes the edit will only remove, which under-claims no budget.
    """
    from .policy import canonical_dict, canonical_sha256

    kept_reqs = {req.id for req in reqs}
    kept_events = {event.name for event in events}
    hard = [req for req in policy.hard if req.id in kept_reqs]
    keep = [event for event in policy.events if event.name in kept_events]
    if len(hard) == len(policy.hard) and len(keep) == len(policy.events):
        return policy
    edited = replace(policy, hard=hard, events=keep)
    canonical = canonical_dict(edited)
    return replace(edited, canonical=canonical, sha256=canonical_sha256(canonical))


def _with(bound, *, reqs=None, events=None):
    reqs = bound.reqs if reqs is None else reqs
    events = bound.events if events is None else events
    by_signal: dict[str, list] = {}
    for req in reqs:
        by_signal.setdefault(req.signal, []).append(req)
    return replace(bound, policy=_dropped_policy(bound.policy, reqs, events), reqs=list(reqs),
                   reqs_by_signal=by_signal, events=list(events))


def _candidates(run, bound, order: Sequence[str]) -> Iterator[tuple[str, str, Callable, bool]]:
    """(target, change, apply, drops) for every relaxation; `apply(reqs, events)` returns the changed lists and
    `drops` is True when the change removes the requirement or event instead of loosening it."""
    rank = {name: i for i, name in enumerate(order)}
    reqs = sorted(bound.reqs, key=lambda req: rank.get(req.id, len(rank)))[:TOP_REQUIREMENTS]
    for req in reqs:
        sig = run.signals[req.signal]
        for change, params in _requirement_variants(req, sig, bound.units.get(req.signal)):
            def apply(current_reqs, current_events, req_id=req.id, params=params):
                if params is None:
                    return [r for r in current_reqs if r.id != req_id], current_events
                return [replace(r, params=params) if r.id == req_id else r for r in current_reqs], current_events
            yield req.id, change, apply, params is None
    for event in bound.events:
        for change, new in _event_variants(event):
            def apply(current_reqs, current_events, name=event.name, new=new):
                if new is None:
                    return current_reqs, [e for e in current_events if e.name != name]
                return current_reqs, [new if e.name == name else e for e in current_events]
            yield f"events.{event.name}", change, apply, new is None


def _measure(data: Mapping, source, policy, max_size) -> dict:
    from .api import load_run
    from .hashing import HashInfo
    from .plan.compile import minimum_size
    from .policy import load_policy

    if max_size is None and data.get("budget_source") == "cli":
        max_size = int(data["max_bytes"])  # the compile's own --max-size
    run, bound = load_run(source, load_policy(policy), max_size=max_size)
    digest = HashInfo.from_json(data["digest"]) if isinstance(data.get("digest"), Mapping) else None
    target = int(data["max_bytes"]) if bound.max_bytes is None else int(bound.max_bytes)
    room = target - POLICY_EDIT_MARGIN

    def size(reqs, events) -> int:
        return minimum_size(run, _with(bound, reqs=reqs, events=events), digest=digest)

    baseline = size(bound.reqs, bound.events)
    order = [str(entry.get("id")) for entry in sorted(
        (e for e in data.get("requirements", []) or [] if isinstance(e, dict)),
        key=lambda e: -int(e.get("standalone_bytes", 0) or 0))]

    rows: list[dict] = []
    # Per target, the mildest loosening that fits (else the one that saves most), and the cost of dropping it.
    loosen: dict[str, tuple[int, str, Callable]] = {}
    drop: dict[str, tuple[int, str, Callable]] = {}
    for target_id, change, apply, drops in _candidates(run, bound, order):
        reqs, events = apply(bound.reqs, bound.events)
        measured = size(reqs, events)
        rows.append({"target": target_id, "change": change, "minimum_bytes": measured,
                     "saved_bytes": baseline - measured, "fits": measured <= room})
        if measured >= baseline:
            continue
        best = drop if drops else loosen
        known = best.get(target_id)
        if known is None or (known[0] > room and measured < known[0]):
            best[target_id] = (measured, change, apply)
    rows.sort(key=lambda row: (not row["fits"], row["minimum_bytes"], row["target"], row["change"]))

    # Loosen first, largest saving first; drop requirements only if loosening everything is not enough.
    steps: list[dict] = []
    reqs, events = bound.reqs, bound.events
    current = baseline
    plan = sorted(loosen.items(), key=lambda item: item[1][0]) + sorted(drop.items(), key=lambda item: item[1][0])
    for target_id, (_, change, apply) in plan:
        if current <= room or len(steps) == MAX_COMBINED:
            break
        trial_reqs, trial_events = apply(reqs, events)
        measured = size(trial_reqs, trial_events)
        if measured >= current:
            continue
        reqs, events, current = trial_reqs, trial_events, measured
        steps.append({"target": target_id, "change": change, "minimum_bytes": current})
    return {
        "max_bytes": target,
        "minimum_bytes": baseline,
        "relaxations": rows,
        "margin_bytes": POLICY_EDIT_MARGIN,
        "combined": {"steps": steps, "minimum_bytes": current, "fits": current <= room},
    }


# ---------------------------------------------------------------------------------------------------------------
# Entry point and text form


def explain(report, *, source=None, policy=None, max_size=None) -> dict:
    """Suggestions for an infeasible-budget report; measured ones when `source` and `policy` are both given.

    `report` is the report's path or its parsed mapping. `max_size` overrides the budget to measure against, as
    `baslt compile --max-size` does; by default a report written under `--max-size` keeps that budget, and any
    other report uses the policy's.
    """
    data = _load_report(report)
    if (source is None) != (policy is None):
        raise UsageError("pass both --source and --policy to measure relaxations, or neither")
    result = {
        "status": "pass",
        "kind": "explanation",
        "max_bytes": int(data["max_bytes"]),
        "minimum_bytes": int(data["minimum_bytes"]),
        "excess_bytes": int(data["minimum_bytes"]) - int(data["max_bytes"]),
        "suggestions": _report_suggestions(data),
        "measured": None,
    }
    if source is not None:
        result["measured"] = _measure(data, source, policy, max_size)
    return result


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def render(result: Mapping) -> str:
    """The text `baslt explain` prints."""
    lines = [
        f"The hard requirements need {result['minimum_bytes']:,} bytes; the budget is {result['max_bytes']:,} "
        f"({result['excess_bytes']:,} over).",
        "",
    ]
    budget, *requirements = result["suggestions"]
    lines.append(f"Smallest budget that works: max_size: {budget['max_size']}")
    if requirements:
        lines += ["", "Largest requirements"]
        lines += _table([(item["target"], f"{item['samples']:,} samples", f"{item['standalone_bytes']:,} bytes",
                          item["change"]) for item in requirements])
    measured = result.get("measured")
    if measured:
        lines += ["", f"Measured on the source (smallest artifact; budget {measured['max_bytes']:,} bytes, "
                  f"FITS leaves {measured['margin_bytes']} bytes for the policy edit)"]
        relaxations = measured["relaxations"]
        rows = [("FITS" if row["fits"] else "", f"{row['minimum_bytes']:,}", row["target"], row["change"])
                for row in relaxations[:SHOWN_RELAXATIONS]]
        lines += _table(rows) if rows else ["  no requirement can be relaxed"]
        if len(relaxations) > SHOWN_RELAXATIONS:
            lines.append(f"  ... {len(relaxations) - SHOWN_RELAXATIONS} more with --json")
        combined = measured["combined"]
        if combined["steps"]:
            verdict = "fits" if combined["fits"] else "still does not fit"
            lines += ["", f"Together ({verdict} at {combined['minimum_bytes']:,} bytes):"]
            lines += [f"  {i}. {step['target']}: {step['change']}" for i, step in enumerate(combined["steps"], 1)]
    return "\n".join(lines) + "\n"
