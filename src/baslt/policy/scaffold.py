"""Starter policies written from a source's signals (``baslt policy init``).

A starter policy protects every signal with one simple guarantee the compiler supports today: global extrema for
continuous and vector signals, state transitions for discrete ones. Its size budget scales with the source. The
YAML form starts with a comment table of the signals found and is written without PyYAML; JSON is also available.
"""

from __future__ import annotations

import glob
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..signals import SignalInfo

KIB = 1024
MIN_BUDGET = 256 * KIB
MAX_BUDGET = 8 * KIB * KIB
DEFAULT_BUDGET = KIB * KIB
BUDGET_RATIO = 50

_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")
_RESERVED = frozenset({"y", "n", "yes", "no", "on", "off", "true", "false", "null"})


def budget_for(source_size: int | None) -> str:
    """A fiftieth of the source, rounded up to a power-of-two KiB and clamped to 256 KiB .. 8 MiB."""
    if source_size is None or source_size <= 0:
        target = DEFAULT_BUDGET
    else:
        target = min(max(source_size / BUDGET_RATIO, MIN_BUDGET), MAX_BUDGET)
    kib = 1 << math.ceil(math.log2(target / KIB))
    return f"{kib // KIB} MiB" if kib >= KIB else f"{kib} KiB"


def starter_policy(infos: Sequence[SignalInfo], *, name: str = "run", source_size: int | None = None,
                   require_time: bool = True) -> dict:
    """The starter policy as a plain mapping.

    With `require_time`, signals whose clock could not be found are excluded, since loading them would fail; file
    sources need this, while in-memory `(t, v)` pairs carry their own time and report no shared clock.
    """
    hard: dict[str, dict] = {}
    untimed: list[str] = []
    for info in infos:
        if require_time and info.time_ref is None:
            untimed.append(info.name)
            continue
        op = "state_transitions" if info.kind == "discrete" else "global_extrema"
        hard[info.name] = {op: {}}
    policy: dict = {"version": 1, "name": name, "artifact": {"max_size": budget_for(source_size)}}
    if untimed:
        policy["signals"] = {"exclude": [glob.escape(signal) for signal in untimed]}
    if hard:
        policy["hard"] = hard
    return policy


def _scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    text = str(value)
    if _PLAIN.match(text) and text.lower() not in _RESERVED:
        return text
    return json.dumps(text, ensure_ascii=False)  # a JSON string is a valid YAML double-quoted scalar


def _yaml(value: object, indent: int) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(child, Mapping) and child:
                lines.append(f"{pad}{_scalar(key)}:")
                lines.extend(_yaml(child, indent + 1))
            elif isinstance(child, Mapping):
                lines.append(f"{pad}{_scalar(key)}: {{}}")
            elif isinstance(child, (list, tuple)):
                items = ", ".join(_scalar(x) for x in child)
                lines.append(f"{pad}{_scalar(key)}: [{items}]")
            else:
                lines.append(f"{pad}{_scalar(key)}: {_scalar(child)}")
    return lines


def _dtype_label(dtype: str) -> str:
    import numpy as np

    kind = np.dtype(dtype).kind
    return "text" if kind in "OUS" else np.dtype(dtype).name


def _signal_table(infos: Sequence[SignalInfo]) -> list[str]:
    rows = [("SIGNAL", "SHAPE", "TYPE", "UNIT", "KIND", "TIME")]
    for info in infos:
        shape = "x".join(str(s) for s in info.shape) or "scalar"
        rows.append((info.name, shape, _dtype_label(info.dtype), info.unit or "-", info.kind, info.time_ref or "none"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["#   " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]


def render_yaml(policy: Mapping, infos: Sequence[SignalInfo], *, source_label: str,
                require_time: bool = True) -> str:
    lines = [
        f"# Starter policy for {source_label}.",
        "# Every signal is protected by one simple guarantee. Edit the hard section to say what must survive,",
        "# for example threshold_crossing for limit crossings, violation for limit exceedances and",
        "# window_extrema for extrema per time window. docs/policy.md lists every option.",
        "#",
        "# Signals found:",
        *_signal_table(infos),
    ]
    missing_unit = [info.name for info in infos if info.unit is None and info.kind != "discrete"]
    if missing_unit:
        example = missing_unit[0]
        alias = example.rsplit("/", 1)[-1]
        lines += [
            "#",
            f"# {len(missing_unit)} signal(s) have no unit in the source. Limits with units (65 kPa, 7 deg) need one;",
            "# declare it under signals.decl, replacing <unit> with the signal's real unit:",
            "#   signals:",
            "#     decl:",
            f"#       {_scalar(alias)}: {{path: {_scalar(example)}, unit: <unit>}}",
        ]
    untimed = [info.name for info in infos if info.time_ref is None] if require_time else []
    if untimed:
        lines += [
            "#",
            f"# Excluded because no time signal was found: {', '.join(untimed)}.",
            "# Set signals.time to the name of the clock, or give each one a time under signals.decl, then remove",
            "# them from signals.exclude.",
        ]
    lines.append("")
    lines.extend(_yaml(policy, 0))
    return "\n".join(lines) + "\n"


def render_json(policy: Mapping) -> str:
    return json.dumps(policy, indent=2, ensure_ascii=False) + "\n"
