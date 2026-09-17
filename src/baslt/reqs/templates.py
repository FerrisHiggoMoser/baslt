"""Starter requirement workbooks and mapping files.

`generic` is a complete example for `examples/rocket_sim.py`: every check type, cases, conditions, events, a curve
limit and a derived signal, with its mapping written as config sheets in the same workbook (or as a mapping file
next to a CSV). `polarion` lays the same idea out as a Polarion round-trip export with custom verification fields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..errors import UsageError

__all__ = ["TEMPLATES", "init_template", "mapping_text", "template_mapping", "template_rows"]

TEMPLATES = ("generic", "polarion")

GENERIC_HEADER = ["ID", "Title", "Check", "Type", "Limit", "Unit", "When", "Applies to", "Tolerance", "Warn margin",
                  "Count", "Case", "Severity", "Status", "Notes"]

# ID, Title, Check, Type, Limit, Unit, When, Applies to, Tolerance, Warn margin, Count, Case, Severity, Status, Notes
GENERIC_ROWS: list[list[str]] = [
    ["", "Structural loads", "", "", "", "", "", "", "", "", "", "", "", "", "A heading row: no status, skipped"],
    ["LV-001", "Max dynamic pressure", "q", "upper", "70", "kPa", "ascent", "", "", "2 kPa", "", "", "major",
     "approved", ""],
    ["LV-002", "AoA at high q", "alpha", "range", "[-7, 7]", "deg", "high_q", "", "100 ms", "", "", "", "major",
     "approved", "Short spikes are tolerated and listed"],
    ["LV-003", "Axial load in load relief", "load", "upper", "5", "g0", "load_relief", "", "100 ms", "", "",
     "load relief", "critical", "approved", "Cases: the first matching row applies"],
    ["LV-003", "Axial load", "load", "upper", "6", "g0", "ascent", "", "100 ms", "", "", "powered", "critical",
     "approved", ""],
    ["LV-004", "q against the Mach envelope", "q", "upper", "curve(q_max_vs_mach, mach)", "", "ascent", "", "",
     "1 kPa", "", "", "major", "approved", "A limit that is a curve over another signal"],
    ["LV-005", "Altitude reached", "alt", "max", ">= 40 km", "", "", "", "", "", "", "", "major", "approved",
     "Aggregate: the maximum over the run"],
    ["LV-006", "MECO time", "MECO", "event", "99 .. 102", "s", "", "", "", "", "1", "", "critical", "approved",
     "Exactly one MECO, inside the window"],
    ["LV-007", "Velocity at MECO", "at('MECO', speed)", "value", ">= 1.8 km/s", "", "", "", "", "", "", "",
     "critical", "approved", ""],
    ["LV-008", "Time above 60 kPa", "q > 60 kPa", "duration", "<= 40 s", "", "ascent", "", "", "", "", "", "minor",
     "approved", ""],
    ["LV-009", "Total AoA exceedance", "abs(alpha) > 7 deg", "duration", "<= 0.5 s", "", "", "", "", "", "", "",
     "major", "approved", ""],
    ["", "Thermal", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    ["LV-010", "Skin temperature", "skin_temp", "upper", "420 K", "", "ascent", "", "1 s", "", "", "", "major",
     "approved", "A data gap inside the window gives WARN"],
    ["LV-011", "Elevon command range", "elevon", "range", "±20", "deg", "after('liftoff', 1 s)", "", "", "", "", "",
     "minor", "approved", ""],
    ["LV-012", "Coast mode after MECO", "mode == 'COAST'", "assert", "", "", "after('MECO')", "", "", "", "", "",
     "major", "approved", ""],
    ["LV-013", "Mean thrust in powered flight", "mean(thrust)", "value", ">= 2.1 MN", "", "ascent", "", "", "", "",
     "", "minor", "approved", ""],
    ["LV-014", "Pitch program start", "time('pitch_start')", "value", "6 ± 1", "s", "", "", "", "", "", "",
     "minor", "approved", ""],
    ["LV-015", "Load relief duration", "load_relief", "duration", ">= 15 s", "", "", "", "", "", "", "", "minor",
     "approved", ""],
    ["LV-016", "Altitude at end of run", "alt", "final", ">= 40 km", "", "", "", "", "", "", "", "minor",
     "approved", ""],
    ["LV-017", "Structural modes", "", "", "", "", "", "", "", "", "", "", "", "approved",
     "No check yet: reported as not covered"],
    ["LV-018", "Draft limit", "q", "upper", "", "kPa", "", "", "", "", "", "", "", "draft",
     "Draft rows are skipped until approved"],
]

SIGNALS_HEADER = ["Alias", "Path", "Unit", "Kind", "Labels", "Expression"]
GENERIC_SIGNALS = [
    ["q", "aero/q", "Pa", "", "", ""],
    ["alpha", "aero/alpha", "deg", "", "", ""],
    ["thrust", "prop/thrust", "N", "", "", ""],
    ["skin_temp", "thermal/skin_temp", "K", "", "", ""],
    ["elevon", "gnc/elevon_cmd", "deg", "", "", ""],
    ["mode", "gnc/mode", "", "discrete", "0=VERTICAL_RISE; 1=PITCH_PROGRAM; 2=LOAD_RELIEF; 3=COAST", ""],
    ["alt", "", "m", "", "", "`nav/position`.z"],
    ["speed", "", "m/s", "", "", "norm(`nav/velocity`)"],
    ["load", "", "m/s^2", "", "", "norm(deriv(movmean(`nav/velocity`, 200 ms)))"],
    ["mach", "", "1", "", "", "speed / 340[m/s]"],
]
EVENTS_HEADER = ["Name", "Signal", "Condition", "Value", "When", "Debounce", "Occurrence"]
GENERIC_EVENTS = [
    ["liftoff", "", "", "", "alt > 1 m", "100 ms", "first"],
    ["pitch_start", "mode", "equals", "PITCH_PROGRAM", "", "", "first"],
    ["MECO", "thrust", "falls_below", "1 MN", "", "250 ms", "first"],
]
CONDITIONS_HEADER = ["Name", "Expression"]
GENERIC_CONDITIONS = [
    ["ascent", "between('liftoff', 'MECO')"],
    ["coast", "after('MECO')"],
    ["load_relief", "mode == 'LOAD_RELIEF'"],
    ["high_q", "q > 20 kPa"],
]
CURVES_HEADER = ["Name", "X", "Y", "X unit", "Y unit", "Outside", "Mode", "Argument"]
GENERIC_CURVES = [
    ["q_max_vs_mach", "0", "72", "1", "kPa", "clamp", "linear", "mach"],
    ["q_max_vs_mach", "1.2", "70", "", "", "", "", ""],
    ["q_max_vs_mach", "3", "69", "", "", "", "", ""],
]
UNITS_HEADER = ["Symbol", "Definition"]
GENERIC_UNITS = [["gee", "9.80665 m/s^2"]]
SETTINGS_HEADER = ["Key", "Value"]
GENERIC_SETTINGS = [
    ["requirements.sheet", "Requirements"],
    ["requirements.where.Status", "approved"],
    ["requirements.passthrough", "Status, Notes"],
    ["time.t0", "liftoff"],
    ["report.title", "LV-3 ascent"],
]

POLARION_HEADER = ["ID", "Title", "Type", "Status", "Description", "Verification Check", "Verification Type",
                   "Verification Limit", "Verification Unit", "Verification Condition",
                   "Verification Applicability", "Verification Tolerance", "Verification Case", "Verification Result",
                   "Verification Evidence"]
POLARION_ROWS = [
    ["LV-100", "Ascent loads", "Heading", "", "", "", "", "", "", "", "", "", "", "", ""],
    ["LV-101", "Max dynamic pressure", "Requirement", "approved",
     "The vehicle shall not exceed 70 kPa dynamic pressure during ascent.", "q", "upper", "70", "kPa", "ascent",
     "", "", "", "", ""],
    ["LV-102", "Angle of attack at high q", "Requirement", "approved",
     "The angle of attack shall stay within ±7° while q exceeds 20 kPa.", "alpha", "range", "[-7, 7]", "deg",
     "high_q", "", "100 ms", "", "", ""],
    ["LV-103", "MECO time", "Requirement", "approved", "Main engine cut-off shall occur between 99 s and 102 s.",
     "MECO", "event", "99 .. 102", "s", "", "", "", "", "", ""],
    ["LV-104", "Structural modes", "Requirement", "approved", "Structural modes shall be damped.", "", "", "", "",
     "", "", "", "", "", ""],
    ["LV-105", "Reentry heating", "Requirement", "draft", "Not yet reviewed.", "", "", "", "", "", "", "", "", "",
     ""],
]
POLARION_SETTINGS = [
    ["requirements.sheet", "Requirements"],
    ["requirements.where.Type", "Requirement"],
    ["requirements.where.Status", "approved"],
    ["requirements.columns.title", "Title"],
    ["requirements.columns.check", "Verification Check"],
    ["requirements.columns.kind", "Verification Type"],
    ["requirements.columns.limit", "Verification Limit"],
    ["requirements.columns.unit", "Verification Unit"],
    ["requirements.columns.when", "Verification Condition"],
    ["requirements.columns.applies_to", "Verification Applicability"],
    ["requirements.columns.tolerance", "Verification Tolerance"],
    ["requirements.columns.case", "Verification Case"],
    ["requirements.write_back.verdict", "Verification Result"],
    ["requirements.write_back.evidence", "Verification Evidence"],
    ["requirements.passthrough", "Status"],
    ["time.t0", "liftoff"],
]

GUIDE_ROWS = [
    ["Column", "What to write", "Examples"],
    ["ID", "The requirement ID. Rows sharing an ID are cases of one requirement; the first case whose When and "
           "Applies to hold decides the limit at each moment.", "LV-001"],
    ["Check", "What is checked: a signal alias, a signal path in backticks, or an expression.",
     "q · `aero/q` · abs(alpha) · norm(`nav/velocity`) · at('MECO', speed) · q > 60 kPa"],
    ["Type", "upper, lower, range: a limit on every sample while When holds. assert: the Check (a condition) must "
             "hold. duration: the time the Check (a condition) holds. value: one number. event: an event's time "
             "and count. max, min, initial, final, mean, rms, integral: that number of the Check while When "
             "holds.", "upper · range · max · duration"],
    ["Limit", "A number with its unit, a comparison, a range, a curve or an expression. A bare number takes its "
              "direction from Type.",
     "<= 70 kPa · [-7, 7] deg · 99 .. 102 s · 6 ± 1 s · curve(q_max_vs_mach, mach) · = 0.9 * param.q_design"],
    ["Unit", "The unit of the limit when the Limit cell has none.", "kPa · deg · g0"],
    ["When", "A condition: a named condition, an expression, or event windows.",
     "ascent · q > 20 kPa · mode == 'LOAD_RELIEF' · after('MECO') · between('liftoff', 'MECO')"],
    ["Applies to", "A condition on run parameters (batch checks with --params).", "payload == 'heavy'"],
    ["Tolerance", "How long a violation may last before it counts, or how many samples.", "100 ms · 3 samples"],
    ["Warn margin", "A band inside the limit that gives WARN.", "2 kPa · 5 %"],
    ["Count", "For event checks: how many times the event must happen.", "1"],
    ["Case", "A name for the case, shown in results.", "heavy payload"],
    ["", "", ""],
    ["Functions", "abs, min, max, clip, where, sign, hypot, norm, sqrt, exp, log, sin, cos, tan, deriv, movmean, "
                  "as_unit, curve · events: after, before, between, during, since, time, count, at · aggregates: "
                  "max, min, initial, final, mean, rms, integral, duration", ""],
    ["Sheets", "Signals: aliases and derived signals. Events: named events. Conditions: named conditions. Curves: "
               "limit curves. Units: extra units. Settings: layout (which columns, which rows, write-back).", ""],
]


def template_rows(template: str) -> tuple[list[str], list[list[str]]]:
    if template == "generic":
        return GENERIC_HEADER, GENERIC_ROWS
    if template == "polarion":
        return POLARION_HEADER, POLARION_ROWS
    raise UsageError(f"unknown template {template!r}; templates are {', '.join(TEMPLATES)}")


def _config_sheets(template: str, signals: Sequence[Sequence[str]], examples: bool
                   ) -> dict[str, tuple[list[str], list[list[str]]]]:
    settings = GENERIC_SETTINGS if template == "generic" else POLARION_SETTINGS
    if not examples:
        settings = [row for row in settings if row[0] not in ("time.t0", "report.title")]
    return {
        "Signals": (SIGNALS_HEADER, [list(row) for row in signals]),
        "Events": (EVENTS_HEADER, GENERIC_EVENTS if examples else []),
        "Conditions": (CONDITIONS_HEADER, GENERIC_CONDITIONS if examples else []),
        "Curves": (CURVES_HEADER, GENERIC_CURVES if examples else []),
        "Units": (UNITS_HEADER, GENERIC_UNITS),
        "Settings": (SETTINGS_HEADER, settings),
    }


def template_mapping(template: str, signals: Sequence[Sequence[str]] | None = None, *, examples: bool = True) -> dict:
    """The template's config sheets as a mapping (for a YAML or JSON mapping file)."""
    rows = signals if signals is not None else GENERIC_SIGNALS
    data: dict = {"version": 1, "name": "LV-3 ascent requirements" if examples else "Requirements"}
    for key, value in _config_sheets(template, rows, examples)["Settings"][1]:
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        if parts[-1] in ("passthrough",) or parts[-2:-1] == ["where"]:
            target[parts[-1]] = [v.strip() for v in value.split(",")]
        else:
            target[parts[-1]] = value
    signals_map: dict = {}
    for alias, path, unit, kind, labels, expr in rows:
        entry: dict = {}
        if path:
            entry["path"] = path
        if expr:
            entry["expr"] = expr
        if unit:
            entry["unit"] = unit
        if kind:
            entry["kind"] = kind
        if labels:
            entry["labels"] = {int(k.strip()): v.strip() for k, v in (part.split("=") for part in labels.split(";"))}
        signals_map[alias] = entry
    data["signals"] = signals_map
    if not examples:
        data["units"] = {symbol: definition for symbol, definition in GENERIC_UNITS}
        return data
    events: dict = {}
    for name, signal, condition, value, when, debounce, occurrence in GENERIC_EVENTS:
        entry = {}
        if signal:
            entry["signal"] = signal
            entry[condition] = value
        if when:
            entry["when"] = when
        if debounce:
            entry["debounce"] = debounce
        if occurrence != "first":
            entry["occurrence"] = occurrence
        events[name] = entry
    data["events"] = events
    data["conditions"] = {name: text for name, text in GENERIC_CONDITIONS}
    curves: dict = {}
    for name, x, y, x_unit, y_unit, outside, mode, arg in GENERIC_CURVES:
        entry = curves.setdefault(name, {"x_unit": x_unit, "y_unit": y_unit, "x": arg, "outside": outside,
                                         "mode": mode, "points": []})
        entry["points"].append([float(x), float(y)])
    data["curves"] = curves
    data["units"] = {symbol: definition for symbol, definition in GENERIC_UNITS}
    return data


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    text = str(value)
    safe = text and all(ch.isalnum() or ch in "_-./ " for ch in text) and not text[0].isdigit() and \
        text.lower() not in ("yes", "no", "true", "false", "null", "on", "off", "y", "n") and text == text.strip()
    return text if safe else json.dumps(text, ensure_ascii=False)


def _yaml_lines(value: object, indent: int) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    for key, child in value.items():  # type: ignore[union-attr]
        head = f"{pad}{_yaml_scalar(key)}:"
        if isinstance(child, Mapping) and child:
            lines.append(head)
            lines.extend(_yaml_lines(child, indent + 1))
        elif isinstance(child, Mapping):
            lines.append(f"{head} {{}}")
        elif isinstance(child, (list, tuple)):
            items = []
            for item in child:
                if isinstance(item, (list, tuple)):
                    items.append("[" + ", ".join(_yaml_scalar(v) for v in item) + "]")
                else:
                    items.append(_yaml_scalar(item))
            lines.append(f"{head} [{', '.join(items)}]")
        else:
            lines.append(f"{head} {_yaml_scalar(child)}")
    return lines


MAPPING_HEADER = """\
# Requirement check mapping (baslt check -m this-file).
# requirements: where the requirements are and which rows count (see docs/requirements.md)
# signals:      aliases for signals in the run, with units, labels for integer modes, and derived signals
# events:       named events used by after(), before(), between(), time(), at(), count()
# conditions:   named conditions usable in When columns
# curves:       limit curves usable as curve(name, argument)
"""


def mapping_text(data: Mapping, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    return MAPPING_HEADER + "\n".join(_yaml_lines(data, 0)) + "\n"


def _signals_from_source(source) -> list[list[str]]:
    from ..sources import open_source

    adapter = open_source(source)
    rows = []
    leaves: dict[str, int] = {}
    infos = adapter.list_signals()
    for info in infos:
        leaf = info.name.rsplit("/", 1)[-1]
        leaves[leaf] = leaves.get(leaf, 0) + 1
    for info in infos:
        leaf = info.name.rsplit("/", 1)[-1]
        alias = leaf if leaves[leaf] == 1 else info.name.replace("/", "_")
        alias = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in alias)
        if not alias or alias[0].isdigit():
            alias = f"s_{alias}"
        rows.append([alias, info.name, info.unit or "", info.kind if info.kind != "continuous" else "", "", ""])
    return rows


def init_template(output: str | Path, *, template: str = "generic", mapping_output: str | Path | None = None,
                  source=None, force: bool = False) -> dict:
    """Write a starter requirements file (xlsx with config sheets, or csv with a mapping file)."""
    from .._io import write_atomic
    from ..tabular import Sheet, Styled, write_xlsx

    header, rows = template_rows(template)
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix not in (".xlsx", ".csv"):
        raise UsageError("the requirements template is written as .xlsx or .csv")
    examples = source is None
    signals = GENERIC_SIGNALS if examples else _signals_from_source(source)
    if not examples:
        # The example rows name the rocket example's signals; keep them as a guide, outside the check.
        status = header.index("Status")
        notes = header.index("Notes") if "Notes" in header else None
        rows = [list(row) for row in rows]
        for row in rows:
            if row[0]:
                row[status] = "example"
                if notes is not None:
                    row[notes] = "Example: point Check at your signals (Signals sheet) and set Status to approved"
    written: list[Path] = []

    def guard(path: Path) -> None:
        if path.exists() and not force:
            raise UsageError(f"{path} already exists; use --force (force=True) to replace it")

    guard(output)
    if suffix == ".xlsx":
        sheets = [Sheet("Requirements", [[Styled(h, "header") for h in header], *rows])]
        for name, (sheet_header, sheet_rows) in _config_sheets(template, signals, examples).items():
            sheets.append(Sheet(name, [[Styled(h, "header") for h in sheet_header], *sheet_rows]))
        sheets.append(Sheet("Guide", [[Styled(h, "header") for h in GUIDE_ROWS[0]], *GUIDE_ROWS[1:]],
                            widths=[16, 90, 70], autofilter=False))
        write_xlsx(output, sheets, title="Requirements")
        written.append(output)
        mapping_path = None
        if mapping_output is not None:
            mapping_path = Path(mapping_output)
    else:
        import csv
        import io

        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        write_atomic(output, buffer.getvalue().encode("utf-8"))
        written.append(output)
        if mapping_output is None:
            import importlib.util

            has_yaml = importlib.util.find_spec("yaml") is not None
            mapping_output = output.with_suffix(".mapping.yaml" if has_yaml else ".mapping.json")
        mapping_path = Path(mapping_output)
    if mapping_path is not None:
        fmt = "json" if mapping_path.suffix.lower() == ".json" else "yaml"
        guard(mapping_path)
        data = template_mapping(template, signals, examples=examples)
        if suffix == ".csv":
            data.setdefault("requirements", {}).pop("sheet", None)
        write_atomic(mapping_path, mapping_text(data, fmt).encode("utf-8"))
        written.append(mapping_path)
    return {
        "status": "pass",
        "template": template,
        "examples_active": examples,
        "requirements": str(output),
        "mapping": str(mapping_path) if mapping_path is not None else None,
        "rows": len(rows),
        "signals": len(signals),
        "written": [str(path) for path in written],
    }
