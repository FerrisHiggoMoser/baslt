"""The mapping: how a requirements table is laid out and what its names mean.

A mapping comes from a YAML or JSON file, from config sheets inside the requirements workbook (Signals, Events,
Conditions, Curves, Units, Settings), or both. Sources merge in this order, later ones winning key by key:

    built-in defaults  <  workbook config sheets  <  mapping file  <  overrides passed by the caller

Every problem is reported with its location (`map.yaml:12:5`, `reqs.xlsx:Signals!C4`) and all of them at once.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import RequirementsError, UsageError
from ..policy.validate import IssueCollector, item_path, join_path, suggest
from ..units import UnitDef
from .units_ext import UnitsError, UnitTable

__all__ = [
    "CONFIG_SHEETS",
    "FIELDS",
    "RESULT_FIELDS",
    "ChecksJoin",
    "Config",
    "CurveDef",
    "Defaults",
    "EventDef",
    "Layout",
    "ParamsSpec",
    "ReportSpec",
    "SignalAlias",
    "TimeSpec",
    "load_config",
]

FIELDS: tuple[str, ...] = (
    "id", "title", "check", "kind", "limit", "lower", "upper", "unit", "when", "applies_to", "tolerance", "margin",
    "case", "count", "severity", "grid", "notes",
)
RESULT_FIELDS: tuple[str, ...] = ("verdict", "value", "limit", "margin", "margin_pct", "at", "evidence", "runs")
CONFIG_SHEETS: tuple[str, ...] = ("Signals", "Events", "Conditions", "Curves", "Units", "Settings")
TOP_KEYS = ("version", "name", "requirements", "checks", "time", "signals", "units", "events", "conditions",
            "curves", "params", "defaults", "report", "archive")
KINDS = ("continuous", "discrete", "vector")
EVENT_CONDITIONS = ("falls_below", "rises_above", "equals")
OCCURRENCE_RE = re.compile(r"^(first|last|all|\d+)$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
EVENT_NAME_RE = re.compile(r"^[^\s'\"#`()]+$")

# Header texts recognized without configuration, after lowercasing and dropping spaces and punctuation.
DEFAULT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "id": ("id", "reqid", "requirementid", "requirement", "identifier", "key", "number", "no", "reqno"),
    "title": ("title", "name", "summary", "description", "requirementtitle", "text"),
    "check": ("check", "signal", "expression", "verificationcheck", "measure", "parameter", "quantity"),
    "kind": ("type", "kind", "checktype", "limittype", "verificationkind", "checkkind"),
    "limit": ("limit", "requirementlimit", "verificationlimit", "threshold", "criterion", "criteria", "value"),
    "lower": ("min", "minimum", "lower", "lowerlimit", "low", "lsl"),
    "upper": ("max", "maximum", "upper", "upperlimit", "high", "usl"),
    "unit": ("unit", "units", "uom"),
    "when": ("when", "condition", "phase", "during", "verificationcondition", "applieswhen"),
    "applies_to": ("appliesto", "applicability", "configuration", "config", "variant",
                   "verificationapplicability"),
    "tolerance": ("tolerance", "persistence", "duration", "allowedduration", "verificationtolerance", "debounce"),
    "margin": ("margin", "warnmargin", "warningmargin", "verificationmargin"),
    "case": ("case", "subcase", "verificationcase"),
    "count": ("count", "expectedcount", "occurrences"),
    "severity": ("severity", "criticality", "priority", "category"),
    "grid": ("grid", "timebase"),
    "notes": ("notes", "note", "comment", "comments", "remarks", "rationale"),
}


def normalize_header(text: object) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").casefold())


@dataclass(slots=True)
class Layout:
    sheet: str | int | None = None
    header_row: int | None = None  # None: detect
    where: dict[str, list[str]] = field(default_factory=dict)
    columns: dict[str, list[str]] = field(default_factory=dict)
    passthrough: list[str] = field(default_factory=list)
    vocab: dict[str, dict[str, str]] = field(default_factory=dict)
    decimal_comma: bool = False
    write_back: dict[str, str] = field(default_factory=dict)
    encoding: str | None = None
    delimiter: str | None = None


@dataclass(slots=True)
class ChecksJoin:
    file: str
    sheet: str | int | None = None
    key: str | None = None
    header_row: int | None = None


@dataclass(slots=True)
class TimeSpec:
    signal: str | None = None
    unit: str | None = None
    on_non_monotonic: str = "error"
    t0: str | None = None


@dataclass(slots=True)
class SignalAlias:
    name: str
    path: str | None = None
    unit: str | None = None
    kind: str | None = None
    labels: dict[int, str] | None = None
    expr: str | None = None
    time: str | None = None
    loc: str | None = None


@dataclass(slots=True)
class EventDef:
    name: str
    signal: str | None = None
    condition: str | None = None
    value: object = None
    when: str | None = None
    param: str | None = None
    hysteresis: str | None = None
    debounce: str | None = None
    occurrence: str = "first"
    loc: str | None = None


@dataclass(slots=True)
class CurveDef:
    name: str
    points: list[tuple[float, float]]
    x_unit: str | None = None
    y_unit: str | None = None
    x: str | None = None  # the default argument (an alias or expression)
    outside: str = "clamp"
    mode: str = "linear"
    loc: str | None = None


@dataclass(slots=True)
class ParamsSpec:
    key: str | None = None
    file: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    from_source: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Defaults:
    tolerance: str = "0 s"
    margin: str | None = None
    grid: str = "first"
    on_gap: str = "warn"
    on_missing_event: str = "warn"
    on_case_overlap: str = "first"
    on_tolerated: str = "pass"


@dataclass(slots=True)
class ReportSpec:
    title: str | None = None
    pages: str = "failed"
    plot_bins: int = 512
    overlays: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Config:
    version: int = 1
    name: str | None = None
    requirements: Layout = field(default_factory=Layout)
    checks: ChecksJoin | None = None
    time: TimeSpec = field(default_factory=TimeSpec)
    signals: dict[str, SignalAlias] = field(default_factory=dict)
    units: UnitTable = field(default_factory=UnitTable)
    unit_defs: dict[str, UnitDef] = field(default_factory=dict)
    events: dict[str, EventDef] = field(default_factory=dict)
    conditions: dict[str, str] = field(default_factory=dict)
    condition_locs: dict[str, str] = field(default_factory=dict)
    curves: dict[str, CurveDef] = field(default_factory=dict)
    params: ParamsSpec = field(default_factory=ParamsSpec)
    defaults: Defaults = field(default_factory=Defaults)
    report: ReportSpec = field(default_factory=ReportSpec)
    archive_max_size: str | None = None
    sources: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    locations: dict[str, str] = field(default_factory=dict)
    base: Path | None = None  # the folder relative file names are resolved against

    @property
    def sha256(self) -> str:
        text = json.dumps(self.data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def location(self, path: str) -> str | None:
        from ..policy.validate import find_location

        return find_location(self.locations, path)


# ------------------------------------------------------------------------------------------------------------
# Sources


def deep_merge(base: Mapping, update: Mapping) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_mapping_file(path: Path) -> tuple[dict, dict[str, str]]:
    from ..policy.parse import parse_json, parse_yaml

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return config_from_workbook(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UsageError(f"mapping file not found: {path}") from None
    except UnicodeDecodeError as exc:
        raise RequirementsError(f"mapping file {path} is not valid UTF-8: {exc}") from None
    if suffix in (".yaml", ".yml"):
        data, locations = parse_yaml(text, path.name)
    elif suffix == ".json":
        data, locations = parse_json(text, path.name), {}
    else:
        raise UsageError(f"cannot tell the mapping format of {path}; use a .yaml, .yml, .json or .xlsx file")
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise RequirementsError(f"{path.name}: a mapping file must hold a mapping at the top, not {type(data).__name__}")
    return dict(data), locations


def _sheet_rows(table) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    """Header names (normalized) and data rows of a config sheet; the header is the first non-empty row."""
    header_row = next((r for r in range(1, table.n_rows + 1) if any(t.strip() for t in table.row_texts(r))), None)
    if header_row is None:
        return [], []
    header = [normalize_header(t) for t in table.row_texts(header_row)]
    rows = []
    for r in range(header_row + 1, table.n_rows + 1):
        texts = table.row_texts(r)
        if not any(t.strip() for t in texts):
            continue
        rows.append((r, {name: texts[c].strip() for c, name in enumerate(header) if c < len(texts) and name}))
    return header, rows


def _split_list(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;\n]", text) if part.strip()]


def _labels_text(text: str) -> dict[str, str]:
    out = {}
    for part in re.split(r"[;,\n]", text):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep:
            key, sep, value = part.partition(":")
        out[key.strip()] = value.strip()
    return out


def config_from_workbook(path: Path) -> tuple[dict, dict[str, str]]:
    """The mapping written as config sheets inside a workbook (sheets that are absent are skipped)."""
    from ..tabular import list_sheets, read_table

    present = {sheet.name.casefold(): sheet.name for sheet in list_sheets(path)}
    data: dict = {}
    locations: dict[str, str] = {}

    def table(name: str):
        actual = present.get(name.casefold())
        return None if actual is None else read_table(path, sheet=actual)

    def loc(t, row: int) -> str:
        return t.location(row)

    signals = table("Signals")
    if signals is not None:
        _, rows = _sheet_rows(signals)
        for r, row in rows:
            alias = row.get("alias") or row.get("name")
            if not alias:
                continue
            entry: dict = {}
            for key, names in (("path", ("path", "signal", "source")), ("unit", ("unit", "units")),
                               ("kind", ("kind", "type")), ("expr", ("expression", "expr", "derived")),
                               ("time", ("time", "clock"))):
                value = next((row[n] for n in names if row.get(n)), "")
                if value:
                    entry[key] = value
            if row.get("labels"):
                entry["labels"] = _labels_text(row["labels"])
            data.setdefault("signals", {})[alias] = entry
            locations[f"signals.{alias}"] = loc(signals, r)

    events = table("Events")
    if events is not None:
        _, rows = _sheet_rows(events)
        for r, row in rows:
            name = row.get("name") or row.get("event")
            if not name:
                continue
            entry = {}
            condition = row.get("condition", "").strip().lower().replace(" ", "_")
            if row.get("signal"):
                entry["signal"] = row["signal"]
            if condition:
                entry[condition] = row.get("value", "")
            for key, names in (("when", ("when", "expression")), ("param", ("parameter", "param")),
                               ("hysteresis", ("hysteresis",)), ("debounce", ("debounce",)),
                               ("occurrence", ("occurrence",))):
                value = next((row[n] for n in names if row.get(n)), "")
                if value:
                    entry[key] = value
            data.setdefault("events", {})[name] = entry
            locations[f"events.{name}"] = loc(events, r)

    conditions = table("Conditions")
    if conditions is not None:
        _, rows = _sheet_rows(conditions)
        for r, row in rows:
            name = row.get("name") or row.get("condition") or row.get("phase")
            text = row.get("expression") or row.get("when") or row.get("definition")
            if name and text:
                data.setdefault("conditions", {})[name] = text
                locations[f"conditions.{name}"] = loc(conditions, r)

    curves = table("Curves")
    if curves is not None:
        _, rows = _sheet_rows(curves)
        for r, row in rows:
            name = row.get("name") or row.get("curve")
            if not name:
                continue
            entry = data.setdefault("curves", {}).setdefault(name, {"points": []})
            locations.setdefault(f"curves.{name}", loc(curves, r))
            entry["points"].append([row.get("x", ""), row.get("y", "")])
            for key, names in (("x_unit", ("xunit",)), ("y_unit", ("yunit",)), ("outside", ("outside",)),
                               ("mode", ("mode",)), ("x", ("argument", "arg"))):
                value = next((row[n] for n in names if row.get(n)), "")
                if value and key not in entry:
                    entry[key] = value

    units = table("Units")
    if units is not None:
        _, rows = _sheet_rows(units)
        for r, row in rows:
            symbol = row.get("symbol") or row.get("unit")
            if not symbol:
                continue
            if row.get("definition") or row.get("equals"):
                data.setdefault("units", {})[symbol] = row.get("definition") or row.get("equals")
            else:
                data.setdefault("units", {})[symbol] = {
                    key: row[key] for key in ("dimension", "factor", "offset") if row.get(key)
                }
            locations[f"units.{symbol}"] = loc(units, r)

    settings = table("Settings")
    if settings is not None:
        _, rows = _sheet_rows(settings)
        for r, row in rows:
            key = row.get("key") or row.get("setting")
            value = row.get("value", "")
            if not key:
                continue
            parts = [part.strip() for part in key.split(".") if part.strip()]
            if not parts:
                continue
            target = data
            for part in parts[:-1]:
                node = target.setdefault(part, {})
                if not isinstance(node, dict):
                    node = target[part] = {}
                target = node
            target[parts[-1]] = value
            locations[".".join(parts)] = loc(settings, r)
    return data, locations


def load_config(mapping: str | Path | Mapping | None = None, *, workbook: str | Path | None = None,
                overrides: Mapping | None = None) -> Config:
    """Build the Config from its sources; raises RequirementsError listing every problem."""
    data: dict = {}
    locations: dict[str, str] = {}
    sources: list[str] = []
    base: Path | None = None
    if workbook is not None and Path(workbook).suffix.lower() in (".xlsx", ".xlsm") and Path(workbook).exists():
        sheet_data, sheet_locations = config_from_workbook(Path(workbook))
        if sheet_data:
            data = deep_merge(data, sheet_data)
            locations.update(sheet_locations)
            sources.append(f"{Path(workbook).name} (config sheets)")
        base = Path(workbook).resolve().parent
    if mapping is not None:
        if isinstance(mapping, Mapping):
            data = deep_merge(data, mapping)
            sources.append("mapping")
        else:
            path = Path(mapping)
            file_data, file_locations = _read_mapping_file(path)
            data = deep_merge(data, file_data)
            locations.update(file_locations)
            sources.append(path.name)
            base = path.resolve().parent
    if overrides:
        data = deep_merge(data, overrides)
    config = _Parser(locations).parse(data)
    config.sources = sources
    config.data = data
    config.locations = locations
    config.base = base
    return config


# ------------------------------------------------------------------------------------------------------------
# Validation


class _Parser:
    def __init__(self, locations: Mapping[str, str]) -> None:
        self.issues = IssueCollector(locations, limit=10_000)

    def fail(self) -> None:
        if self.issues.issues:
            raise RequirementsError(self.issues.issues)

    # ----- scalars -------------------------------------------------------------------------------------------

    def keys(self, mapping: object, allowed: Sequence[str], path: str) -> dict:
        if mapping is None:
            return {}
        if not isinstance(mapping, Mapping):
            self.issues.add(path, f"expected a mapping, got {type(mapping).__name__}")
            return {}
        for key in mapping:
            if key not in allowed:
                hint = suggest(str(key), allowed)
                self.issues.add(join_path(path, key), f"unknown key {key!r}" + (f"; did you mean {hint!r}?" if hint else ""))
        return dict(mapping)

    def text(self, value: object, path: str, *, required: bool = False) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            if required:
                self.issues.add(path, "a value is required")
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            self.issues.add(path, f"expected text, got {type(value).__name__}")
            return None
        return str(value).strip()

    def choice(self, value: object, choices: Sequence[str], path: str, default: str) -> str:
        text = self.text(value, path)
        if text is None:
            return default
        lowered = text.lower()
        if lowered not in choices:
            self.issues.add(path, f"expected one of {', '.join(choices)}, got {text!r}")
            return default
        return lowered

    def boolean(self, value: object, path: str, default: bool = False) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        self.issues.add(path, f"expected true or false, got {value!r}")
        return default

    def integer(self, value: object, path: str, *, minimum: int = 0, maximum: int | None = None) -> int | None:
        if value is None or value == "":
            return None
        try:
            number = int(str(value).strip()) if not isinstance(value, bool) else None
        except ValueError:
            number = None
        if number is None or number < minimum or (maximum is not None and number > maximum):
            bound = f" between {minimum} and {maximum}" if maximum is not None else f" of at least {minimum}"
            self.issues.add(path, f"expected a whole number{bound}, got {value!r}")
            return None
        return number

    def text_list(self, value: object, path: str) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return _split_list(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return [str(value)]
        if isinstance(value, Sequence):
            out = []
            for i, item in enumerate(value):
                text = self.text(item, item_path(path, i))
                if text is not None:
                    out.append(text)
            return out
        self.issues.add(path, f"expected a list, got {type(value).__name__}")
        return []

    def name(self, key: object, path: str, pattern: re.Pattern, what: str) -> str | None:
        if not isinstance(key, str) or not pattern.match(key):
            self.issues.add(path, f"{what} {key!r} must be a name made of letters, digits and underscores")
            return None
        return key

    # ----- sections ------------------------------------------------------------------------------------------

    def parse(self, data: object) -> Config:
        root = self.keys(data, TOP_KEYS, "")
        config = Config()
        version = root.get("version", 1)
        if str(version).strip() not in ("1", "1.0"):
            self.issues.add("version", f"this build reads mapping version 1, got {version!r}")
        config.name = self.text(root.get("name"), "name")
        unit_defs = self.units(root.get("units"), "units")
        config.unit_defs = unit_defs
        config.units = UnitTable(unit_defs)
        config.requirements = self.layout(root.get("requirements"), "requirements")
        config.checks = self.checks(root.get("checks"), "checks")
        config.time = self.time(root.get("time"), "time")
        config.signals = self.signals(root.get("signals"), "signals", config.units)
        config.events = self.events(root.get("events"), "events", config.units)
        config.conditions = self.conditions(root.get("conditions"), "conditions")
        config.curves = self.curves(root.get("curves"), "curves", config.units)
        config.params = self.params(root.get("params"), "params", config.units)
        config.defaults = self.defaults(root.get("defaults"), "defaults", config.units)
        config.report = self.report(root.get("report"), "report")
        archive = self.keys(root.get("archive"), ("max_size",), "archive")
        config.archive_max_size = self.text(archive.get("max_size"), "archive.max_size")
        clashes = (set(config.signals) & set(config.conditions))
        for name in sorted(clashes):
            self.issues.add(f"conditions.{name}", f"{name!r} is both a signal alias and a condition")
        self.fail()
        return config

    def layout(self, value: object, path: str) -> Layout:
        spec = self.keys(value, ("sheet", "header_row", "where", "columns", "passthrough", "vocab",
                                 "decimal_comma", "write_back", "encoding", "delimiter"), path)
        layout = Layout()
        sheet = spec.get("sheet")
        if isinstance(sheet, int) and not isinstance(sheet, bool):
            layout.sheet = sheet
        elif isinstance(sheet, str) and sheet.strip().isdigit():
            layout.sheet = int(sheet)
        else:
            layout.sheet = self.text(sheet, join_path(path, "sheet"))
        header = spec.get("header_row")
        if not (header is None or (isinstance(header, str) and header.strip().lower() in ("", "auto"))):
            layout.header_row = self.integer(header, join_path(path, "header_row"), minimum=1)
        where = spec.get("where") or {}
        if not isinstance(where, Mapping):
            self.issues.add(join_path(path, "where"), "expected a mapping of column name to allowed values")
        else:
            for column, allowed in where.items():
                layout.where[str(column)] = self.text_list(allowed, join_path(join_path(path, "where"), column))
        columns = spec.get("columns") or {}
        if not isinstance(columns, Mapping):
            self.issues.add(join_path(path, "columns"), "expected a mapping of field to column header")
        else:
            for key, headers in columns.items():
                where_path = join_path(join_path(path, "columns"), key)
                if key not in FIELDS:
                    hint = suggest(str(key), FIELDS)
                    self.issues.add(where_path, f"unknown field {key!r}" + (f"; did you mean {hint!r}?" if hint else "")
                                    + f"; fields are {', '.join(FIELDS)}")
                    continue
                if isinstance(headers, str):
                    names = [part.strip() for part in headers.split("|") if part.strip()]
                else:
                    names = self.text_list(headers, where_path)
                layout.columns[key] = names
        layout.passthrough = self.text_list(spec.get("passthrough"), join_path(path, "passthrough"))
        vocab = spec.get("vocab") or {}
        if not isinstance(vocab, Mapping):
            self.issues.add(join_path(path, "vocab"), "expected a mapping of field to {cell text: meaning}")
        else:
            for key, table in vocab.items():
                vocab_path = join_path(join_path(path, "vocab"), key)
                if key not in FIELDS:
                    self.issues.add(vocab_path, f"unknown field {key!r}")
                    continue
                if not isinstance(table, Mapping):
                    self.issues.add(vocab_path, "expected a mapping of cell text to meaning")
                    continue
                layout.vocab[key] = {str(k).strip().casefold(): str(v).strip() for k, v in table.items()}
        layout.decimal_comma = self.boolean(spec.get("decimal_comma"), join_path(path, "decimal_comma"))
        write_back = spec.get("write_back") or {}
        if not isinstance(write_back, Mapping):
            self.issues.add(join_path(path, "write_back"), "expected a mapping of result field to column header")
        else:
            for key, header in write_back.items():
                wb_path = join_path(join_path(path, "write_back"), key)
                if key not in RESULT_FIELDS:
                    self.issues.add(wb_path, f"unknown result field {key!r}; result fields are "
                                             f"{', '.join(RESULT_FIELDS)}")
                    continue
                text = self.text(header, wb_path, required=True)
                if text:
                    layout.write_back[key] = text
        layout.encoding = self.text(spec.get("encoding"), join_path(path, "encoding"))
        layout.delimiter = self.text(spec.get("delimiter"), join_path(path, "delimiter"))
        return layout

    def checks(self, value: object, path: str) -> ChecksJoin | None:
        if value is None:
            return None
        spec = self.keys(value, ("file", "sheet", "key", "header_row"), path)
        file = self.text(spec.get("file"), join_path(path, "file"), required=True)
        if file is None:
            return None
        header = spec.get("header_row")
        return ChecksJoin(
            file=file,
            sheet=self.text(spec.get("sheet"), join_path(path, "sheet")),
            key=self.text(spec.get("key"), join_path(path, "key")),
            header_row=None if header in (None, "", "auto") else self.integer(header, join_path(path, "header_row"),
                                                                            minimum=1),
        )

    def time(self, value: object, path: str) -> TimeSpec:
        spec = self.keys(value, ("signal", "unit", "on_non_monotonic", "t0"), path)
        return TimeSpec(
            signal=self.text(spec.get("signal"), join_path(path, "signal")),
            unit=self.text(spec.get("unit"), join_path(path, "unit")),
            on_non_monotonic=self.choice(spec.get("on_non_monotonic"), ("error", "sort", "drop"),
                                         join_path(path, "on_non_monotonic"), "error"),
            t0=self.text(spec.get("t0"), join_path(path, "t0")),
        )

    def units(self, value: object, path: str) -> dict[str, UnitDef]:
        spec = self.keys(value, list(value) if isinstance(value, Mapping) else (), path)
        out: dict[str, UnitDef] = {}
        base = UnitTable()
        for symbol, definition in spec.items():
            unit_path = join_path(path, symbol)
            if isinstance(definition, (str, int, float)) and not isinstance(definition, bool):
                try:
                    q = base.parse(definition)
                except UnitsError as exc:
                    self.issues.add(unit_path, f"cannot read {definition!r}: {exc}")
                    continue
                if q.unit is None:
                    self.issues.add(unit_path, "write the definition with a unit, such as '9.80665 m/s^2'")
                    continue
                known = base.lookup(q.unit)
                out[str(symbol)] = UnitDef(str(symbol), known.dimension, q.value * known.factor)
                continue
            fields = self.keys(definition, ("dimension", "factor", "offset"), unit_path)
            dimension = self.text(fields.get("dimension"), join_path(unit_path, "dimension"), required=True)
            try:
                factor = float(fields.get("factor", 1.0))
                offset = float(fields.get("offset", 0.0) or 0.0)
            except (TypeError, ValueError):
                self.issues.add(unit_path, "factor and offset must be numbers")
                continue
            if factor <= 0:
                self.issues.add(join_path(unit_path, "factor"), "the factor must be positive")
                continue
            if dimension:
                out[str(symbol)] = UnitDef(str(symbol), dimension.lower().replace(" ", "_"), factor, offset)
        return out

    def signals(self, value: object, path: str, units: UnitTable) -> dict[str, SignalAlias]:
        spec = self.keys(value, list(value) if isinstance(value, Mapping) else (), path)
        out: dict[str, SignalAlias] = {}
        for alias, definition in spec.items():
            alias_path = join_path(path, alias)
            name = self.name(alias, alias_path, IDENTIFIER_RE, "alias")
            if name is None:
                continue
            entry = SignalAlias(name=name, loc=self.issues.locations.get(alias_path))
            if isinstance(definition, str):
                entry.path = definition.strip()
            else:
                fields = self.keys(definition, ("path", "unit", "kind", "labels", "expr", "time"), alias_path)
                entry.path = self.text(fields.get("path"), join_path(alias_path, "path"))
                entry.unit = self.text(fields.get("unit"), join_path(alias_path, "unit"))
                entry.kind = self.choice(fields.get("kind"), KINDS, join_path(alias_path, "kind"), "") or None
                entry.expr = self.text(fields.get("expr"), join_path(alias_path, "expr"))
                entry.time = self.text(fields.get("time"), join_path(alias_path, "time"))
                entry.labels = self.labels(fields.get("labels"), join_path(alias_path, "labels"))
            if (entry.path is None) == (entry.expr is None):
                self.issues.add(alias_path, "give either a path or an expr")
            # An unknown unit is kept as an opaque label: limits on the signal must then be bare numbers.
            out[name] = entry
        return out

    def labels(self, value: object, path: str) -> dict[int, str] | None:
        if value is None or value == "":
            return None
        items: dict = {}
        if isinstance(value, str):
            items = _labels_text(value)
        elif isinstance(value, Mapping):
            items = dict(value)
        elif isinstance(value, Sequence):
            items = {i: label for i, label in enumerate(value)}
        else:
            self.issues.add(path, "expected a mapping of code to label, such as {0: IDLE, 1: BURN}")
            return None
        out: dict[int, str] = {}
        for code, label in items.items():
            try:
                number = int(str(code).strip())
            except ValueError:
                self.issues.add(path, f"label codes are whole numbers, got {code!r}")
                continue
            text = str(label).strip()
            if not text:
                self.issues.add(path, f"code {number} has an empty label")
                continue
            if text in out.values():
                self.issues.add(path, f"label {text!r} is used twice")
            out[number] = text
        return out

    def events(self, value: object, path: str, units: UnitTable) -> dict[str, EventDef]:
        spec = self.keys(value, list(value) if isinstance(value, Mapping) else (), path)
        out: dict[str, EventDef] = {}
        for name, definition in spec.items():
            event_path = join_path(path, name)
            if self.name(name, event_path, EVENT_NAME_RE, "event") is None:
                continue
            fields = self.keys(definition, ("signal", *EVENT_CONDITIONS, "when", "param", "hysteresis", "debounce",
                                            "occurrence"), event_path)
            event = EventDef(name=name, loc=self.issues.locations.get(event_path))
            event.signal = self.text(fields.get("signal"), join_path(event_path, "signal"))
            conditions = [c for c in EVENT_CONDITIONS if c in fields]
            if len(conditions) > 1:
                self.issues.add(event_path, f"give only one of {', '.join(EVENT_CONDITIONS)}")
            if conditions:
                event.condition = conditions[0]
                event.value = fields[conditions[0]]
                if event.value is None or event.value == "":
                    self.issues.add(join_path(event_path, event.condition), "a value is required")
            event.when = self.text(fields.get("when"), join_path(event_path, "when"))
            event.param = self.text(fields.get("param"), join_path(event_path, "param"))
            event.hysteresis = self.text(fields.get("hysteresis"), join_path(event_path, "hysteresis"))
            event.debounce = self.text(fields.get("debounce"), join_path(event_path, "debounce"))
            occurrence = self.text(fields.get("occurrence"), join_path(event_path, "occurrence")) or "first"
            if not OCCURRENCE_RE.match(occurrence.lower()):
                self.issues.add(join_path(event_path, "occurrence"), "expected first, last or a trigger number")
            event.occurrence = occurrence.lower()
            ways = sum(x is not None for x in (event.condition, event.when, event.param))
            if event.signal is not None and event.condition is None and ways == 0:
                self.issues.add(event_path, "a signal needs falls_below, rises_above or equals")
            elif ways != 1:
                self.issues.add(event_path, "define the event with exactly one of: a signal with falls_below, "
                                            "rises_above or equals; a when expression; a param")
            elif event.condition is not None and event.signal is None:
                self.issues.add(event_path, f"{event.condition} needs a signal")
            elif event.condition is None and event.signal is not None:
                self.issues.add(event_path, "a signal belongs with falls_below, rises_above or equals, not with "
                                            "when or param")
            for key in ("hysteresis", "debounce"):
                text = getattr(event, key)
                if text is None:
                    continue
                try:
                    q = units.parse(text)
                    if key == "debounce" and q.unit is not None and units.dimension(q.unit) != "time":
                        raise UnitsError(f"{text!r} is not a duration")
                    if q.value < 0:
                        raise UnitsError(f"{text!r} must not be negative")
                except UnitsError as exc:
                    self.issues.add(join_path(event_path, key), str(exc))
            out[name] = event
        return out

    def conditions(self, value: object, path: str) -> dict[str, str]:
        spec = self.keys(value, list(value) if isinstance(value, Mapping) else (), path)
        out: dict[str, str] = {}
        for name, text in spec.items():
            cond_path = join_path(path, name)
            if self.name(name, cond_path, IDENTIFIER_RE, "condition") is None:
                continue
            expression = self.text(text, cond_path, required=True)
            if expression:
                out[name] = expression
        return out

    def curves(self, value: object, path: str, units: UnitTable) -> dict[str, CurveDef]:
        spec = self.keys(value, list(value) if isinstance(value, Mapping) else (), path)
        out: dict[str, CurveDef] = {}
        for name, definition in spec.items():
            curve_path = join_path(path, name)
            if self.name(name, curve_path, re.compile(r"^[A-Za-z_][\w.-]*$"), "curve") is None:
                continue
            fields = self.keys(definition, ("points", "x", "y", "x_unit", "y_unit", "outside", "mode"), curve_path)
            raw = fields.get("points")
            xs = fields.get("x") if isinstance(fields.get("x"), Sequence) and not isinstance(fields.get("x"), str) else None
            if raw is None and xs is not None and isinstance(fields.get("y"), Sequence):
                raw = list(zip(xs, fields["y"]))
            points: list[tuple[float, float]] = []
            if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) < 2:
                self.issues.add(curve_path, "a curve needs at least two points, as [[x, y], ...]")
                continue
            for i, point in enumerate(raw):
                try:
                    x, y = point
                    points.append((float(str(x).replace(",", ".") if isinstance(x, str) else x),
                                   float(str(y).replace(",", ".") if isinstance(y, str) else y)))
                except (TypeError, ValueError):
                    self.issues.add(item_path(join_path(curve_path, "points"), i), f"expected [x, y] numbers, got {point!r}")
            xs_sorted = [p[0] for p in points]
            if any(b <= a for a, b in zip(xs_sorted, xs_sorted[1:])):
                self.issues.add(join_path(curve_path, "points"), "x values must be strictly increasing")
            curve = CurveDef(name=name, points=points, loc=self.issues.locations.get(curve_path))
            curve.x_unit = self.text(fields.get("x_unit"), join_path(curve_path, "x_unit"))
            curve.y_unit = self.text(fields.get("y_unit"), join_path(curve_path, "y_unit"))
            for key in ("x_unit", "y_unit"):
                symbol = getattr(curve, key)
                if symbol is not None and not units.known(symbol):
                    self.issues.add(join_path(curve_path, key), units.unknown(symbol))
            if isinstance(fields.get("x"), str):
                curve.x = fields["x"].strip() or None
            curve.outside = self.choice(fields.get("outside"), ("clamp", "none", "error"),
                                        join_path(curve_path, "outside"), "clamp")
            curve.mode = self.choice(fields.get("mode"), ("linear", "step"), join_path(curve_path, "mode"), "linear")
            out[name] = curve
        return out

    def params(self, value: object, path: str, units: UnitTable) -> ParamsSpec:
        spec = self.keys(value, ("key", "file", "units", "from_source"), path)
        params = ParamsSpec(key=self.text(spec.get("key"), join_path(path, "key")),
                            file=self.text(spec.get("file"), join_path(path, "file")),
                            from_source=self.text_list(spec.get("from_source"), join_path(path, "from_source")))
        declared = spec.get("units") or {}
        if not isinstance(declared, Mapping):
            self.issues.add(join_path(path, "units"), "expected a mapping of parameter name to unit")
        else:
            for name, symbol in declared.items():
                text = self.text(symbol, join_path(join_path(path, "units"), name), required=True)
                if text is None:
                    continue
                if not units.known(text):
                    self.issues.add(join_path(join_path(path, "units"), name), units.unknown(text))
                params.units[str(name)] = text
        return params

    def defaults(self, value: object, path: str, units: UnitTable) -> Defaults:
        from .limits import LimitError, parse_margin, parse_tolerance

        spec = self.keys(value, ("tolerance", "margin", "grid", "on_gap", "on_missing_event", "on_case_overlap",
                                 "on_tolerated"), path)
        defaults = Defaults()
        tolerance = self.text(spec.get("tolerance"), join_path(path, "tolerance"))
        if tolerance is not None:
            try:
                parse_tolerance(tolerance, units=units)
                defaults.tolerance = tolerance
            except LimitError as exc:
                self.issues.add(join_path(path, "tolerance"), str(exc))
        margin = spec.get("margin")
        if margin is not None:
            text = self.text(margin, join_path(path, "margin"))
            try:
                parse_margin(text, units=units)
                defaults.margin = text
            except LimitError as exc:
                self.issues.add(join_path(path, "margin"), str(exc))
        defaults.grid = self.text(spec.get("grid"), join_path(path, "grid")) or "first"
        defaults.on_gap = self.choice(spec.get("on_gap"), ("warn", "ignore", "fail"), join_path(path, "on_gap"), "warn")
        defaults.on_missing_event = self.choice(spec.get("on_missing_event"), ("warn", "fail", "na"),
                                                join_path(path, "on_missing_event"), "warn")
        defaults.on_case_overlap = self.choice(spec.get("on_case_overlap"), ("first", "error"),
                                               join_path(path, "on_case_overlap"), "first")
        defaults.on_tolerated = self.choice(spec.get("on_tolerated"), ("pass", "warn"),
                                            join_path(path, "on_tolerated"), "pass")
        return defaults

    def report(self, value: object, path: str) -> ReportSpec:
        spec = self.keys(value, ("title", "pages", "plot_bins", "overlays"), path)
        report = ReportSpec(title=self.text(spec.get("title"), join_path(path, "title")))
        report.pages = self.choice(spec.get("pages"), ("failed", "all", "none"), join_path(path, "pages"), "failed")
        bins = self.integer(spec.get("plot_bins"), join_path(path, "plot_bins"), minimum=64, maximum=8192)
        if bins is not None:
            report.plot_bins = bins
        report.overlays = self.text_list(spec.get("overlays"), join_path(path, "overlays"))
        return report

