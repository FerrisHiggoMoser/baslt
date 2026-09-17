"""Read a requirements table into Requirements: find the header, map columns, filter rows, group cases.

- The sheet is the configured one, else a sheet named "Requirements", else the first visible sheet that is not a
  config sheet.
- The header row is the configured one, else the first of the first 20 rows in which at least two cells name a
  known field (ID, Check, Limit, ...). Header names match case-, space- and punctuation-insensitively; the
  mapping's `columns` come first, then built-in synonyms.
- `where` keeps only rows whose cells hold one of the allowed values (Polarion exports carry headings and prose
  rows). Rows without an ID are skipped; a row with a check but no ID is reported.
- Rows sharing an ID are the cases of one requirement, in row order. A requirement without a check is NOT
  COVERED. A cell that cannot be read makes its requirement an ERROR without stopping the others.
- With `checks:`, the check columns come from a second, engineer-owned table joined by ID.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from fnmatch import fnmatchcase
from pathlib import Path

from ..errors import Issue, RequirementsError
from ..tabular import Cell, Table, list_sheets, read_table
from .config import CONFIG_SHEETS, DEFAULT_SYNONYMS, FIELDS, Config, Layout, normalize_header
from .limits import LimitError, parse_count, parse_limit, parse_limit_columns, parse_margin, parse_tolerance
from .model import KINDS, Case, Loc, Requirement, RequirementSet
from .units_ext import UnitsError

__all__ = ["BUILTIN_KIND_WORDS", "find_header", "load_requirements", "map_columns", "resolve_file"]

HEADER_SCAN_ROWS = 20
FIELD_PRIORITY: tuple[str, ...] = (
    "id", "check", "kind", "limit", "title", "unit", "when", "applies_to", "tolerance", "margin", "case", "count",
    "severity", "grid", "lower", "upper", "notes",
)
CHECK_FIELDS: tuple[str, ...] = tuple(f for f in FIELDS if f not in ("id", "title"))
SKIP_SHEETS = {name.casefold() for name in (*CONFIG_SHEETS, "Guide", "Checks", "Readme", "Help")}

BUILTIN_KIND_WORDS: dict[str, str] = {
    **{kind: kind for kind in KINDS},
    "upperlimit": "upper", "maximumlimit": "upper", "maxlimit": "upper", "ceiling": "upper", "nottoexceed": "upper",
    "notexceed": "upper", "le": "upper", "<=": "upper", "≤": "upper",
    "lowerlimit": "lower", "minimumlimit": "lower", "minlimit": "lower", "floor": "lower", "ge": "lower",
    ">=": "lower", "≥": "lower",
    "band": "range", "within": "range", "between": "range", "rangelimit": "range",
    "shallhold": "assert", "always": "assert", "invariant": "assert", "condition": "assert", "check": "assert",
    "timeabove": "duration", "totaltime": "duration", "time": "duration",
    "scalar": "value", "number": "value", "calculated": "value",
    "occurs": "event", "eventtime": "event", "timing": "event", "occurrence": "event",
    "maximum": "max", "peak": "max", "minimum": "min", "average": "mean", "avg": "mean",
    "end": "final", "finalvalue": "final", "last": "final", "start": "initial", "initialvalue": "initial",
    "first": "initial",
}


def resolve_file(name: str | Path, config: Config, fallback: Path | None = None) -> Path:
    """A file named in the mapping, relative to the mapping (or the requirements file when there is none)."""
    path = Path(name).expanduser()
    if path.is_absolute():
        return path
    base = config.base or fallback or Path.cwd()
    return base / path


def _pick_sheet(path: Path, layout: Layout) -> str | int | None:
    if layout.sheet is not None or path.suffix.lower() not in (".xlsx", ".xlsm"):
        return layout.sheet
    sheets = list_sheets(path)
    for sheet in sheets:
        if sheet.name.casefold() in ("requirements", "reqs"):
            return sheet.name
    for skip in (SKIP_SHEETS, SKIP_SHEETS - {"checks"}):
        for sheet in sheets:
            if not sheet.hidden and sheet.name.casefold() not in skip:
                return sheet.name
    return None


def _candidates(field: str, layout: Layout) -> list[str]:
    configured = [normalize_header(name) for name in layout.columns.get(field, [])]
    return configured if configured else list(DEFAULT_SYNONYMS.get(field, ()))


def find_header(table: Table, layout: Layout, *, header_row: int | None = None) -> int:
    """The header row: the configured one, or the first row naming at least two known fields."""
    if header_row is not None:
        if header_row > table.n_rows:
            raise RequirementsError(f"{table.location(header_row)}: the header row is past the end of the table "
                                    f"({table.n_rows} rows)")
        return header_row
    wanted = set()
    for field in FIELDS:
        wanted.update(_candidates(field, layout))
    for key in layout.where:
        wanted.add(normalize_header(key))
    for r in range(1, min(table.n_rows, HEADER_SCAN_ROWS) + 1):
        names = {normalize_header(text) for text in table.row_texts(r) if text.strip()}
        if len(names & wanted) >= 2:
            return r
    where = f"{table.path.name}" + (f":{table.sheet}" if table.sheet else "")
    raise RequirementsError(
        f"{where}: no header row found in the first {HEADER_SCAN_ROWS} rows (looked for column names such as ID, "
        "Check, Type, Limit, Unit, When); set requirements.header_row and requirements.columns in the mapping"
    )


def map_columns(table: Table, header_row: int, layout: Layout, fields: Sequence[str] = FIELD_PRIORITY
                ) -> tuple[dict[str, int], list[str]]:
    """Field -> column, and notes about headers that appear twice."""
    headers = [normalize_header(text) for text in table.row_texts(header_row)]
    notes: list[str] = []
    seen: dict[str, int] = {}
    for col, name in enumerate(headers, start=1):
        if not name:
            continue
        if name in seen:
            notes.append(f"{table.location(header_row, col)}: column {table.text(header_row, col)!r} appears twice; "
                         "the first one is used")
            continue
        seen[name] = col
    taken: set[int] = set()
    columns: dict[str, int] = {}
    filters = {normalize_header(key) for key in layout.where}
    for field in fields:
        configured = bool(layout.columns.get(field))
        for candidate in _candidates(field, layout):
            col = seen.get(candidate)
            if col is None or col in taken:
                continue
            if candidate in filters and not configured:
                # a column that filters rows (a Polarion "Type" of Heading or Requirement) is not a check field
                notes.append(f"{table.location(header_row, col)}: column {table.text(header_row, col)!r} filters "
                             f"rows (requirements.where), so it is not read as the {field} field; name that "
                             f"column under requirements.columns.{field} if it is")
                continue
            columns[field] = col
            taken.add(col)
            break
    return columns, notes


def _cell_value(cell: Cell | None) -> object:
    """What a limit parser gets: numbers stay numbers unless the cell shows them differently (percent)."""
    if cell is None:
        return None
    if isinstance(cell.value, float) and not cell.text.endswith("%"):
        return cell.value
    return cell.text


def _vocab(text: str, field: str, layout: Layout) -> str:
    table = layout.vocab.get(field)
    if table:
        mapped = table.get(text.strip().casefold())
        if mapped is not None:
            return mapped
    return text


def _kind(text: str) -> str | None:
    word = text.strip().casefold()
    if not word:
        return None
    return BUILTIN_KIND_WORDS.get(word) or BUILTIN_KIND_WORDS.get(normalize_header(word))


def _nothing_selected(layout: Layout, skipped: int, only) -> str:
    """Why a table gave no requirement to check."""
    reasons = []
    if skipped:
        filters = "; ".join(f"{column}: {', '.join(values)}" for column, values in layout.where.items())
        reasons.append(f"{skipped} rows were left out by requirements.where ({filters})")
    if only:
        reasons.append(f"--only kept {', '.join(only)}")
    return "no requirement to check" + (": " + " and ".join(reasons) if reasons else "")


class _Loader:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.layout = config.requirements
        self.units = config.units
        self.issues: list[Issue] = []
        self.warnings: list[Issue] = []
        defaults = config.defaults
        try:
            self.default_tolerance = parse_tolerance(defaults.tolerance, units=self.units)
        except LimitError:
            self.default_tolerance = parse_tolerance("", units=self.units)
        try:
            self.default_margin = parse_margin(defaults.margin, units=self.units) if defaults.margin else None
        except LimitError:
            self.default_margin = None

    def where_columns(self, table: Table, header_row: int) -> dict[int, set[str]]:
        headers = {normalize_header(text): col for col, text in enumerate(table.row_texts(header_row), start=1)}
        out: dict[int, set[str]] = {}
        for key, allowed in self.layout.where.items():
            col = headers.get(normalize_header(key))
            if col is None:
                raise RequirementsError(f"{table.location(header_row)}: requirements.where names column {key!r}, "
                                        f"which the header row does not have")
            out[col] = {value.strip().casefold() for value in allowed}
        return out

    def rows(self, table: Table, header_row: int, columns: dict[str, int], where: dict[int, set[str]]
             ) -> tuple[list[tuple[int, dict[str, Cell | None]]], int]:
        kept: list[tuple[int, dict[str, Cell | None]]] = []
        skipped = 0
        for r in range(header_row + 1, table.n_rows + 1):
            cells = {field: table.cell(r, col) for field, col in columns.items()}
            if all(cell is None or cell.is_empty for cell in cells.values()):
                continue
            if any(table.text(r, col).strip().casefold() not in allowed for col, allowed in where.items()):
                skipped += 1
                continue
            kept.append((r, cells))
        return kept, skipped

    def case(self, rid: str, r: int, cells: dict[str, Cell | None], table: Table, issues: list[Issue],
             columns: dict[str, int]) -> Case:
        layout = self.layout

        def text(field: str) -> str:
            cell = cells.get(field)
            return _vocab(cell.text.strip(), field, layout) if cell is not None else ""

        def where(field: str) -> str:
            return table.location(r, columns.get(field))

        def problem(field: str, message: str) -> None:
            issues.append(Issue(path=f"{rid}.{field}", message=message, location=where(field)))

        unit = text("unit") or None
        if unit is not None and not self.units.known(unit):
            self.warnings.append(Issue(path=f"{rid}.unit", message=self.units.unknown(unit) +
                                       "; limits with it must be bare numbers", location=where("unit")))
        limit = None
        has_limit = "limit" in cells and cells["limit"] is not None and not cells["limit"].is_empty
        has_bounds = any(f in cells and cells[f] is not None and not cells[f].is_empty for f in ("lower", "upper"))
        try:
            if has_limit and has_bounds:
                problem("limit", "give the limit either in the Limit column or in the Min/Max columns, not both")
            elif has_limit:
                limit = parse_limit(_cell_value(cells["limit"]), unit=unit, units=self.units,
                                    decimal_comma=layout.decimal_comma)
            elif has_bounds:
                limit = parse_limit_columns(_cell_value(cells.get("lower")), _cell_value(cells.get("upper")),
                                            unit=unit, units=self.units, decimal_comma=layout.decimal_comma)
        except (LimitError, UnitsError) as exc:
            problem("limit", str(exc))

        tolerance = self.default_tolerance
        try:
            tolerance = parse_tolerance(_cell_value(cells.get("tolerance")), units=self.units,
                                        default=self.default_tolerance, decimal_comma=layout.decimal_comma)
        except (LimitError, UnitsError) as exc:
            problem("tolerance", str(exc))
        margin = self.default_margin
        try:
            cell = cells.get("margin")
            if cell is not None and not cell.is_empty:
                margin = parse_margin(_cell_value(cell), units=self.units, decimal_comma=layout.decimal_comma)
        except (LimitError, UnitsError) as exc:
            problem("margin", str(exc))
        count = None
        try:
            count = parse_count(_cell_value(cells.get("count")))
        except LimitError as exc:
            problem("count", str(exc))
        return Case(
            label=text("case"),
            when=text("when") or None,
            applies_to=text("applies_to") or None,
            limit=limit,
            tolerance=tolerance,
            margin=margin,
            severity=text("severity") or None,
            count=count,
            unit=unit,
            loc=Loc(table.location(r)),
            row=r,
            cells={field: where(field) for field in cells},
        )

    def group(self, table: Table, rows, columns: dict[str, int], header_row: int, passthrough: dict[str, int]
              ) -> list[Requirement]:
        order: dict[str, Requirement] = {}
        texts_by_req: dict[str, list[tuple[int, dict[str, str]]]] = {}
        for r, cells in rows:
            rid = cells["id"].text.strip() if cells.get("id") is not None else ""
            if not rid:
                if any(cells.get(f) is not None and not cells[f].is_empty for f in ("check", "limit")):
                    self.warnings.append(Issue(path="", message="a row with a check has no ID and was skipped",
                                               location=table.location(r)))
                continue
            texts = {f: (_vocab(c.text.strip(), f, self.layout) if c is not None else "") for f, c in cells.items()}
            req = order.get(rid)
            if req is None:
                req = order[rid] = Requirement(
                    id=rid, title="", kind=None, check="", cases=[], loc=Loc(table.location(r)), row=r,
                    passthrough={name: table.text(r, col) for name, col in passthrough.items()},
                )
                texts_by_req[rid] = []
            texts_by_req[rid].append((r, texts))
            req.rows.append(r)
            if not req.title and texts.get("title"):
                req.title = texts["title"]
            if req.grid is None and texts.get("grid"):
                req.grid = texts["grid"]

        for rid, req in order.items():
            entries = texts_by_req[rid]
            checks = [(r, t["check"]) for r, t in entries if t.get("check")]
            if not checks:
                req.covered = False
                if any(t.get("limit") or t.get("lower") or t.get("upper") for _, t in entries):
                    self.warnings.append(Issue(path=rid, message="has a limit but no check, so it is not covered",
                                               location=req.loc.text))
                continue
            req.check = checks[0][1]
            different = [(r, c) for r, c in checks if c != req.check]
            if different:
                r, other = different[0]
                req.issues.append(Issue(path=f"{rid}.check", message=(
                    f"its rows name different checks ({req.check!r} in row {checks[0][0]}, {other!r} in row {r}); "
                    "rows sharing an ID are cases of one check"), location=table.location(r)))
            kinds = [(r, t["kind"]) for r, t in entries if t.get("kind")]
            if kinds:
                parsed = _kind(kinds[0][1])
                if parsed is None:
                    req.issues.append(Issue(path=f"{rid}.kind", message=(
                        f"unknown check type {kinds[0][1]!r}; types are {', '.join(KINDS)} (or map yours under "
                        "requirements.vocab.kind)"), location=table.location(kinds[0][0])))
                req.kind = parsed
                for r, other in kinds[1:]:
                    if _kind(other) != parsed:
                        req.issues.append(Issue(path=f"{rid}.kind", message=(
                            f"its rows name different check types ({kinds[0][1]!r} and {other!r})"),
                            location=table.location(r)))
                        break
        return list(order.values())


def _file_hash(paths: Sequence[Path], extra: str) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(extra.encode("utf-8"))
    return digest.hexdigest()


def load_requirements(path: str | Path, config: Config, *, only: Sequence[str] | None = None) -> RequirementSet:
    """Read and group the requirements of `path` as `config` lays them out. Raises RequirementsError for a table
    that cannot be used at all; problems of single requirements are kept on the requirement."""
    path = Path(path)
    layout = config.requirements
    loader = _Loader(config)
    table = read_table(path, sheet=_pick_sheet(path, layout), encoding=layout.encoding, delimiter=layout.delimiter)
    for note in table.issues:
        loader.warnings.append(Issue(path="", message=note, location=path.name))
    header_row = find_header(table, layout, header_row=layout.header_row)
    columns, notes = map_columns(table, header_row, layout)
    loader.warnings.extend(Issue(path="", message=note) for note in notes)
    if "id" not in columns:
        raise RequirementsError(f"{table.location(header_row)}: no ID column found; name it ID or set "
                                "requirements.columns.id in the mapping")
    missing = [field for field in layout.columns if field not in columns]
    for field in missing:
        names = ", ".join(repr(n) for n in layout.columns[field])
        raise RequirementsError(f"{table.location(header_row)}: requirements.columns.{field} names {names}, which the "
                                "header row does not have")
    passthrough: dict[str, int] = {}
    headers = {normalize_header(text): col for col, text in enumerate(table.row_texts(header_row), start=1)}
    for name in layout.passthrough:
        col = headers.get(normalize_header(name))
        if col is None:
            loader.warnings.append(Issue(path="requirements.passthrough",
                                         message=f"column {name!r} is not in the header row"))
        else:
            passthrough[name] = col
    where = loader.where_columns(table, header_row)
    rows, skipped = loader.rows(table, header_row, columns, where)

    hashed = [path]
    checks_table = None
    unmatched: list[str] = []
    if config.checks is not None:
        checks_path = resolve_file(config.checks.file, config, path.resolve().parent)
        hashed.append(checks_path)
        checks_layout = Layout(sheet=config.checks.sheet, header_row=config.checks.header_row,
                               columns=dict(layout.columns), vocab=layout.vocab, decimal_comma=layout.decimal_comma)
        if config.checks.key:
            checks_layout.columns["id"] = [config.checks.key]
        checks_table = read_table(checks_path, sheet=_pick_sheet(checks_path, checks_layout))
        checks_header = find_header(checks_table, checks_layout, header_row=checks_layout.header_row)
        check_columns, check_notes = map_columns(checks_table, checks_header, checks_layout)
        loader.warnings.extend(Issue(path="", message=note) for note in check_notes)
        if "id" not in check_columns:
            raise RequirementsError(f"{checks_table.location(checks_header)}: the checks table has no ID column; set "
                                    "checks.key to its header")
        check_rows, _ = loader.rows(checks_table, checks_header, check_columns, {})
        by_id: dict[str, list] = {}
        for r, cells in check_rows:
            rid = cells["id"].text.strip() if cells.get("id") is not None else ""
            if rid:
                by_id.setdefault(rid, []).append((r, cells))
        all_ids = {table.text(r, columns["id"]).strip() for r in range(header_row + 1, table.n_rows + 1)}
        unmatched = sorted(rid for rid in by_id if rid not in all_ids)
        titles: dict[str, tuple[int, dict]] = {}
        for r, cells in rows:
            if cells.get("id") is not None:
                titles.setdefault(cells["id"].text.strip(), (r, cells))
        requirements = []
        for rid, (r, cells) in titles.items():
            if not rid:
                continue
            rows_for = by_id.get(rid)
            if rows_for:
                merged = [(cr, {**{f: cells.get(f) for f in ("title",)}, **ccells, "id": cells["id"]})
                          for cr, ccells in rows_for]
                reqs = loader.group(checks_table, merged, check_columns, checks_header, {})
                req = reqs[0]
                req.passthrough = {name: table.text(r, col) for name, col in passthrough.items()}
                req.title = cells["title"].text.strip() if cells.get("title") is not None else req.title
                req.loc = Loc(table.location(r))
                req.row = r
                req.rows = [r]
            else:
                req = Requirement(id=rid, title=cells["title"].text.strip() if cells.get("title") is not None else "",
                                  kind=None, check="", cases=[], loc=Loc(table.location(r)), row=r, rows=[r],
                                  passthrough={name: table.text(r, col) for name, col in passthrough.items()},
                                  covered=False)
            requirements.append(req)
        case_rows = by_id
        case_table = checks_table
        case_columns = check_columns
    else:
        requirements = loader.group(table, rows, columns, header_row, passthrough)
        case_rows = {}
        for r, cells in rows:
            rid = cells["id"].text.strip() if cells.get("id") is not None else ""
            if rid:
                case_rows.setdefault(rid, []).append((r, cells))
        case_table = table
        case_columns = columns

    for req in requirements:
        if not req.covered:
            continue
        for r, cells in case_rows.get(req.id, []):
            req.cases.append(loader.case(req.id, r, cells, case_table, req.issues, case_columns))
        if len(req.cases) > 1:
            for case in req.cases:
                if not case.label:
                    case.label = f"row {case.row}"
        empty = [case.row for case in req.cases if case.limit is None]
        unreadable = any(issue.path == f"{req.id}.limit" for issue in req.issues)
        if req.kind not in (None, "assert", "event") and len(empty) == len(req.cases) and not unreadable:
            article = "an" if req.kind[0] in "aeiou" else "a"
            req.issues.append(Issue(path=f"{req.id}.limit", message=f"{article} {req.kind} check needs a limit",
                                    location=req.loc.text))

    if only:
        requirements = [req for req in requirements if any(fnmatchcase(req.id, pattern) for pattern in only)]
    if not any(req.covered for req in requirements):
        loader.warnings.append(Issue(path="requirements", message=_nothing_selected(layout, skipped, only),
                                     location=table.location(header_row)))
    not_covered = [req.id for req in requirements if not req.covered]
    sheet = f"{table.sheet or ''}|{config.checks.sheet if config.checks else ''}"
    reqset = RequirementSet(
        requirements=requirements,
        config=config,
        table=table,
        sha256=_file_hash(hashed, sheet),
        header_row=header_row,
        columns=columns,
        issues=loader.issues,
        warnings=loader.warnings,
        skipped_rows=skipped,
        not_covered=not_covered,
        unmatched_checks=unmatched,
        checks_table=checks_table,
    )
    for rid in unmatched:
        reqset.warnings.append(Issue(path=rid, message="the checks table has this ID but the requirements table "
                                                       "does not", location=checks_table.path.name))
    return reqset
