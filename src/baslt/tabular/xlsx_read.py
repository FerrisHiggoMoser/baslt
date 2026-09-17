"""Read .xlsx sheets with the standard library: cell values, the text a person sees, merged cells and sheet names.

An .xlsx file is a zip of XML parts. The reader follows the package relationships to the workbook, its sheets,
shared strings and styles, so it accepts files from Excel, LibreOffice, Polarion and generating libraries alike,
in the transitional and the strict namespaces.

- Shared and inline strings join their rich-text runs, skip phonetic runs and decode `_xHHHH_` escapes.
- Numbers keep their value; their text follows the cell format where it matters: a percent cell holding 0.05
  reads "5%", a date cell reads as an ISO date, an integral number reads without ".0".
- A formula without a cached value (a file saved by a library, never opened in Excel) reads as empty and is
  flagged, so callers can ask for the file to be opened and saved once.
- Parts are size-capped and a DOCTYPE is refused, so a hostile file cannot expand entities or exhaust memory.
"""

from __future__ import annotations

import math
import posixpath
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

from ..errors import TableError
from .table import Cell, SheetInfo, Table, column_letter, split_ref

__all__ = ["SheetPart", "open_workbook", "read_sheet", "workbook_sheets"]

MAX_PART_BYTES = 512 * 1024 * 1024
MAX_RATIO = 1000

REL_NAMESPACES = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "http://purl.oclc.org/ooxml/officeDocument/relationships",
)
DATE_BUILTINS = frozenset({14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47})
PERCENT_BUILTINS = frozenset({9, 10})

_ESCAPE_RE = re.compile(r"_x([0-9A-Fa-f]{4})_")
_QUOTED_RE = re.compile(r'"[^"]*"|\[[^\]]*\]|\\.')


def local(tag: str) -> str:
    """The name of an element or attribute without its namespace or prefix."""
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def rel_id(element: ElementTree.Element) -> str | None:
    for key, value in element.attrib.items():
        if key.startswith("{") and key[1:].split("}", 1)[0] in REL_NAMESPACES and local(key) == "id":
            return value
    return None


def unescape(text: str) -> str:
    """Decode the `_xHHHH_` escapes spreadsheet XML uses for characters XML cannot hold."""
    if "_x" not in text:
        return text
    return _ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


class SheetPart:
    """Where one sheet lives inside the package."""

    def __init__(self, name: str, part: str, hidden: bool) -> None:
        self.name = name
        self.part = part
        self.hidden = hidden


class Workbook:
    """An open .xlsx package: its sheets, shared strings and number formats, read lazily."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.zip = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise TableError(f"{path} is not a valid .xlsx workbook: {exc}") from None
        self.names = {info.filename: info for info in self.zip.infolist()}
        self.workbook_part = self._office_document()
        self.date1904 = False
        self.sheets = self._sheets()
        self._shared: list[str] | None = None
        self._formats: tuple[list[str], set[int], set[int]] | None = None

    def close(self) -> None:
        self.zip.close()

    def __enter__(self) -> Workbook:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- parts ------------------------------------------------------------------------------------------

    def read(self, part: str) -> bytes:
        info = self.names.get(part)
        if info is None:
            raise TableError(f"{self.path.name}: the workbook has no part {part!r}")
        bomb = info.file_size > 16 * 1024 * 1024 and info.file_size > MAX_RATIO * max(info.compress_size, 1)
        if info.file_size > MAX_PART_BYTES or bomb:
            raise TableError(f"{self.path.name}: part {part} is too large to read safely ({info.file_size} bytes)")
        data = self.zip.read(part)
        head = data[:4096]
        if b"<!DOCTYPE" in head or b"<!ENTITY" in data:
            raise TableError(f"{self.path.name}: part {part} declares a DOCTYPE, which is not allowed")
        return data

    def xml(self, part: str) -> ElementTree.Element:
        try:
            return ElementTree.fromstring(self.read(part))
        except ElementTree.ParseError as exc:
            raise TableError(f"{self.path.name}: part {part} is not well-formed XML: {exc}") from None

    def rels(self, part: str) -> dict[str, tuple[str, str]]:
        """Relationship id -> (type, resolved part name) for `part`."""
        folder, name = posixpath.split(part)
        rels_part = posixpath.join(folder, "_rels", f"{name}.rels")
        if rels_part not in self.names:
            return {}
        out: dict[str, tuple[str, str]] = {}
        for rel in self.xml(rels_part):
            if local(rel.tag) != "Relationship":
                continue
            if rel.get("TargetMode") == "External":
                out[rel.get("Id", "")] = (rel.get("Type", ""), rel.get("Target", ""))
                continue
            target = rel.get("Target", "")
            resolved = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
            out[rel.get("Id", "")] = (rel.get("Type", ""), resolved)
        return out

    def _office_document(self) -> str:
        for rel_type, target in self.rels("").values():
            if rel_type.endswith("/officeDocument"):
                return target
        if "xl/workbook.xml" in self.names:
            return "xl/workbook.xml"
        raise TableError(f"{self.path.name} has no workbook part; is it an .xlsx file?")

    def _sheets(self) -> list[SheetPart]:
        root = self.xml(self.workbook_part)
        rels = self.rels(self.workbook_part)
        sheets: list[SheetPart] = []
        for element in root.iter():
            name = local(element.tag)
            if name == "workbookPr":
                self.date1904 = element.get("date1904", "0").lower() in ("1", "true")
            if name != "sheet":
                continue
            rid = rel_id(element)
            rel_type, target = rels.get(rid or "", ("", ""))
            if not rel_type.endswith("/worksheet"):
                continue  # chart sheets and dialog sheets hold no cells
            sheets.append(SheetPart(element.get("name", ""), target, element.get("state", "visible") != "visible"))
        if not sheets:
            raise TableError(f"{self.path.name} has no worksheets")
        return sheets

    def related(self, suffix: str) -> str | None:
        for rel_type, target in self.rels(self.workbook_part).values():
            if rel_type.endswith(suffix):
                return target
        return None

    # ----- shared strings and formats ---------------------------------------------------------------------

    @property
    def shared(self) -> list[str]:
        if self._shared is None:
            part = self.related("/sharedStrings")
            self._shared = [] if part is None or part not in self.names else [
                string_item(si) for si in self.xml(part) if local(si.tag) == "si"
            ]
        return self._shared

    @property
    def formats(self) -> tuple[list[str], set[int], set[int]]:
        """Per cell style: the format code, and the styles showing percentages and dates."""
        if self._formats is None:
            codes: dict[int, str] = {}
            styles: list[int] = []
            part = self.related("/styles")
            if part is not None and part in self.names:
                root = self.xml(part)
                for element in root:
                    if local(element.tag) == "numFmts":
                        for fmt in element:
                            codes[int(fmt.get("numFmtId", "0"))] = fmt.get("formatCode", "")
                    elif local(element.tag) == "cellXfs":
                        styles = [int(xf.get("numFmtId", "0")) for xf in element if local(xf.tag) == "xf"]
            percent: set[int] = set()
            dates: set[int] = set()
            texts: list[str] = []
            for index, fmt_id in enumerate(styles):
                code = codes.get(fmt_id, "")
                texts.append(code)
                if fmt_id in PERCENT_BUILTINS or _is_percent(code):
                    percent.add(index)
                elif fmt_id in DATE_BUILTINS or _is_date(code):
                    dates.add(index)
            self._formats = (texts, percent, dates)
        return self._formats

    def find(self, sheet: str | int | None) -> SheetPart:
        if sheet is None:
            for part in self.sheets:
                if not part.hidden:
                    return part
            return self.sheets[0]
        if isinstance(sheet, int) and not isinstance(sheet, bool):
            if 1 <= sheet <= len(self.sheets):
                return self.sheets[sheet - 1]
            raise TableError(f"{self.path.name} has {len(self.sheets)} sheets; there is no sheet {sheet}")
        wanted = str(sheet).strip().casefold()
        for part in self.sheets:
            if part.name.casefold() == wanted:
                return part
        names = ", ".join(repr(part.name) for part in self.sheets)
        raise TableError(f"{self.path.name} has no sheet {sheet!r}; its sheets are {names}")


def _strip_quoted(code: str) -> str:
    return _QUOTED_RE.sub("", code)


def _is_percent(code: str) -> bool:
    return "%" in _strip_quoted(code)


def _is_date(code: str) -> bool:
    bare = _strip_quoted(code).lower()
    if not bare or bare == "general":
        return False
    return any(token in bare for token in ("yy", "dd", "mm", "hh", "ss", "d/", "m/", "/d", "/m")) or \
        bare.strip() in ("d", "m", "y", "h")


def string_item(element: ElementTree.Element) -> str:
    """The text of a shared or inline string: plain `<t>` or rich-text runs, without phonetic runs."""
    parts: list[str] = []
    for child in element:
        name = local(child.tag)
        if name == "t":
            parts.append(child.text or "")
        elif name == "r":
            for run in child:
                if local(run.tag) == "t":
                    parts.append(run.text or "")
    return unescape("".join(parts))


def number_text(value: float) -> str:
    if not math.isfinite(value):
        return repr(value)
    if value == int(value) and abs(value) < 2**53:
        return str(int(value))
    return repr(value)


def percent_text(value: float) -> str:
    scaled = round(value * 100, 10)
    return f"{number_text(scaled)}%"


def date_text(value: float, date1904: bool) -> str:
    base = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
    try:
        moment = base + timedelta(days=value)
    except OverflowError:
        return number_text(value)
    if moment.time() == datetime.min.time():
        return moment.date().isoformat()
    return moment.isoformat(timespec="seconds")


def open_workbook(path: str | Path) -> Workbook:
    return Workbook(Path(path))


def workbook_sheets(path: str | Path) -> list[SheetInfo]:
    with open_workbook(path) as book:
        return [SheetInfo(name=part.name, part=part.part, hidden=part.hidden) for part in book.sheets]


def read_sheet(path: str | Path, sheet: str | int | None = None) -> Table:
    path = Path(path)
    with open_workbook(path) as book:
        part = book.find(sheet)
        rows, merged, uncached = _cells(book, part)
    issues: list[str] = []
    if uncached:
        shown = ", ".join(uncached[:5]) + (f" and {len(uncached) - 5} more" if len(uncached) > 5 else "")
        issues.append(
            f"{path.name}:{part.name}: formula cells {shown} have no saved value; open the workbook in Excel and "
            "save it once so their results are stored"
        )
    return Table(path=path, sheet=part.name, rows=rows, format="xlsx", merged=merged, issues=issues)


def _cells(book: Workbook, part: SheetPart) -> tuple[list[list[Cell | None]], list[tuple[int, int, int, int]], list[str]]:
    root = book.xml(part.part)
    shared = book.shared
    _, percent, dates = book.formats
    rows: dict[int, dict[int, Cell]] = {}
    merged: list[tuple[int, int, int, int]] = []
    uncached: list[str] = []
    for container in root:
        name = local(container.tag)
        if name == "mergeCells":
            for item in container:
                ref = item.get("ref", "")
                if ":" in ref:
                    first, last = ref.split(":", 1)
                    (r1, c1), (r2, c2) = split_ref(first), split_ref(last)
                    merged.append((r1, c1, r2, c2))
            continue
        if name != "sheetData":
            continue
        row_number = 0
        for row in container:
            if local(row.tag) != "row":
                continue
            row_number = int(row.get("r", row_number + 1))
            col_number = 0
            cells = rows.setdefault(row_number, {})
            for element in row:
                if local(element.tag) != "c":
                    continue
                ref = element.get("r")
                if ref:
                    _, col_number = split_ref(ref)
                else:
                    col_number += 1
                cell = _cell(element, row_number, col_number, shared, percent, dates, book.date1904)
                if cell.formula_uncached:
                    uncached.append(cell.ref)
                cells[col_number] = cell
    out: list[list[Cell | None]] = []
    last_row = max((r for r, cells in rows.items() if cells), default=0)
    for r in range(1, last_row + 1):
        cells = rows.get(r, {})
        width = max(cells, default=0)
        out.append([cells.get(c) for c in range(1, width + 1)])
    return out, merged, uncached


def _cell(element: ElementTree.Element, row: int, col: int, shared: list[str], percent: set[int], dates: set[int],
          date1904: bool) -> Cell:
    ref = f"{column_letter(col)}{row}"
    kind = element.get("t", "n")
    style = int(element.get("s", "0") or 0)
    value_text: str | None = None
    has_formula = False
    inline: ElementTree.Element | None = None
    for child in element:
        name = local(child.tag)
        if name == "v":
            value_text = child.text or ""
        elif name == "f":
            has_formula = True
        elif name == "is":
            inline = child
    if kind == "inlineStr":
        text = string_item(inline) if inline is not None else ""
        return Cell(value=text or None, text=text, ref=ref)
    if value_text is None:
        return Cell(value=None, text="", ref=ref, formula_uncached=has_formula)
    if kind == "s":
        try:
            text = shared[int(value_text)]
        except (ValueError, IndexError):
            text = ""
        return Cell(value=text or None, text=text, ref=ref)
    if kind in ("str", "d"):
        text = unescape(value_text)
        return Cell(value=text or None, text=text, ref=ref)
    if kind == "b":
        flag = value_text.strip() in ("1", "true")
        return Cell(value=flag, text="TRUE" if flag else "FALSE", ref=ref)
    if kind == "e":
        return Cell(value=None, text=value_text, ref=ref)
    try:
        number = float(value_text)
    except ValueError:
        return Cell(value=value_text or None, text=value_text, ref=ref)
    if style in percent:
        text = percent_text(number)
    elif style in dates:
        text = date_text(number, date1904)
    else:
        text = number_text(number)
    return Cell(value=number, text=text, ref=ref)
