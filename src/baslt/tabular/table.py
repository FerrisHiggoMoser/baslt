"""The table model shared by every reader: rows of cells with typed values, display text and locations."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import TableError
from ..sources.csv_text import DELIMITER_CANDIDATES

__all__ = [
    "Cell",
    "SheetInfo",
    "Table",
    "column_index",
    "column_letter",
    "list_sheets",
    "read_table",
    "split_ref",
]

ZIP_MAGIC = b"PK\x03\x04"
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
CSV_SUFFIXES = (".csv", ".tsv", ".txt")
XLSX_SUFFIXES = (".xlsx", ".xlsm")
REQIF_SUFFIXES = (".reqif", ".reqifz")
SNIFF_BYTES = 64 * 1024
FALLBACK_ENCODING = "cp1252"

_REF_RE = re.compile(r"^\$?([A-Za-z]{1,3})\$?(\d+)$")


@dataclass(slots=True)
class Cell:
    """One cell. `value` is typed for spreadsheets (str, float, bool) and text for CSV; `text` is what a person sees."""

    value: str | float | bool | None
    text: str
    ref: str
    formula_uncached: bool = False

    @property
    def is_empty(self) -> bool:
        return self.value is None and not self.text.strip()


@dataclass(slots=True)
class SheetInfo:
    name: str
    part: str
    hidden: bool = False


@dataclass(slots=True)
class Table:
    """A sheet or CSV file. Rows and columns are 1-based in every method; `rows[r - 1][c - 1]` is cell (r, c)."""

    path: Path
    sheet: str | None
    rows: list[list[Cell | None]]
    format: str
    merged: list[tuple[int, int, int, int]] = field(default_factory=list)  # (row1, col1, row2, col2), inclusive
    issues: list[str] = field(default_factory=list)
    delimiter: str | None = None
    encoding: str | None = None
    row_labels: list[str] | None = None  # what a row is called in its file (ReqIF object ids)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def cell(self, row: int, col: int) -> Cell | None:
        if 1 <= row <= len(self.rows):
            cells = self.rows[row - 1]
            if 1 <= col <= len(cells):
                return cells[col - 1]
        return None

    def text(self, row: int, col: int) -> str:
        cell = self.cell(row, col)
        return "" if cell is None else cell.text

    def row_texts(self, row: int) -> list[str]:
        if not 1 <= row <= len(self.rows):
            return []
        return ["" if cell is None else cell.text for cell in self.rows[row - 1]]

    def location(self, row: int, col: int | None = None) -> str:
        """Where a cell is, as a person would look for it: `reqs.xlsx:Requirements!F12` or `reqs.csv:12:6`."""
        if self.format == "csv":
            return f"{self.path.name}:{row}" + (f":{col}" if col is not None else "")
        if self.format == "reqif":
            label = self.row_labels[row - 1] if self.row_labels and 1 <= row <= len(self.row_labels) else f"row {row}"
            return f"{self.path.name}:{label}" + (f" ({self.text(1, col)})" if col is not None else "")
        where = f"{self.path.name}:{self.sheet}!" if self.sheet else f"{self.path.name}:"
        return where + (f"{column_letter(col)}{row}" if col is not None else f"row {row}")


def column_letter(col: int) -> str:
    """1 -> A, 27 -> AA."""
    if col < 1:
        raise ValueError(f"column numbers start at 1, got {col}")
    letters = ""
    while col:
        col, rest = divmod(col - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


def column_index(letters: str) -> int:
    """A -> 1, AA -> 27."""
    col = 0
    for char in letters.upper():
        if not "A" <= char <= "Z":
            raise ValueError(f"not a column name: {letters!r}")
        col = col * 26 + (ord(char) - 64)
    return col


def split_ref(ref: str) -> tuple[int, int]:
    """'F12' -> (12, 6)."""
    match = _REF_RE.match(ref.strip())
    if match is None:
        raise ValueError(f"not a cell reference: {ref!r}")
    return int(match.group(2)), column_index(match.group(1))


def _kind(path: Path, head: bytes) -> str:
    if head.startswith(OLE_MAGIC):
        raise TableError(
            f"{path} is a legacy .xls file or a password-protected workbook; save it as .xlsx without a password"
        )
    suffix = path.suffix.lower()
    if suffix in REQIF_SUFFIXES:
        return "reqif"
    if head.startswith(ZIP_MAGIC):
        return "xlsx"
    if suffix in XLSX_SUFFIXES:
        raise TableError(f"{path} is not a valid .xlsx workbook (it is not a zip file)")
    if head.lstrip().startswith(b"<"):
        return "reqif"
    return "csv"


def _head(path: Path) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(8)
    except FileNotFoundError:
        raise TableError(f"no such file: {path}") from None
    except OSError as exc:
        raise TableError(f"cannot read {path}: {exc.strerror or exc}") from None


def list_sheets(path: str | Path) -> list[SheetInfo]:
    """The sheets of a workbook in order; a CSV file is one sheet named after the file."""
    path = Path(path)
    kind = _kind(path, _head(path))
    if kind == "xlsx":
        from .xlsx_read import workbook_sheets

        return workbook_sheets(path)
    return [SheetInfo(name=path.stem, part="", hidden=False)]


def read_table(path: str | Path, *, sheet: str | int | None = None, encoding: str | None = None,
               delimiter: str | None = None) -> Table:
    """Read a CSV/TSV file, an .xlsx sheet or a ReqIF file as a Table.

    `sheet` is a sheet name (case-insensitive) or a 1-based index; the default is the first visible sheet.
    CSV files are read as UTF-8 (with or without BOM) and fall back to Windows-1252 with a note; the delimiter is
    sniffed among comma, semicolon and tab unless given.
    """
    path = Path(path)
    kind = _kind(path, _head(path))
    if kind == "xlsx":
        from .xlsx_read import read_sheet

        return read_sheet(path, sheet)
    if kind == "reqif":
        from .reqif import read_reqif

        return read_reqif(path)
    return read_csv(path, encoding=encoding, delimiter=delimiter)


def read_csv(path: Path, *, encoding: str | None = None, delimiter: str | None = None) -> Table:
    raw = path.read_bytes()
    issues: list[str] = []
    if encoding is not None:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            raise TableError(f"cannot decode {path} as {encoding}: {exc}") from None
        used = encoding
    else:
        try:
            text, used = raw.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            text, used = raw.decode(FALLBACK_ENCODING, errors="replace"), FALLBACK_ENCODING
            issues.append(f"{path.name} is not UTF-8; it was read as Windows-1252")
    if delimiter is None:
        delimiter = sniff_table_delimiter(text[:SNIFF_BYTES])
    elif delimiter in ("tab", "\\t"):
        delimiter = "\t"
    if len(delimiter) != 1:
        raise TableError(f"a CSV delimiter is one character, got {delimiter!r}")
    rows: list[list[Cell | None]] = []
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        for r, fields in enumerate(reader, start=1):
            stripped = [value.strip() for value in fields]
            rows.append([
                Cell(value=value or None, text=value, ref=f"{column_letter(c)}{r}")
                for c, value in enumerate(stripped, start=1)
            ])
    except csv.Error as exc:
        raise TableError(f"{path.name}: {exc}") from None
    while rows and all(cell is None or cell.is_empty for cell in rows[-1]):
        rows.pop()
    return Table(path=path, sheet=None, rows=rows, format="csv", issues=issues, delimiter=delimiter, encoding=used)


def sniff_table_delimiter(sample: str) -> str:
    """The delimiter that splits the most records into the same number (> 1) of fields; comma when none does.

    Records are read with the csv module, so quoted fields holding delimiters or newlines count correctly. Ties
    go to the earlier candidate (comma, semicolon, tab).
    """
    best_score: tuple[float, int, int] | None = None
    best = ","
    for position, delimiter in enumerate(DELIMITER_CANDIDATES):
        try:
            records = [row for row in csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter) if row]
        except csv.Error:
            continue
        if len(records) > 1 and not sample.endswith(("\n", "\r")):
            records = records[:-1]  # the sample may cut the last record short
        if not records:
            continue
        counts: dict[int, int] = {}
        for row in records:
            counts[len(row)] = counts.get(len(row), 0) + 1
        modal, hits = max(counts.items(), key=lambda item: (item[1], item[0]))
        if modal < 2:
            continue
        score = (hits / len(records), modal, -position)
        if best_score is None or score > best_score:
            best_score, best = score, delimiter
    return best
