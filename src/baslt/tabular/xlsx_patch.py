"""Write values into an existing .xlsx sheet without disturbing anything else in the workbook.

Re-serializing a sheet with an XML library renames namespace prefixes and drops declarations that
`mc:Ignorable` depends on, which makes Excel offer to "repair" the file, and round-trip exports (Polarion's among
them) carry metadata that must survive untouched. So the patcher splices bytes instead:

- expat reports where every row and cell of the sheet starts and ends;
- edited cells are replaced in place (keeping their style), new cells and rows are inserted in order as inline
  strings or numbers, `spans` hints are dropped from edited rows and the `dimension` grows to cover the edits;
- every other byte of the sheet, and every other part of the package, is copied unchanged.

Cells holding a formula are never overwritten (the workbook's calculation chain would point at a formula that no
longer exists); they are reported instead.
"""

from __future__ import annotations

import io
import math
import re
import xml.parsers.expat
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from .._io import write_atomic
from ..errors import TableError
from .table import column_letter, split_ref
from .xlsx_read import open_workbook
from .xlsx_write import _number, _text

__all__ = ["CellEdit", "patch_sheet_xml", "patch_xlsx"]

_SPANS_KEYS = ("spans",)
_REF_RE = re.compile(r"^([A-Z]+\d+)(?::([A-Z]+\d+))?$")


@dataclass(frozen=True)
class CellEdit:
    row: int  # 1-based
    col: int  # 1-based
    value: str | float | int | bool | None


@dataclass
class _Element:
    qname: str
    attrs: list[tuple[str, str]]
    start: int
    tag_end: int
    self_closing: bool
    end: int = -1  # exclusive end of the whole element
    close_start: int = -1  # start of the end tag (-1 when self-closing)
    children: list[_Element] = field(default_factory=list)
    has_formula: bool = False
    number: int = 0  # row number or column number

    def get(self, key: str) -> str | None:
        for name, value in self.attrs:
            if name == key:
                return value
        return None

    @property
    def prefix(self) -> str:
        return self.qname[: self.qname.index(":") + 1] if ":" in self.qname else ""


def _tag_end(data: bytes, start: int) -> int:
    """Index just past the '>' that closes the start tag at `start`, skipping quoted attribute values."""
    quote = 0
    i = start
    n = len(data)
    while i < n:
        byte = data[i]
        if quote:
            if byte == quote:
                quote = 0
        elif byte in (0x22, 0x27):
            quote = byte
        elif byte == 0x3E:
            return i + 1
        i += 1
    raise TableError("the sheet XML ends inside a tag")


def _scan(data: bytes) -> tuple[_Element | None, _Element | None, list[_Element]]:
    """The dimension element, the sheetData element and its rows (with their cells)."""
    parser = xml.parsers.expat.ParserCreate()
    parser.ordered_attributes = True
    stack: list[_Element] = []
    dimension: _Element | None = None
    sheet_data: _Element | None = None
    rows: list[_Element] = []

    def start(name: str, attrs: list[str]) -> None:
        nonlocal dimension, sheet_data
        pos = parser.CurrentByteIndex
        end = _tag_end(data, pos)
        element = _Element(name, list(zip(attrs[0::2], attrs[1::2])), pos, end, data[end - 2:end] == b"/>")
        local = name.rsplit(":", 1)[-1]
        depth = len(stack)
        if depth == 1 and local == "dimension":
            dimension = element
        elif depth == 1 and local == "sheetData":
            sheet_data = element
        elif depth == 2 and local == "row" and stack[-1] is sheet_data:
            rows.append(element)
        elif depth == 3 and local == "c" and rows and stack[-1] is rows[-1]:
            rows[-1].children.append(element)
        elif depth == 4 and local == "f" and rows and rows[-1].children and stack[3] is rows[-1].children[-1]:
            rows[-1].children[-1].has_formula = True
        stack.append(element)

    def end(name: str) -> None:
        element = stack.pop()
        if element.self_closing:
            element.end = element.tag_end
        else:
            element.close_start = parser.CurrentByteIndex
            element.end = data.index(b">", element.close_start) + 1

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    try:
        parser.Parse(data, True)
    except xml.parsers.expat.ExpatError as exc:
        raise TableError(f"the sheet XML is not well-formed: {exc}") from None
    return dimension, sheet_data, rows


def _start_tag(element: _Element, *, drop: Sequence[str] = (), set_attrs: Sequence[tuple[str, str]] = (),
               self_closing: bool | None = None) -> str:
    attrs = [(k, v) for k, v in element.attrs if k not in drop]
    for key, value in set_attrs:
        for i, (k, _) in enumerate(attrs):
            if k == key:
                attrs[i] = (key, value)
                break
        else:
            if key == "r":
                attrs.insert(0, (key, value))  # where Excel writes it
            else:
                attrs.append((key, value))
    text = "".join(f" {k}={quoteattr(v)}" for k, v in attrs)
    closing = element.self_closing if self_closing is None else self_closing
    return f"<{element.qname}{text}{'/>' if closing else '>'}"


def _cell_xml(prefix: str, ref: str, value: object, style: str | None) -> str:
    s_attr = f' s="{style}"' if style else ""
    c = f"{prefix}c"
    if value is None:
        return f'<{c} r="{ref}"{s_attr}/>'
    if isinstance(value, bool):
        return f'<{c} r="{ref}"{s_attr} t="b"><{prefix}v>{int(value)}</{prefix}v></{c}>'
    if isinstance(value, int) and abs(value) < 2**53:
        return f'<{c} r="{ref}"{s_attr}><{prefix}v>{value}</{prefix}v></{c}>'
    if isinstance(value, float) and math.isfinite(value):
        return f'<{c} r="{ref}"{s_attr}><{prefix}v>{_number(value)}</{prefix}v></{c}>'
    text = _text(str(value) if not isinstance(value, float) else repr(value))
    return (f'<{c} r="{ref}"{s_attr} t="inlineStr"><{prefix}is><{prefix}t xml:space="preserve">'
            f"{escape(text)}</{prefix}t></{prefix}is></{c}>")


def patch_sheet_xml(data: bytes, edits: Sequence[CellEdit]) -> tuple[bytes, list[str]]:
    """The sheet XML with `edits` applied, and notes about edits that were not applied."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise TableError("UTF-16 sheet XML is not supported")
    wanted: dict[int, dict[int, object]] = {}
    for edit in edits:
        if edit.row < 1 or edit.col < 1:
            raise ValueError(f"cell positions start at 1, got row {edit.row} column {edit.col}")
        wanted.setdefault(edit.row, {})[edit.col] = edit.value
    notes: list[str] = []
    if not wanted:
        return data, notes

    dimension, sheet_data, rows = _scan(data)
    if sheet_data is None:
        raise TableError("the sheet has no sheetData element")

    splices: list[tuple[int, int, str]] = []
    row_number = 0
    prefix = sheet_data.prefix
    by_number: dict[int, _Element] = {}
    for row in rows:
        given = row.get("r")
        row.number = int(given) if given else row_number + 1
        row_number = row.number
        by_number[row.number] = row
        col_number = 0
        for cell in row.children:
            ref = cell.get("r")
            cell.number = split_ref(ref)[1] if ref else col_number + 1
            col_number = cell.number
        if not given and row.number not in wanted:
            splices.append((row.start, row.tag_end, _start_tag(row, set_attrs=[("r", str(row.number))])))

    max_row = max(wanted)
    max_col = max(max(cols) for cols in wanted.values())

    for number, cols in sorted(wanted.items()):
        row = by_number.get(number)
        if row is None:
            continue
        cells = {cell.number: cell for cell in row.children}
        inserts: dict[int, list[str]] = {}  # byte position -> new cells in column order
        for col, value in sorted(cols.items()):
            ref = f"{column_letter(col)}{number}"
            existing = cells.get(col)
            if existing is not None:
                if existing.has_formula:
                    notes.append(f"{ref} holds a formula and was left unchanged")
                    continue
                splices.append((existing.start, existing.end,
                                _cell_xml(existing.prefix or row.prefix, ref, value, existing.get("s"))))
                continue
            later = [cell for cell in row.children if cell.number > col]
            position = later[0].start if later else (row.close_start if not row.self_closing else row.end)
            inserts.setdefault(position, []).append(_cell_xml(row.prefix, ref, value, None))
        for cell in row.children:
            if cell.get("r") is None and cell.number not in cols:
                ref = f"{column_letter(cell.number)}{number}"
                splices.append((cell.start, cell.tag_end, _start_tag(cell, set_attrs=[("r", ref)])))
        if row.self_closing:
            added = "".join(text for _, texts in sorted(inserts.items()) for text in texts)
            new_row = (_start_tag(row, drop=_SPANS_KEYS, set_attrs=[("r", str(number))], self_closing=False)
                       + added + f"</{row.qname}>")
            splices.append((row.start, row.end, new_row))
        else:
            splices.append((row.start, row.tag_end,
                            _start_tag(row, drop=_SPANS_KEYS, set_attrs=[("r", str(number))])))
            for position, texts in inserts.items():
                splices.append((position, position, "".join(texts)))

    new_rows = {number: cols for number, cols in wanted.items() if number not in by_number}
    if new_rows:
        row_q = f"{prefix}row"
        grouped: dict[int, list[str]] = {}
        for number, cols in sorted(new_rows.items()):
            text = f'<{row_q} r="{number}">' + "".join(
                _cell_xml(prefix, f"{column_letter(col)}{number}", value, None) for col, value in sorted(cols.items())
            ) + f"</{row_q}>"
            later = [row for row in rows if row.number > number]
            position = later[0].start if later else (
                sheet_data.close_start if not sheet_data.self_closing else -1)
            grouped.setdefault(position, []).append(text)
        for position, texts in grouped.items():
            if position == -1:
                splices.append((sheet_data.start, sheet_data.end,
                                _start_tag(sheet_data, self_closing=False) + "".join(texts)
                                + f"</{sheet_data.qname}>"))
            else:
                splices.append((position, position, "".join(texts)))

    if dimension is not None:
        match = _REF_RE.match(dimension.get("ref") or "")
        if match is not None:
            last = match.group(2) or match.group(1)
            last_row, last_col = split_ref(last)
            first = match.group(1)
            new_last = f"{column_letter(max(last_col, max_col))}{max(last_row, max_row)}"
            if new_last != last or match.group(2) is None:
                splices.append((dimension.start, dimension.tag_end,
                                _start_tag(dimension, set_attrs=[("ref", f"{first}:{new_last}")])))

    out = bytearray(data)
    # Replacements never overlap; insertions at a position go after a replacement ending there.
    for start, end, text in sorted(splices, key=lambda item: (item[0], item[1]), reverse=True):
        out[start:end] = text.encode("utf-8")
    return bytes(out), notes


def patch_xlsx(src: str | Path, dst: str | Path, *, sheet: str | int | None,
               edits: Sequence[CellEdit]) -> list[str]:
    """Copy `src` to `dst` with `edits` written into `sheet`; returns notes about edits that were skipped."""
    src, dst = Path(src), Path(dst)
    with open_workbook(src) as book:
        part = book.find(sheet)
        patched, notes = patch_sheet_xml(book.read(part.part), edits)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for info in book.zip.infolist():
                data = patched if info.filename == part.part else book.zip.read(info)
                copy = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                copy.compress_type = info.compress_type
                copy.external_attr = info.external_attr
                copy.create_system = info.create_system
                copy.comment = info.comment
                copy.extra = b""
                archive.writestr(copy, data)
            archive.comment = book.zip.comment
    write_atomic(dst, buffer.getvalue())
    return notes
