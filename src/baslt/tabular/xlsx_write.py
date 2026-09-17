"""Write .xlsx workbooks with the standard library: typed cells, verdict colours, links, frozen headers, filters.

The output is the smallest package Excel, LibreOffice and openpyxl open without complaint: content types,
package and workbook relationships, core and app properties, one styles part, deduplicated shared strings and one
part per sheet. Bytes are deterministic (fixed zip timestamps, fixed part order), so the same results give the same
file.

Cells are `str`, `int`, `float`, `bool`, `None`, `Styled(value, style)` or `Link(text, target)`. Non-finite numbers
are written as text, characters XML cannot hold are escaped the way Excel does, and strings are cut at Excel's
32,767-character limit. A sheet longer than Excel's row limit is truncated with a note in its last row.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
from xml.sax.saxutils import escape, quoteattr

from .._io import write_atomic
from .table import column_letter

__all__ = ["STYLES", "Link", "Sheet", "Styled", "sanitize_sheet_names", "write_xlsx", "xlsx_bytes"]

MAX_ROWS = 1_048_576
MAX_COLS = 16_384
MAX_TEXT = 32_767
MAX_NAME = 31
MAX_WIDTH = 60.0
MIN_WIDTH = 6.0
WIDTH_SAMPLE_ROWS = 1000
ZIP_DATE = (1980, 1, 1, 0, 0, 0)

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
REL_BASE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

# style name -> index into cellXfs (see _styles_xml)
STYLES: dict[str, int] = {
    "default": 0, "bold": 1, "header": 2, "pass": 3, "warn": 4, "fail": 5, "na": 6, "error": 7,
    "pct": 8, "num": 9, "link": 10, "wrap": 11,
}

# XML cannot hold these, and parsers turn a bare carriage return into a newline, so Excel escapes them all.
_ILLEGAL_RE = re.compile("[\x00-\x08\x0b\x0c\r\x0e-\x1f\ufffe\uffff\ud800-\udfff]")
_ESCAPE_LIKE_RE = re.compile(r"_(x[0-9A-Fa-f]{4}_)")
_BAD_NAME_RE = re.compile(r"[\[\]:*?/\\]")


class Styled(NamedTuple):
    value: object
    style: str


class Link(NamedTuple):
    text: str
    target: str  # a relative path or URL, or "#Sheet!A1" inside the workbook


@dataclass
class Sheet:
    name: str
    rows: Sequence[Sequence[object]]
    widths: Sequence[float] | None = None
    freeze: tuple[int, int] = (1, 0)  # rows, columns kept in view
    autofilter: bool = True
    hidden: bool = False


def _text(value: str) -> str:
    value = _ESCAPE_LIKE_RE.sub(r"_x005F_\1", value)
    value = _ILLEGAL_RE.sub(lambda m: f"_x{ord(m.group(0)):04X}_", value)
    if len(value) > MAX_TEXT:
        value = value[: MAX_TEXT - 1] + "…"
    return value


def sanitize_sheet_names(names: Sequence[str]) -> list[str]:
    """Names Excel accepts: no []:*?/\\, at most 31 characters, not empty, unique ignoring case."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = _BAD_NAME_RE.sub("_", str(raw)).strip().strip("'") or "Sheet"
        name = name[:MAX_NAME]
        base, k = name, 2
        while name.casefold() in seen:
            suffix = f" ({k})"
            name = base[: MAX_NAME - len(suffix)] + suffix
            k += 1
        seen.add(name.casefold())
        out.append(name)
    return out


class _Strings:
    def __init__(self) -> None:
        self.index: dict[str, int] = {}
        self.count = 0

    def add(self, text: str) -> int:
        self.count += 1
        found = self.index.get(text)
        if found is None:
            found = self.index[text] = len(self.index)
        return found

    def xml(self) -> str:
        items = []
        for text in self.index:
            space = ' xml:space="preserve"' if text != text.strip() or "\n" in text else ""
            items.append(f"<si><t{space}>{escape(text)}</t></si>")
        return (f'{XML_HEAD}<sst xmlns="{NS_MAIN}" count="{self.count}" uniqueCount="{len(self.index)}">'
                + "".join(items) + "</sst>")


def _number(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _display_length(value: object) -> int:
    if isinstance(value, (Styled, Link)):
        value = value[0]
    if value is None:
        return 0
    if isinstance(value, float):
        return len(_number(value)) if math.isfinite(value) else 4
    text = str(value)
    return max((len(line) for line in text.splitlines()), default=0)


def _sheet_xml(sheet: Sheet, strings: _Strings) -> tuple[str, list[tuple[str, str]], int, int]:
    """The worksheet XML, its external links (rel id, target), and its used rows and columns."""
    rows = list(sheet.rows)
    note = None
    if len(rows) > MAX_ROWS:
        note = f"… {len(rows) - (MAX_ROWS - 1)} more rows are not shown (Excel's row limit)"
        rows = rows[: MAX_ROWS - 1]
    n_cols = min(max((len(row) for row in rows), default=0), MAX_COLS)
    body: list[str] = []
    links: list[tuple[str, str]] = []
    hyperlinks: list[str] = []
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, value in enumerate(list(row)[:MAX_COLS], start=1):
            ref = f"{column_letter(c)}{r}"
            style = 0
            if isinstance(value, Styled):
                style = STYLES.get(value.style, 0)
                value = value.value
            if isinstance(value, Link):
                style = style or STYLES["link"]
                if value.target.startswith("#"):
                    hyperlinks.append(f"<hyperlink ref={quoteattr(ref)} location={quoteattr(value.target[1:])} "
                                      f"display={quoteattr(value.text)}/>")
                else:
                    rid = f"rId{len(links) + 1}"
                    links.append((rid, value.target))
                    hyperlinks.append(f'<hyperlink ref={quoteattr(ref)} r:id="{rid}"/>')
                value = value.text
            s_attr = f' s="{style}"' if style else ""
            if value is None:
                if style:
                    cells.append(f'<c r="{ref}"{s_attr}/>')
                continue
            if isinstance(value, bool):
                cells.append(f'<c r="{ref}"{s_attr} t="b"><v>{int(value)}</v></c>')
            elif isinstance(value, int) and abs(value) < 2**53:
                cells.append(f'<c r="{ref}"{s_attr}><v>{value}</v></c>')
            elif isinstance(value, float) and math.isfinite(value):
                cells.append(f'<c r="{ref}"{s_attr}><v>{_number(value)}</v></c>')
            else:
                if isinstance(value, float):
                    text = "NaN" if math.isnan(value) else ("inf" if value > 0 else "-inf")
                else:
                    text = str(value)
                cells.append(f'<c r="{ref}"{s_attr} t="s"><v>{strings.add(_text(text))}</v></c>')
        if cells:
            body.append(f'<row r="{r}">' + "".join(cells) + "</row>")
    if note is not None:
        r = len(rows) + 1
        body.append(f'<row r="{r}"><c r="A{r}" t="s"><v>{strings.add(note)}</v></c></row>')
    used_rows = len(rows) + (1 if note else 0)

    widths = list(sheet.widths) if sheet.widths is not None else [
        min(MAX_WIDTH, max(MIN_WIDTH, 1.2 * max((_display_length(row[c]) for row in rows[:WIDTH_SAMPLE_ROWS]
                                                  if c < len(row)), default=0) + 2))
        for c in range(n_cols)
    ]
    parts = [f'{XML_HEAD}<worksheet xmlns="{NS_MAIN}" xmlns:r="{NS_REL}">']
    last = f"{column_letter(max(n_cols, 1))}{max(used_rows, 1)}"
    parts.append(f'<dimension ref="A1:{last}"/>')
    freeze_rows, freeze_cols = sheet.freeze
    if (freeze_rows or freeze_cols) and used_rows:
        top_left = f"{column_letter(freeze_cols + 1)}{freeze_rows + 1}"
        pane_name = "bottomRight" if freeze_rows and freeze_cols else ("bottomLeft" if freeze_rows else "topRight")
        split = (f' xSplit="{freeze_cols}"' if freeze_cols else "") + (f' ySplit="{freeze_rows}"' if freeze_rows else "")
        parts.append(f'<sheetViews><sheetView workbookViewId="0"><pane{split} topLeftCell="{top_left}" '
                     f'activePane="{pane_name}" state="frozen"/><selection pane="{pane_name}"/></sheetView></sheetViews>')
    parts.append('<sheetFormatPr defaultRowHeight="15"/>')
    if widths:
        parts.append("<cols>" + "".join(
            f'<col min="{c}" max="{c}" width="{w:.2f}" customWidth="1"/>' for c, w in enumerate(widths, start=1)
        ) + "</cols>")
    parts.append("<sheetData>" + "".join(body) + "</sheetData>")
    if sheet.autofilter and used_rows > 0 and n_cols > 0:
        parts.append(f'<autoFilter ref="A1:{column_letter(n_cols)}{used_rows}"/>')
    if hyperlinks:
        parts.append("<hyperlinks>" + "".join(hyperlinks) + "</hyperlinks>")
    parts.append('<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>')
    parts.append("</worksheet>")
    return "".join(parts), links, used_rows, n_cols


def _styles_xml() -> str:
    fonts = [
        '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>',
        '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font>',
        '<font><u/><sz val="11"/><color rgb="FF0563C1"/><name val="Calibri"/><family val="2"/></font>',
        '<font><sz val="11"/><color rgb="FF006100"/><name val="Calibri"/><family val="2"/></font>',
        '<font><sz val="11"/><color rgb="FF9C5700"/><name val="Calibri"/><family val="2"/></font>',
        '<font><b/><sz val="11"/><color rgb="FF9C0006"/><name val="Calibri"/><family val="2"/></font>',
        '<font><sz val="11"/><color rgb="FF595959"/><name val="Calibri"/><family val="2"/></font>',
        '<font><b/><sz val="11"/><color rgb="FF833C0B"/><name val="Calibri"/><family val="2"/></font>',
    ]

    def solid(rgb: str) -> str:
        return f'<fill><patternFill patternType="solid"><fgColor rgb="FF{rgb}"/><bgColor indexed="64"/></patternFill></fill>'

    fills = [
        '<fill><patternFill patternType="none"/></fill>',
        '<fill><patternFill patternType="gray125"/></fill>',
        solid("D9E1F2"), solid("C6EFCE"), solid("FFEB9C"), solid("FFC7CE"), solid("EDEDED"), solid("F8CBAD"),
    ]
    # (font, fill, numFmt, wrap) per style index, in STYLES order
    xfs = [(0, 0, 0, False), (1, 0, 0, False), (1, 2, 0, False), (3, 3, 0, False), (4, 4, 0, False),
           (5, 5, 0, False), (6, 6, 0, False), (7, 7, 0, False), (0, 0, 10, False), (0, 0, 164, False),
           (2, 0, 0, False), (0, 0, 0, True)]
    xf_xml = []
    for font, fill, fmt, wrap in xfs:
        attrs = f'numFmtId="{fmt}" fontId="{font}" fillId="{fill}" borderId="0" xfId="0"'
        attrs += (' applyFont="1"' if font else "") + (' applyFill="1"' if fill else "")
        attrs += ' applyNumberFormat="1"' if fmt else ""
        if wrap:
            xf_xml.append(f'<xf {attrs} applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>')
        else:
            xf_xml.append(f"<xf {attrs}/>")
    return (
        f'{XML_HEAD}<styleSheet xmlns="{NS_MAIN}">'
        '<numFmts count="1"><numFmt numFmtId="164" formatCode="0.000###"/></numFmts>'
        f'<fonts count="{len(fonts)}">{"".join(fonts)}</fonts>'
        f'<fills count="{len(fills)}">{"".join(fills)}</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        f'<cellXfs count="{len(xf_xml)}">{"".join(xf_xml)}</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '<dxfs count="0"/><tableStyles count="0" defaultTableStyle="TableStyleMedium2" '
        'defaultPivotStyle="PivotStyleLight16"/></styleSheet>'
    )


def xlsx_bytes(sheets: Sequence[Sheet], *, creator: str = "baslt", title: str | None = None) -> bytes:
    """The workbook as bytes. At least one sheet is required; the first visible sheet is shown on opening."""
    if not sheets:
        raise ValueError("a workbook needs at least one sheet")
    names = sanitize_sheet_names([sheet.name for sheet in sheets])
    if all(sheet.hidden for sheet in sheets):
        raise ValueError("a workbook needs at least one visible sheet")
    active = next(i for i, sheet in enumerate(sheets) if not sheet.hidden)
    strings = _Strings()
    files: list[tuple[str, str]] = []
    sheet_entries = []
    defined = []
    content = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    workbook_rels = []
    for k, (sheet, name) in enumerate(zip(sheets, names), start=1):
        xml, links, used_rows, n_cols = _sheet_xml(sheet, strings)
        files.append((f"xl/worksheets/sheet{k}.xml", xml))
        if links:
            rels = "".join(
                f'<Relationship Id="{rid}" Type="{REL_BASE}/hyperlink" Target={quoteattr(target)} '
                'TargetMode="External"/>' for rid, target in links
            )
            files.append((f"xl/worksheets/_rels/sheet{k}.xml.rels",
                          f'{XML_HEAD}<Relationships xmlns="{NS_PKG_REL}">{rels}</Relationships>'))
        state = ' state="hidden"' if sheet.hidden else ""
        sheet_entries.append(f'<sheet name={quoteattr(name)} sheetId="{k}"{state} r:id="rId{k}"/>')
        workbook_rels.append(f'<Relationship Id="rId{k}" Type="{REL_BASE}/worksheet" '
                             f'Target="worksheets/sheet{k}.xml"/>')
        content.append(f'<Override PartName="/xl/worksheets/sheet{k}.xml" '
                       'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        if sheet.autofilter and used_rows > 0 and n_cols > 0:
            quoted = "'" + name.replace("'", "''") + "'"
            defined.append(f'<definedName name="_xlnm._FilterDatabase" localSheetId="{k - 1}" hidden="1">'
                           f"{escape(quoted)}!$A$1:${column_letter(n_cols)}${used_rows}</definedName>")
    n = len(sheets)
    workbook_rels.append(f'<Relationship Id="rId{n + 1}" Type="{REL_BASE}/styles" Target="styles.xml"/>')
    workbook_rels.append(f'<Relationship Id="rId{n + 2}" Type="{REL_BASE}/sharedStrings" Target="sharedStrings.xml"/>')
    content += [
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>',
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    workbook = (
        f'{XML_HEAD}<workbook xmlns="{NS_MAIN}" xmlns:r="{NS_REL}">'
        f'<bookViews><workbookView activeTab="{active}"/></bookViews>'
        f'<sheets>{"".join(sheet_entries)}</sheets>'
        + (f'<definedNames>{"".join(defined)}</definedNames>' if defined else "")
        + "</workbook>"
    )
    title_xml = f"<dc:title>{escape(title)}</dc:title>" if title else ""
    core = (
        f'{XML_HEAD}<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"{title_xml}<dc:creator>{escape(creator)}</dc:creator></cp:coreProperties>"
    )
    app = (
        f'{XML_HEAD}<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
        f"<Application>{escape(creator)}</Application></Properties>"
    )
    package_rels = (
        f'{XML_HEAD}<Relationships xmlns="{NS_PKG_REL}">'
        f'<Relationship Id="rId1" Type="{REL_BASE}/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/'
        'core-properties" Target="docProps/core.xml"/>'
        f'<Relationship Id="rId3" Type="{REL_BASE}/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )
    parts = [
        ("[Content_Types].xml", f'{XML_HEAD}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                                + "".join(content) + "</Types>"),
        ("_rels/.rels", package_rels),
        ("docProps/core.xml", core),
        ("docProps/app.xml", app),
        ("xl/workbook.xml", workbook),
        ("xl/_rels/workbook.xml.rels",
         f'{XML_HEAD}<Relationships xmlns="{NS_PKG_REL}">{"".join(workbook_rels)}</Relationships>'),
        ("xl/styles.xml", _styles_xml()),
        *files,
        ("xl/sharedStrings.xml", strings.xml()),
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in parts:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, text.encode("utf-8"))
    return buffer.getvalue()


def write_xlsx(path: str | Path, sheets: Sequence[Sheet], *, creator: str = "baslt", title: str | None = None) -> Path:
    """Write the workbook atomically and return its path."""
    path = Path(path)
    write_atomic(path, xlsx_bytes(sheets, creator=creator, title=title))
    return path
