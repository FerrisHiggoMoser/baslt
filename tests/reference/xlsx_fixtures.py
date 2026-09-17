"""Hand-written .xlsx packages covering what real workbooks contain, built part by part.

Every fixture is a zip of XML text written here, not by a spreadsheet library, so the reader is tested against
the file format rather than against another implementation of it.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
STRICT_MAIN = "http://purl.oclc.org/ooxml/spreadsheetml/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
STRICT_REL = "http://purl.oclc.org/ooxml/officeDocument/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

CONTENT_TYPES = (
    HEAD + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/></Types>'
)


def package_rels(target: str = "xl/workbook.xml", rel: str = REL) -> str:
    return (f'{HEAD}<Relationships xmlns="{PKG}"><Relationship Id="rId1" Type="{rel}/officeDocument" '
            f'Target="{target}"/></Relationships>')


def workbook(sheets: list[tuple[str, str]], *, main: str = MAIN, rel: str = REL, date1904: bool = False,
             prefix: str = "") -> str:
    """`sheets` is [(name, state)]; sheet k (1-based) uses relationship rId{k}."""
    p = f"{prefix}:" if prefix else ""
    ns = f'xmlns:{prefix}="{main}"' if prefix else f'xmlns="{main}"'
    pr = f'<{p}workbookPr date1904="1"/>' if date1904 else ""
    entries = "".join(
        f'<{p}sheet name="{name}" sheetId="{k}"{"" if state == "visible" else f" state={chr(34)}{state}{chr(34)}"} '
        f'r:id="rId{k}"/>'
        for k, (name, state) in enumerate(sheets, start=1)
    )
    return f'{HEAD}<{p}workbook {ns} xmlns:r="{rel}">{pr}<{p}sheets>{entries}</{p}sheets></{p}workbook>'


def workbook_rels(targets: list[tuple[str, str]], *, extra: list[tuple[str, str]] = (), rel: str = REL) -> str:
    """`targets` is [(type suffix, target)] for rId1.., then `extra` for rId100.."""
    items = [f'<Relationship Id="rId{k}" Type="{rel}/{kind}" Target="{target}"/>'
             for k, (kind, target) in enumerate(targets, start=1)]
    items += [f'<Relationship Id="rId{100 + k}" Type="{rel}/{kind}" Target="{target}"/>'
              for k, (kind, target) in enumerate(extra)]
    return f'{HEAD}<Relationships xmlns="{PKG}">{"".join(items)}</Relationships>'


def sheet(rows_xml: str, *, main: str = MAIN, extra: str = "", prefix: str = "", head: str = "") -> str:
    p = f"{prefix}:" if prefix else ""
    ns = f'xmlns:{prefix}="{main}"' if prefix else f'xmlns="{main}"'
    return f"{HEAD}<{p}worksheet {ns} xmlns:r=\"{REL}\">{head}<{p}sheetData>{rows_xml}</{p}sheetData>{extra}</{p}worksheet>"


def shared_strings(items: list[str]) -> str:
    return f'{HEAD}<sst xmlns="{MAIN}" count="{len(items)}" uniqueCount="{len(items)}">{"".join(items)}</sst>'


def styles(xfs: list[int], custom: dict[int, str] | None = None) -> str:
    fmts = "".join(f'<numFmt numFmtId="{k}" formatCode={quoteattr(code)}/>' for k, code in (custom or {}).items())
    cell_xfs = "".join(f'<xf numFmtId="{fmt}" fontId="0" fillId="0" borderId="0" xfId="0"/>' for fmt in xfs)
    return (f'{HEAD}<styleSheet xmlns="{MAIN}"><numFmts count="{len(custom or {})}">{fmts}</numFmts>'
            f'<cellXfs count="{len(xfs)}">{cell_xfs}</cellXfs></styleSheet>')


OVERRIDES = {
    "xl/workbook.xml": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
    "xl/sharedStrings.xml": "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml",
    "xl/styles.xml": "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml",
}
SHEET_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"


def content_types(names) -> str:
    """Content types naming the workbook, sheet, string and style parts present, as real files do."""
    items = []
    for name in names:
        kind = OVERRIDES.get(name) or (SHEET_TYPE if name.startswith("xl/worksheets/sheet") else None)
        if kind is not None:
            items.append(f'<Override PartName="/{name}" ContentType="{kind}"/>')
    return CONTENT_TYPES.replace("</Types>", "".join(items) + "</Types>")


def build(path: Path, parts: dict[str, str | bytes]) -> Path:
    """Zip the parts; a `[Content_Types].xml` equal to the bare CONTENT_TYPES gets overrides for what is present."""
    if parts.get("[Content_Types].xml") == CONTENT_TYPES:
        parts = {**parts, "[Content_Types].xml": content_types(parts)}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data.encode("utf-8") if isinstance(data, str) else data)
    return path


def simple(path: Path, rows_xml: str, *, strings: list[str] | None = None, xfs: list[int] | None = None,
           custom: dict[int, str] | None = None, sheet_extra: str = "", date1904: bool = False) -> Path:
    """One visible sheet named 'Requirements' with optional shared strings and styles."""
    targets = [("worksheet", "worksheets/sheet1.xml")]
    extra = []
    parts: dict[str, str | bytes] = {"[Content_Types].xml": CONTENT_TYPES, "_rels/.rels": package_rels()}
    if strings is not None:
        extra.append(("sharedStrings", "sharedStrings.xml"))
        parts["xl/sharedStrings.xml"] = shared_strings(strings)
    if xfs is not None:
        extra.append(("styles", "styles.xml"))
        parts["xl/styles.xml"] = styles(xfs, custom)
    parts["xl/workbook.xml"] = workbook([("Requirements", "visible")], date1904=date1904)
    parts["xl/_rels/workbook.xml.rels"] = workbook_rels(targets, extra=extra)
    parts["xl/worksheets/sheet1.xml"] = sheet(rows_xml, extra=sheet_extra)
    return build(path, parts)
