"""ReqIF (OMG Requirements Interchange Format) exports as tables, for Polarion, DOORS and other tools.

One row per SPEC-OBJECT in the order of the specification hierarchy (objects no specification lists follow in file
order), one column per attribute long name in the order the object types define them, plus `ReqIF Type` (the
object's type, such as Heading or Requirement) and `ReqIF Level` (its depth in the hierarchy). XHTML values become
plain text, enumerations their long names (joined with ", " when several are chosen), numbers and booleans their
values. `.reqifz` archives are read from their first .reqif member.

Like the .xlsx reader this refuses DOCTYPE declarations and caps sizes, so a hostile file cannot expand entities or
exhaust memory.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from ..errors import TableError
from .table import Cell, Table

__all__ = ["read_reqif"]

MAX_BYTES = 256 * 1024 * 1024
MAX_RATIO = 200
BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "pre"}
EXTRA_COLUMNS = ("ReqIF Type", "ReqIF Level")
MAX_DEPTH = 256


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element, name: str):
    for child in element:
        if _local(child.tag) == name:
            return child
    return None


def _ref(element, container: str) -> str | None:
    """The text of the single *-REF element inside `container` (e.g. TYPE or DEFINITION)."""
    holder = _child(element, container)
    if holder is None:
        return None
    for child in holder:
        if _local(child.tag).endswith("-REF"):
            return (child.text or "").strip()
    return None


def _xhtml_text(element) -> str:
    parts: list[str] = []

    def walk(node) -> None:
        name = _local(node.tag)
        if name == "br":
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
        if name in BLOCKS and name != "br":
            parts.append("\n")
        if node.tail:
            parts.append(node.tail)

    for child in element:
        walk(child)
    lines = [" ".join(line.split()) for line in "".join(parts).splitlines()]
    return "\n".join(line for line in lines if line)


def _read_bytes(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise TableError(f"cannot read {path}: {exc.strerror or exc}") from None
    if size > MAX_BYTES:
        raise TableError(f"{path.name} is larger than {MAX_BYTES // (1024 * 1024)} MiB")
    data = path.read_bytes()
    if not data.startswith(b"PK\x03\x04"):
        return data
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise TableError(f"{path.name} is not a valid .reqifz archive: {exc}") from None
    members = [info for info in archive.infolist() if info.filename.lower().endswith(".reqif")]
    if not members:
        raise TableError(f"{path.name} holds no .reqif file")
    info = members[0]
    if info.file_size > MAX_BYTES or info.file_size > MAX_RATIO * max(info.compress_size, 1):
        raise TableError(f"{path.name}: {info.filename} expands too much to be read safely")
    return archive.read(info)


def read_reqif(path: str | Path) -> Table:
    path = Path(path)
    data = _read_bytes(path)
    if b"<!DOCTYPE" in data[:4096] or b"<!ENTITY" in data:
        raise TableError(f"{path.name} declares a DOCTYPE, which is not allowed")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise TableError(f"{path.name} is not valid ReqIF XML: {exc}") from None
    if _local(root.tag) != "REQ-IF":
        raise TableError(f"{path.name} is not a ReqIF file (its root element is {_local(root.tag)})")
    content = None
    for element in root.iter():
        if _local(element.tag) == "REQ-IF-CONTENT":
            content = element
            break
    if content is None:
        raise TableError(f"{path.name} has no REQ-IF-CONTENT")

    enum_names: dict[str, str] = {}
    datatypes = _child(content, "DATATYPES")
    for definition in (datatypes if datatypes is not None else []):
        for value in definition.iter():
            if _local(value.tag) == "ENUM-VALUE":
                enum_names[value.get("IDENTIFIER", "")] = value.get("LONG-NAME") or value.get("IDENTIFIER", "")

    attribute_names: dict[str, str] = {}
    type_names: dict[str, str] = {}
    columns: list[str] = []
    types = _child(content, "SPEC-TYPES")
    for spec_type in (types if types is not None else []):
        type_names[spec_type.get("IDENTIFIER", "")] = spec_type.get("LONG-NAME") or spec_type.get("IDENTIFIER", "")
        if _local(spec_type.tag) != "SPEC-OBJECT-TYPE":
            continue
        attributes = _child(spec_type, "SPEC-ATTRIBUTES")
        for attribute in (attributes if attributes is not None else []):
            ident = attribute.get("IDENTIFIER", "")
            name = attribute.get("LONG-NAME") or ident
            attribute_names[ident] = name
            if name not in columns:
                columns.append(name)

    objects: dict[str, dict] = {}
    order: list[str] = []
    spec_objects = _child(content, "SPEC-OBJECTS")
    for obj in (spec_objects if spec_objects is not None else []):
        ident = obj.get("IDENTIFIER", "")
        values: dict[str, str] = {}
        holder = _child(obj, "VALUES")
        for value in (holder if holder is not None else []):
            kind = _local(value.tag)
            definition = _ref(value, "DEFINITION")
            name = attribute_names.get(definition or "", definition or "")
            if not name:
                continue
            if kind == "ATTRIBUTE-VALUE-XHTML":
                inner = _child(value, "THE-VALUE")
                values[name] = _xhtml_text(inner) if inner is not None else ""
            elif kind == "ATTRIBUTE-VALUE-ENUMERATION":
                chosen = _child(value, "VALUES")
                refs = [(child.text or "").strip() for child in (chosen if chosen is not None else [])]
                values[name] = ", ".join(enum_names.get(r, r) for r in refs)
            else:
                values[name] = value.get("THE-VALUE", "")
        objects[ident] = {"values": values, "type": type_names.get(_ref(obj, "TYPE") or "", ""), "level": None}
        order.append(ident)

    listed: list[str] = []
    specifications = _child(content, "SPECIFICATIONS")

    def walk(children, level: int) -> None:
        if level > MAX_DEPTH:
            raise TableError(f"{path.name}: the specification hierarchy is nested more than {MAX_DEPTH} levels")
        for hierarchy in (children if children is not None else []):
            target = _ref(hierarchy, "OBJECT")
            if target in objects and objects[target]["level"] is None:
                objects[target]["level"] = level
                listed.append(target)
            walk(_child(hierarchy, "CHILDREN"), level + 1)

    for specification in (specifications if specifications is not None else []):
        walk(_child(specification, "CHILDREN"), 1)
    listed += [ident for ident in order if objects[ident]["level"] is None]

    header = [*columns, *EXTRA_COLUMNS]
    rows: list[list[Cell | None]] = [[Cell(name, name, f"header/{name}") for name in header]]
    labels = ["header"]
    for ident in listed:
        obj = objects[ident]
        row: list[Cell | None] = []
        for name in columns:
            text = obj["values"].get(name)
            row.append(None if text is None else Cell(text, text, f"{ident}/{name}"))
        level = obj["level"]
        row.append(Cell(obj["type"], obj["type"], f"{ident}/type"))
        row.append(Cell(float(level), str(level), f"{ident}/level") if level is not None else None)
        rows.append(row)
        labels.append(obj["values"].get("ReqIF.ForeignID") or ident)
    return Table(path=path, sheet=None, rows=rows, format="reqif", row_labels=labels)
