"""ReqIF files for tests, shaped like a Polarion export: string, XHTML, enumeration and number attributes, a
specification hierarchy with headings, and an optional .reqifz archive."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

NS = 'xmlns="http://www.omg.org/spec/ReqIF/20110401/reqif.xsd" xmlns:xhtml="http://www.w3.org/1999/xhtml"'

# (identifier, type, level, {attribute: value}); XHTML values are given as markup
OBJECTS = [
    ("obj-h1", "Heading", 1, {"ReqIF.ChapterName": "Structural loads"}),
    ("obj-1", "Requirement", 2, {"ReqIF.ForeignID": "LV-001", "ReqIF.Name": "Max dynamic pressure",
                                 "ReqIF.Text": "<xhtml:div><xhtml:p>The <xhtml:b>dynamic pressure</xhtml:b> "
                                               "shall not exceed</xhtml:p><xhtml:p>70 kPa.</xhtml:p></xhtml:div>",
                                 "Check": "q", "Limit": "<= 70 kPa", "Status": ["approved"], "Priority": "2"}),
    ("obj-2", "Requirement", 2, {"ReqIF.ForeignID": "LV-002", "ReqIF.Name": "AoA", "Check": "alpha",
                                 "Limit": "[-7, 7] deg", "Status": ["draft"]}),
    ("obj-3", "Requirement", 3, {"ReqIF.ForeignID": "LV-003", "ReqIF.Name": "Prose", "Status": ["approved",
                                                                                                 "reviewed"]}),
]
ORPHAN = ("obj-9", "Requirement", None, {"ReqIF.ForeignID": "LV-009", "ReqIF.Name": "Not in a specification",
                                         "Check": "q", "Limit": "<= 100 kPa", "Status": ["approved"]})
STRINGS = ("ReqIF.ForeignID", "ReqIF.Name", "ReqIF.ChapterName", "Check", "Limit")


def reqif_text(objects=OBJECTS, orphan=ORPHAN, *, doctype: str = "") -> str:
    attrs = []
    for name in STRINGS:
        attrs.append(f'<ATTRIBUTE-DEFINITION-STRING IDENTIFIER="ad-{name}" LONG-NAME={quoteattr(name)}>'
                     '<TYPE><DATATYPE-DEFINITION-STRING-REF>dt-string</DATATYPE-DEFINITION-STRING-REF></TYPE>'
                     '</ATTRIBUTE-DEFINITION-STRING>')
    attrs.append('<ATTRIBUTE-DEFINITION-XHTML IDENTIFIER="ad-text" LONG-NAME="ReqIF.Text">'
                 '<TYPE><DATATYPE-DEFINITION-XHTML-REF>dt-xhtml</DATATYPE-DEFINITION-XHTML-REF></TYPE>'
                 '</ATTRIBUTE-DEFINITION-XHTML>')
    attrs.append('<ATTRIBUTE-DEFINITION-ENUMERATION IDENTIFIER="ad-status" LONG-NAME="Status" MULTI-VALUED="true">'
                 '<TYPE><DATATYPE-DEFINITION-ENUMERATION-REF>dt-status</DATATYPE-DEFINITION-ENUMERATION-REF></TYPE>'
                 '</ATTRIBUTE-DEFINITION-ENUMERATION>')
    attrs.append('<ATTRIBUTE-DEFINITION-INTEGER IDENTIFIER="ad-priority" LONG-NAME="Priority">'
                 '<TYPE><DATATYPE-DEFINITION-INTEGER-REF>dt-int</DATATYPE-DEFINITION-INTEGER-REF></TYPE>'
                 '</ATTRIBUTE-DEFINITION-INTEGER>')
    enum_values = "".join(f'<ENUM-VALUE IDENTIFIER="ev-{v}" LONG-NAME="{v}"><PROPERTIES><EMBEDDED-VALUE KEY="{k}" '
                          'OTHER-CONTENT=""/></PROPERTIES></ENUM-VALUE>'
                          for k, v in enumerate(("draft", "approved", "reviewed")))
    body = []
    for ident, kind, _, values in [*objects, *([orphan] if orphan else [])]:
        items = []
        for name, value in values.items():
            if name == "ReqIF.Text":
                items.append('<ATTRIBUTE-VALUE-XHTML><DEFINITION><ATTRIBUTE-DEFINITION-XHTML-REF>ad-text'
                             f'</ATTRIBUTE-DEFINITION-XHTML-REF></DEFINITION><THE-VALUE>{value}</THE-VALUE>'
                             '</ATTRIBUTE-VALUE-XHTML>')
            elif name == "Status":
                refs = "".join(f"<ENUM-VALUE-REF>ev-{v}</ENUM-VALUE-REF>" for v in value)
                items.append('<ATTRIBUTE-VALUE-ENUMERATION><DEFINITION><ATTRIBUTE-DEFINITION-ENUMERATION-REF>'
                             'ad-status</ATTRIBUTE-DEFINITION-ENUMERATION-REF></DEFINITION>'
                             f'<VALUES>{refs}</VALUES></ATTRIBUTE-VALUE-ENUMERATION>')
            elif name == "Priority":
                items.append(f'<ATTRIBUTE-VALUE-INTEGER THE-VALUE="{value}"><DEFINITION>'
                             '<ATTRIBUTE-DEFINITION-INTEGER-REF>ad-priority</ATTRIBUTE-DEFINITION-INTEGER-REF>'
                             '</DEFINITION></ATTRIBUTE-VALUE-INTEGER>')
            else:
                items.append(f'<ATTRIBUTE-VALUE-STRING THE-VALUE={quoteattr(value)}><DEFINITION>'
                             f'<ATTRIBUTE-DEFINITION-STRING-REF>ad-{escape(name)}</ATTRIBUTE-DEFINITION-STRING-REF>'
                             '</DEFINITION></ATTRIBUTE-VALUE-STRING>')
        body.append(f'<SPEC-OBJECT IDENTIFIER="{ident}" LAST-CHANGE="2026-01-01T00:00:00Z"><TYPE>'
                    f'<SPEC-OBJECT-TYPE-REF>st-{kind}</SPEC-OBJECT-TYPE-REF></TYPE><VALUES>{"".join(items)}'
                    '</VALUES></SPEC-OBJECT>')

    def hierarchy(index: int) -> tuple[str, int]:
        ident, _, level, _ = objects[index]
        children = []
        k = index + 1
        while k < len(objects) and objects[k][2] > level:
            if objects[k][2] == level + 1:
                text, k = hierarchy(k)
                children.append(text)
            else:
                k += 1
        inner = f"<CHILDREN>{''.join(children)}</CHILDREN>" if children else ""
        return (f'<SPEC-HIERARCHY IDENTIFIER="sh-{ident}"><OBJECT><SPEC-OBJECT-REF>{ident}</SPEC-OBJECT-REF>'
                f'</OBJECT>{inner}</SPEC-HIERARCHY>'), k

    tops, k = [], 0
    while k < len(objects):
        text, k = hierarchy(k)
        tops.append(text)
    types = "".join(f'<SPEC-OBJECT-TYPE IDENTIFIER="st-{kind}" LONG-NAME="{kind}"><SPEC-ATTRIBUTES>'
                    f'{"".join(attrs)}</SPEC-ATTRIBUTES></SPEC-OBJECT-TYPE>' for kind in ("Heading", "Requirement"))
    return (f'<?xml version="1.0" encoding="UTF-8"?>{doctype}<REQ-IF {NS}><THE-HEADER><REQ-IF-HEADER '
            'IDENTIFIER="hdr"><TITLE>LV-3</TITLE></REQ-IF-HEADER></THE-HEADER><CORE-CONTENT><REQ-IF-CONTENT>'
            '<DATATYPES><DATATYPE-DEFINITION-STRING IDENTIFIER="dt-string" MAX-LENGTH="4000"/>'
            '<DATATYPE-DEFINITION-XHTML IDENTIFIER="dt-xhtml"/><DATATYPE-DEFINITION-INTEGER IDENTIFIER="dt-int" '
            f'MIN="0" MAX="10"/><DATATYPE-DEFINITION-ENUMERATION IDENTIFIER="dt-status"><SPECIFIED-VALUES>'
            f'{enum_values}</SPECIFIED-VALUES></DATATYPE-DEFINITION-ENUMERATION></DATATYPES>'
            f'<SPEC-TYPES>{types}<SPECIFICATION-TYPE IDENTIFIER="spt" LONG-NAME="Spec"/></SPEC-TYPES>'
            f'<SPEC-OBJECTS>{"".join(body)}</SPEC-OBJECTS><SPEC-RELATIONS/><SPECIFICATIONS>'
            '<SPECIFICATION IDENTIFIER="spec-1" LONG-NAME="Ascent"><TYPE><SPECIFICATION-TYPE-REF>spt'
            f'</SPECIFICATION-TYPE-REF></TYPE><CHILDREN>{"".join(tops)}</CHILDREN></SPECIFICATION>'
            '</SPECIFICATIONS></REQ-IF-CONTENT></CORE-CONTENT></REQ-IF>')


def write_reqif(path: Path, **kwargs) -> Path:
    path.write_text(reqif_text(**kwargs), encoding="utf-8")
    return path


def write_reqifz(path: Path, **kwargs) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("images/logo.png", b"\x89PNG")
        archive.writestr("export.reqif", reqif_text(**kwargs))
    return path
