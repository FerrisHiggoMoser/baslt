"""Text-level CSV helpers with no numpy: delimiter sniffing, field splitting and `name [unit]` headers.

The signal reader (`csv_src.py`) and the generic table reader (`baslt.tabular`) share them, and the table reader
must stay importable without numpy.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence

__all__ = [
    "DELIMITER_CANDIDATES",
    "WHITESPACE",
    "count_fields",
    "non_blank",
    "parse_header_field",
    "sniff_delimiter",
    "split_fields",
]

WHITESPACE = "whitespace"
DELIMITER_CANDIDATES: tuple[str, ...] = (",", ";", "\t")

_UNIT_RE = re.compile(r"^(?P<name>.*?)\s*(?:\[(?P<bracket>[^\[\]]*)\]|\((?P<paren>[^()]*)\))\s*$")
_UNIT_TOKEN_RE = re.compile(r"^(?:\[[^\[\]]*\]|\([^()]*\))$")


def non_blank(row: list[str]) -> bool:
    return bool(row) and not (len(row) == 1 and not row[0].strip())


def split_fields(line: str, delimiter: str) -> list[str]:
    if delimiter == WHITESPACE:
        tokens: list[str] = []
        for token in line.split():
            if tokens and _UNIT_TOKEN_RE.match(token):
                tokens[-1] = f"{tokens[-1]} {token}"
            else:
                tokens.append(token)
        return tokens
    return [f.strip() for f in next(csv.reader([line], delimiter=delimiter))]


def count_fields(line: str, delimiter: str) -> int:
    return len(split_fields(line, delimiter))


def sniff_delimiter(lines: Sequence[str]) -> str | None:
    """The candidate splitting every line into the same number (> 1) of fields; most fields wins."""
    best, best_count = None, 1
    for delimiter in DELIMITER_CANDIDATES:
        counts = {count_fields(line, delimiter) for line in lines}
        if len(counts) == 1:
            count = counts.pop()
            if count > best_count:
                best, best_count = delimiter, count
    if best is not None:
        return best
    counts = {count_fields(line, WHITESPACE) for line in lines}
    if len(counts) == 1 and counts.pop() > 1:
        return WHITESPACE
    return None


def parse_header_field(raw: str) -> tuple[str, str | None]:
    field = raw.strip().strip('"').strip()
    match = _UNIT_RE.match(field)
    if match is None:
        return field, None
    unit = match.group("bracket") if match.group("bracket") is not None else match.group("paren")
    unit = unit.strip() if unit is not None else None
    return match.group("name").strip(), unit or None
