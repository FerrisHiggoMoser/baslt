"""CSV source adapter.

- The delimiter is sniffed among comma, semicolon, tab and whitespace.
- Headers like ``name [unit]`` or ``name (unit)`` give units; the canonical name is the header without its unit.
- Leading blank lines and lines starting with ``#`` before the header are skipped.
- Time: ``time_hints[name]`` -> a column named t, time, timestamp or tout (case-insensitive, in that order of
  preference) -> ``global_time`` -> the first column. Every column named like a time column is a time signal, so
  none of them is listed as a data signal.
- A row of units under the header (``s``, ``m/s``) is read as units, not data, when no cell in it is a number and
  a later row has one. Numbers written with a decimal comma (``0,5``) in a file the comma does not split are read
  as numbers.
- Numbers load through ``np.loadtxt`` as float64. When a column will not parse as a number, the numeric columns are
  read again without it and it is read as text, so parsing stays inside numpy: empty cells in numeric columns
  become NaN and non-numeric columns become strings, which ``normalize_signal`` turns into enum codes. Only a file
  numpy cannot tokenize at all (blank lines between data rows, quoted newlines, ragged rows) falls back to the
  standard csv module.
- A numeric column whose first cell is an integer literal and whose values are all finite integers loads as int64.
"""

from __future__ import annotations

import csv
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from ..errors import SourceError, UsageError
from ..signals import Run, Signal, SignalInfo, SourceMeta, normalize_signal
from .base import (
    canonical_name,
    file_size,
    guess_kind,
    merge_time_options,
    normalize_hints,
    prescale_time,
    select_names,
)
from .csv_text import WHITESPACE
from .csv_text import non_blank as _non_blank
from .csv_text import parse_header_field as _parse_header_field
from .csv_text import sniff_delimiter as _sniff_delimiter
from .csv_text import split_fields as _split_fields

TIME_COLUMN_NAMES: tuple[str, ...] = ("t", "time", "timestamp", "tout")
SNIFF_LINES = 20

_INT_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_COMMA_RE = re.compile(r"^[+-]?\d+,\d+(?:[eE][+-]?\d+)?$")
_SNIFF_CELLS = 50
_FAILED_COLUMN_RE = re.compile(r"\brow \d+, column (\d+)\b")
_MAX_EXACT_INT = 2.0**53


@dataclass(slots=True)
class _Table:
    names: list[str]
    units: list[str | None]
    columns: dict[str, np.ndarray]
    issues: list[str]                          # problems of the file as a whole
    column_issues: dict[str, list[str]]        # problems of one column, keyed by column name


class CsvSource:
    format: ClassVar[str] = "csv"
    extensions: ClassVar[tuple[str, ...]] = (".csv", ".tsv", ".txt")

    def __init__(
        self,
        path: str | Path,
        *,
        delimiter: str | None = None,
        encoding: str = "utf-8-sig",
        units_row: bool | None = None,
        decimal_comma: bool | None = None,
        time_hints: Mapping[str, str] | None = None,
        global_time: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._units_row = units_row  # None: decide from the file
        self._decimal_comma = decimal_comma
        if delimiter is not None and delimiter in (" ", WHITESPACE):
            delimiter = WHITESPACE
        elif delimiter is not None and len(delimiter) != 1:
            raise UsageError(f"CSV delimiter must be one character or 'whitespace', got {delimiter!r}")
        self._delimiter = delimiter
        self._encoding = encoding
        self._time_hints = normalize_hints(time_hints)
        self._global_time = global_time
        self._table: _Table | None = None

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool:
        if not head or b"\x00" in head:
            return False
        try:
            text = head.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.start < len(head) - 4:
                return False
            text = head[: exc.start].decode("utf-8")
        lines = [ln for ln in text.lstrip("﻿").splitlines() if ln.strip() and not ln.startswith("#")]
        if len(text) >= len(head) and len(lines) > 1:
            lines = lines[:-1]  # the last line may be cut
        if not lines or any(ch < " " and ch != "\t" for ln in lines for ch in ln):
            return False
        return _sniff_delimiter(lines[:SNIFF_LINES]) is not None

    # ----------------------------------------------------------------------------------------------------------

    def _read_table(self) -> _Table:
        if self._table is not None:
            return self._table
        try:
            self._table = self._parse()
        except OSError as exc:
            raise SourceError(f"cannot read CSV source {self.path}: {exc.strerror or exc}") from None
        except UnicodeDecodeError as exc:
            raise SourceError(f"CSV source {self.path} is not valid {self._encoding} text: {exc.reason}") from None
        return self._table

    def _parse(self) -> _Table:
        header_line, skip, first_data, sample = self._scan_head()
        if header_line is None:
            return _Table(names=[], units=[], columns={}, issues=[])
        delimiter = self._delimiter or _sniff_delimiter([header_line, *sample]) or ","
        header_fields = _split_fields(header_line, delimiter)
        has_header = not all(_looks_numeric(f) for f in header_fields)
        if has_header:
            names, units = [], []
            for i, raw in enumerate(header_fields):
                name, unit = _parse_header_field(raw)
                names.append(name or f"column_{i + 1}")
                units.append(unit)
            first_tokens = _split_fields(first_data, delimiter) if first_data is not None else None
        else:
            names = [f"column_{i + 1}" for i in range(len(header_fields))]
            units = [None] * len(names)
            first_tokens = header_fields
            skip -= 1
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise SourceError(f"CSV source {self.path}: duplicate column {name!r}")
            seen.add(name)

        issues: list[str] = []
        column_issues: dict[str, list[str]] = {}
        later = [_split_fields(line, delimiter) for line in sample[1:4]]
        units_row = self._units_row
        if units_row is None:
            units_row = has_header and first_tokens is not None and _is_units_row(first_tokens, later)
        if units_row and has_header and first_tokens is not None:
            for i, token in enumerate(first_tokens[:len(units)]):
                text = token.strip().strip('"')
                if text and not units[i]:
                    units[i] = text
            issues.append("the row under the header holds units, not data")
            skip += 1
            first_tokens = later[0] if later else None
        if first_tokens is None:
            columns = {name: np.empty(0, dtype=np.float64) for name in names}
        else:
            columns = self._load_columns(delimiter, skip, names, column_issues, self._decimal_comma)
            for name, token in zip(names, first_tokens):
                columns[name] = _maybe_integer(columns[name], token)
        return _Table(names=names, units=units, columns=columns, issues=issues, column_issues=column_issues)

    def _scan_head(self) -> tuple[str | None, int, str | None, list[str]]:
        """Return (header line, lines up to and including the header, first data line, sample data lines)."""
        header: str | None = None
        skip = 0
        sample: list[str] = []
        with open(self.path, encoding=self._encoding, newline="") as fh:
            for line in fh:
                stripped = line.rstrip("\r\n")
                if header is None:
                    skip += 1
                    if not stripped.strip() or stripped.startswith("#"):
                        continue
                    header = stripped
                    continue
                if stripped.strip():
                    sample.append(stripped)
                    if len(sample) >= SNIFF_LINES:
                        break
        return header, skip, (sample[0] if sample else None), sample

    def _load_columns(self, delimiter: str, skip: int, names: list[str], column_issues: dict[str, list[str]],
                      decimal_comma: bool | None = None) -> dict[str, np.ndarray]:
        """Read every column with the fastest parser the file allows."""
        if decimal_comma is not True:
            matrix = self._load_fast(delimiter, skip, len(names))
            if matrix is not None:
                by_column = np.ascontiguousarray(matrix.T)  # one contiguous row per column
                del matrix
                return {name: by_column[i] for i, name in enumerate(names)}
            mixed = self._load_mixed(delimiter, skip, names, column_issues, decimal_comma)
            if mixed is not None:
                return mixed
        return self._load_fallback(delimiter, skip, names, column_issues, decimal_comma)

    def _loadtxt(self, delimiter: str, skip: int, dtype: np.dtype | type,
                 usecols: Sequence[int] | None) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.loadtxt(self.path, dtype=dtype, delimiter=None if delimiter == WHITESPACE else delimiter,
                              skiprows=skip, comments=None, quotechar='"', encoding=self._encoding, ndmin=2,
                              usecols=usecols)

    def _load_fast(self, delimiter: str, skip: int, ncols: int) -> np.ndarray | None:
        try:
            matrix = self._loadtxt(delimiter, skip, np.float64, None)
        except ValueError:
            return None
        if matrix.shape[1] != ncols:
            return None
        return matrix

    def _load_mixed(self, delimiter: str, skip: int, names: list[str], column_issues: dict[str, list[str]],
                    decimal_comma: bool | None = None) -> dict[str, np.ndarray] | None:
        """Numeric columns as float64 and the remaining ones as text, or None when numpy cannot tokenize the file.

        numpy names the column it could not convert, so the columns that need reading as text are found by
        reloading without them: the loop runs once per such column, never once per sample.
        """
        ncols = len(names)
        text: list[int] = []
        numeric: np.ndarray | None = None
        numeric_cols: list[int] = []
        while True:
            numeric_cols = [i for i in range(ncols) if i not in text]
            if not numeric_cols:
                break
            try:
                numeric = self._loadtxt(delimiter, skip, np.float64, numeric_cols)
            except ValueError as exc:
                column = _failed_column(exc)
                if column is None or column in text or not 0 <= column < ncols:
                    return None  # not a per-cell conversion failure: the file needs the csv module
                text.append(column)
                continue
            if numeric.shape[1] != len(numeric_cols):
                return None
            break
        if not text:
            return None  # the full-width pass already failed, so let the csv module say why
        text.sort()
        try:
            raw = self._loadtxt(delimiter, skip, np.str_, text)
        except ValueError:
            return None
        if raw.shape[1] != len(text) or (numeric is not None and numeric.shape[0] != raw.shape[0]):
            return None
        if not self._rows_are_full(delimiter, skip, ncols, set(text), raw.dtype.itemsize):
            return None
        columns: dict[str, np.ndarray] = {}
        if numeric is not None:
            for position, index in enumerate(numeric_cols):
                columns[names[index]] = np.ascontiguousarray(numeric[:, position])
        found: dict[str, list[str]] = {}
        for position, index in enumerate(text):
            name = names[index]
            columns[name] = _column_from_text(name, raw[:, position], found, decimal_comma)
        if set(columns) != set(names):
            return None
        column_issues.update(found)
        return {name: columns[name] for name in names}

    def _rows_are_full(self, delimiter: str, skip: int, ncols: int, text: set[int], itemsize: int) -> bool:
        """Whether every data row holds exactly `ncols` fields.

        `usecols` makes numpy ignore the fields it did not select, so a row with extra fields would pass
        unnoticed. The structure is checked once at full width with a dtype no cell can fail: float64 for the
        numeric columns and text as wide as the widest cell for the rest.
        """
        width = max(itemsize // np.dtype("U1").itemsize, 1)
        fields = [(f"f{i}", f"U{width}" if i in text else "f8") for i in range(ncols)]
        try:
            self._loadtxt(delimiter, skip, np.dtype(fields), None)
        except ValueError:
            return False
        return True

    def _load_fallback(self, delimiter: str, skip: int, names: list[str], column_issues: dict[str, list[str]],
                       decimal_comma: bool | None = None) -> dict[str, np.ndarray]:
        with open(self.path, encoding=self._encoding, newline="") as fh:
            for _ in range(skip):
                next(fh, None)
            if delimiter == WHITESPACE:
                rows = list(filter(None, map(str.split, fh)))
            else:
                rows = list(filter(_non_blank, csv.reader(fh, delimiter=delimiter)))
        ncols = len(names)
        lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
        bad = np.flatnonzero(lengths != ncols)
        if bad.size:
            i = int(bad[0])
            raise SourceError(f"CSV source {self.path}: data row {i + 1} has {int(lengths[i])} fields "
                              f"but the header has {ncols}")
        columns: dict[str, np.ndarray] = {}
        cells = list(zip(*rows)) if rows else [() for _ in names]
        for name, col in zip(names, cells):
            text = np.array(col, dtype=str) if col else np.empty(0, dtype="<U1")
            columns[name] = _column_from_text(name, text, column_issues, decimal_comma)
        return columns

    # ----------------------------------------------------------------------------------------------------------

    def _plan(self, table: _Table, hints: Mapping[str, str], global_time: str | None
              ) -> tuple[list[str], dict[str, str]]:
        """Data column names in file order and the time column of every column."""
        names = table.names
        if not names:
            return [], {}
        time_set = {n for n in names if n.lower() in TIME_COLUMN_NAMES}
        if time_set:
            # The preferred time name wins, as it does for the sibling rule of the other adapters.
            default = min(time_set, key=lambda n: (TIME_COLUMN_NAMES.index(n.lower()), names.index(n)))
        elif global_time:
            default = canonical_name(global_time)
            if default not in table.columns:
                raise SourceError(f"global time signal {global_time!r} is not a column of {self.path}")
            time_set = {default}
        else:
            default = names[0]
            time_set = {default}
        refs = dict.fromkeys(names, default)
        for name, ref in hints.items():
            target = canonical_name(ref)
            if target not in table.columns:
                raise SourceError(f"time for {name!r}: time_hints names {ref!r}, which is not a column of {self.path}")
            time_set.add(target)
            if name in refs:
                refs[name] = target
        return [n for n in names if n not in time_set], refs

    def parameters(self) -> dict[str, object]:
        """CSV files hold no run parameters."""
        return {}

    def list_signals(self) -> list[SignalInfo]:
        table = self._read_table()
        data, refs = self._plan(table, self._time_hints, self._global_time)
        units = dict(zip(table.names, table.units))
        infos = []
        for name in data:
            col = table.columns[name]
            infos.append(SignalInfo(name=name, path=name, shape=(int(col.shape[0]),), dtype=col.dtype.str,
                                    unit=units[name], kind=guess_kind(col.dtype.kind, col.shape), n=int(col.shape[0]),
                                    time_ref=refs.get(name)))
        return infos

    def load(
        self,
        names: Sequence[str] | None = None,
        *,
        time_hints: Mapping[str, str] | None = None,
        global_time: str | None = None,
        time_scale: float = 1.0,
        on_non_monotonic: str = "error",
    ) -> Run:
        table = self._read_table()
        hints, gtime = merge_time_options(self._time_hints, self._global_time, time_hints, global_time)
        data, refs = self._plan(table, hints, gtime)
        selected = select_names(names, data, table.names, what="column")
        units = dict(zip(table.names, table.units))
        issues = list(table.issues)
        reported: set[str] = set()
        signals: dict[str, Signal] = {}
        times: dict[str, tuple[np.ndarray, float]] = {}
        for name in selected:
            ref = refs[name]
            for column in (ref, name):  # how a column was read matters whenever it is used
                if column not in reported:
                    reported.add(column)
                    issues.extend(table.column_issues.get(column, ()))
            if ref not in times:
                times[ref] = prescale_time(table.columns[ref], time_scale)
            t, remaining = times[ref]
            sig, sig_issues = normalize_signal(name, t, table.columns[name], path=name, unit=units[name],
                                               time_scale=remaining, on_non_monotonic=on_non_monotonic)
            signals[name] = sig
            issues.extend(sig_issues)
        meta = SourceMeta(path=str(self.path), format="csv", size_bytes=file_size(self.path), issues=issues)
        return Run(signals=signals, meta=meta)


# --------------------------------------------------------------------------------------------------------------


def _as_decimal_comma(values: np.ndarray, empty: np.ndarray) -> np.ndarray | None:
    """Numbers written the European way (`0,5`), which only reach one cell when the delimiter is not a comma."""
    filled = values[~empty]
    if not filled.size or not all(_DECIMAL_COMMA_RE.match(str(value)) for value in filled[:_SNIFF_CELLS]):
        return None
    try:
        return np.where(empty, "nan", np.char.replace(values, ",", ".")).astype(np.float64)
    except ValueError:
        return None


def _is_number(text: str) -> bool:
    try:
        float(text.strip().strip('"'))
    except ValueError:
        return False
    return True


def _looks_numeric(text: str) -> bool:
    """A number, or a blank cell: a missing value in a headerless file is not a column name."""
    return not text.strip() or _is_number(text)


def _is_units_row(tokens: Sequence[str], later: Sequence[Sequence[str]]) -> bool:
    """Whether the row under the header holds units (`s`, `m/s`) rather than data: no cell is a number here,
    some cell is a number further down, and the row is not simply blank."""
    filled = [token for token in tokens if token.strip()]
    if not filled or any(_is_number_like(token) for token in filled):
        return False
    return any(_is_number_like(token) for row in later for token in row if token.strip())


def _is_number_like(text: str) -> bool:
    """A number as anyone writes it, with a decimal point or a decimal comma."""
    return _is_number(text) or bool(_DECIMAL_COMMA_RE.match(text.strip().strip('"')))


def _failed_column(exc: ValueError) -> int | None:
    """The zero-based column numpy could not convert, taken from its message, or None for any other failure."""
    match = _FAILED_COLUMN_RE.search(str(exc))
    return int(match.group(1)) - 1 if match is not None else None


def _column_from_text(name: str, text: np.ndarray, column_issues: dict[str, list[str]],
                      decimal_comma: bool | None = None) -> np.ndarray:
    """One column of cells as float64 with empty cells as NaN, or as stripped text when it is not numeric."""
    values = np.char.strip(text) if text.size else np.empty(0, dtype="<U1")
    empty = values == ""
    try:
        if decimal_comma is True:
            raise ValueError("commas are decimal points in this file")
        numbers = np.where(empty, "nan", values).astype(np.float64)
    except ValueError:
        if decimal_comma is False:
            return values
        decimal_comma = _as_decimal_comma(values, empty)
        if decimal_comma is None:
            return values
        column_issues.setdefault(name, []).append("decimal commas read as decimal points")
        numbers = decimal_comma
    n_empty = int(empty.sum())
    if n_empty:
        column_issues.setdefault(name, []).append(f"{name}: {n_empty} empty cells read as NaN")
    return numbers


def _maybe_integer(col: np.ndarray, first_token: str) -> np.ndarray:
    if col.dtype != np.float64 or col.size == 0 or not _INT_RE.match(first_token.strip().strip('"')):
        return col
    if not np.isfinite(col).all() or np.abs(col).max() > _MAX_EXACT_INT or not (col == np.trunc(col)).all():
        return col
    return col.astype(np.int64)
