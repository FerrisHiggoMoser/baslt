"""Random tables survive writing and reading: ours to ours, ours to openpyxl, openpyxl to ours, and patching."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("hypothesis")
openpyxl = pytest.importorskip("openpyxl")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from baslt.tabular import CellEdit, Sheet, patch_xlsx, read_table, write_xlsx  # noqa: E402

# openpyxl does not decode the _xHHHH_ escapes, so text shared with it avoids control characters and that pattern.
plain_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc"), blacklist_characters="￾￿"),
    min_size=1, max_size=40,
).filter(lambda s: "_x" not in s and s.strip() == s and s.strip())
any_text = st.text(min_size=1, max_size=40).filter(lambda s: s.strip() == s and s.strip())
numbers = st.floats(allow_nan=False, allow_infinity=False, width=64) | st.integers(-(2**52), 2**52)
cells = st.one_of(st.none(), plain_text, numbers, st.booleans())
tables = st.lists(st.lists(cells, min_size=1, max_size=6), min_size=1, max_size=12)


def expected_text(value):
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        number = float(value)
        return str(int(number)) if number == int(number) and abs(number) < 2**53 else repr(number)
    return value


def ours(path):
    table = read_table(path)
    return trimmed([[None if c is None or c.is_empty else c.text for c in row] for row in table.rows])


def trimmed(out):
    for row in out:
        while row and row[-1] is None:
            row.pop()
    while out and not out[-1]:
        out.pop()
    return out


def normalized(rows):
    return trimmed([[None if v is None else expected_text(v) for v in row] for row in rows])


settings_ = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


@settings_
@given(tables)
def test_written_tables_read_back_with_both_readers(tmp_path, rows):
    path = write_xlsx(tmp_path / "t.xlsx", [Sheet("T", rows)])
    assert ours(path) == normalized(rows)
    sheet = openpyxl.load_workbook(path)["T"]
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row, start=1):
            got = sheet.cell(r, c).value
            if value is None:
                assert got is None
            elif isinstance(value, bool):
                assert got is value
            elif isinstance(value, (int, float)):
                assert got == value or (math.isclose(got, value) and float(value) == float(got))
            else:
                assert got == value


@settings_
@given(st.lists(st.lists(st.one_of(st.none(), any_text, numbers), min_size=1, max_size=5), min_size=1, max_size=8))
def test_our_escapes_survive_our_reader(tmp_path, rows):
    path = write_xlsx(tmp_path / "e.xlsx", [Sheet("E", rows)])
    assert ours(path) == normalized(rows)


# openpyxl writes floats with 16 significant digits, so values near the float maximum would overflow on reading.
modest = st.one_of(st.none(), plain_text, st.floats(-1e300, 1e300), st.integers(-(2**52), 2**52), st.booleans())


@settings_
@given(st.lists(st.lists(modest, min_size=1, max_size=6), min_size=1, max_size=12))
def test_openpyxl_tables_read_with_ours(tmp_path, rows):
    book = openpyxl.Workbook()
    sheet = book.active
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row, start=1):
            if value is not None:
                sheet.cell(r, c, value)
    path = tmp_path / "o.xlsx"
    book.save(path)
    table = read_table(path)
    # openpyxl stores floats with 16 significant digits, so numbers are compared by value.
    assert len(ours(path)) == len(normalized(rows))
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row, start=1):
            cell = table.cell(r, c)
            if value is None or isinstance(value, str):
                assert (None if cell is None or cell.is_empty else cell.text) == value
            elif isinstance(value, bool):
                assert cell.value is value
            else:
                assert math.isclose(cell.value, float(value), rel_tol=1e-15, abs_tol=1e-300)


@settings_
@given(tables, st.lists(st.tuples(st.integers(1, 15), st.integers(1, 8), cells), min_size=1, max_size=10))
def test_patching_matches_writing_the_edited_table(tmp_path, rows, edits):
    src = write_xlsx(tmp_path / "src.xlsx", [Sheet("T", rows, autofilter=False)])
    patch_xlsx(src, tmp_path / "dst.xlsx", sheet="T", edits=[CellEdit(r, c, v) for r, c, v in edits])
    grid = [list(row) for row in rows]
    for r, c, value in edits:
        while len(grid) < r:
            grid.append([])
        while len(grid[r - 1]) < c:
            grid[r - 1].append(None)
        grid[r - 1][c - 1] = value
    assert ours(tmp_path / "dst.xlsx") == normalized(grid)
    sheet = openpyxl.load_workbook(tmp_path / "dst.xlsx")["T"]
    for r, c, value in edits:
        final = grid[r - 1][c - 1]
        got = sheet.cell(r, c).value
        assert (got is None) if final is None else (got == final)
