"""Tables in and out: CSV, TSV and .xlsx with the standard library only, and ReqIF requirement exports.

Nothing here imports numpy, so reading a requirements sheet or writing a results workbook stays fast and works in
the smallest install.

    read_table(path, sheet=None)       -> Table (rows of Cell: typed value, display text, reference)
    list_sheets(path)                  -> [SheetInfo]
    write_xlsx(path, [Sheet, ...])     -> a styled workbook (Styled cells, Link cells, frozen header, filter)
    patch_xlsx(src, dst, sheet=, edits=[CellEdit]) -> a copy with values written in, everything else untouched
"""

from __future__ import annotations

from .table import Cell, SheetInfo, Table, column_index, column_letter, list_sheets, read_table, split_ref
from .xlsx_patch import CellEdit, patch_xlsx
from .xlsx_write import STYLES, Link, Sheet, Styled, write_xlsx, xlsx_bytes

__all__ = [
    "STYLES",
    "Cell",
    "CellEdit",
    "Link",
    "Sheet",
    "SheetInfo",
    "Styled",
    "Table",
    "column_index",
    "column_letter",
    "list_sheets",
    "patch_xlsx",
    "read_table",
    "split_ref",
    "write_xlsx",
    "xlsx_bytes",
]
