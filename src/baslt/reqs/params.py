"""Run parameters: a table with one row per run, matched to runs by id, file name or path.

The key column is `params.key` (default: the first of run, id, name, file, case). A run matches a row whose key
equals the run's id, its file name, its file stem or its path relative to the batch root. Values stay text unless
they read as numbers; `params.units` gives numbers a unit.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import RequirementsError
from ..tabular import read_table
from .config import Config, normalize_header

__all__ = ["ParamsTable", "load_params", "row_for"]

KEY_NAMES = ("run", "id", "runid", "name", "file", "case", "path")


class ParamsTable:
    def __init__(self, path: Path, key: str, rows: dict[str, dict[str, object]], columns: list[str]) -> None:
        self.path = path
        self.key = key
        self.rows = rows
        self.columns = columns

    def names(self) -> set[str]:
        return set(self.columns)


def _value(text: str) -> object:
    stripped = text.strip()
    if stripped == "":
        return ""
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    return number


def load_params(path, config: Config) -> ParamsTable:
    path = Path(path)
    table = read_table(path)
    if table.n_rows == 0:
        raise RequirementsError(f"{path.name}: the parameter table is empty")
    header = table.row_texts(1)
    names = [normalize_header(text) for text in header]
    wanted = normalize_header(config.params.key) if config.params.key else None
    if wanted is not None:
        if wanted not in names:
            raise RequirementsError(f"{path.name}: params.key names column {config.params.key!r}, which the table "
                                    "does not have")
        key_col = names.index(wanted)
    else:
        key_col = next((names.index(k) for k in KEY_NAMES if k in names), 0)
    columns = [text.strip() for c, text in enumerate(header) if c != key_col and text.strip()]
    rows: dict[str, dict[str, object]] = {}
    for r in range(2, table.n_rows + 1):
        texts = table.row_texts(r)
        if key_col >= len(texts) or not texts[key_col].strip():
            continue
        key = texts[key_col].strip()
        if key in rows:
            raise RequirementsError(f"{table.location(r)}: run {key!r} appears twice in the parameter table")
        rows[key] = {header[c].strip(): _value(texts[c]) for c in range(len(texts))
                     if c != key_col and c < len(header) and header[c].strip()}
    return ParamsTable(path, header[key_col].strip(), rows, columns)


def candidates(source, run_id: str | None = None, root: Path | None = None) -> list[str]:
    out: list[str] = []
    if run_id:
        out.append(run_id)
    if isinstance(source, (str, Path)):
        path = Path(source)
        out += [path.name, path.stem]
        if root is not None:
            try:
                relative = path.resolve().relative_to(root.resolve())
                out += [relative.as_posix(), relative.with_suffix("").as_posix()]
            except ValueError:
                pass
        out.append(str(path))
    return list(dict.fromkeys(out))


def row_for(table: ParamsTable, source, config: Config, *, run_id: str | None = None,
            root: Path | None = None) -> dict[str, object]:
    """The row of `source`; an empty mapping when the table has no row for it."""
    for key in candidates(source, run_id, root):
        found = table.rows.get(key)
        if found is not None:
            return dict(found)
    if len(table.rows) == 1 and not isinstance(source, (str, Path)):
        return dict(next(iter(table.rows.values())))
    return {}
