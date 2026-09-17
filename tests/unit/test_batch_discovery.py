"""Finding the runs of a batch and naming them."""

from __future__ import annotations

from pathlib import Path

import pytest

from baslt.errors import SourceError, UsageError
from baslt.reqs.batch import discover, run_ids

pytestmark = pytest.mark.minimal


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("t,x\n0,1\n", encoding="utf-8")
    return path


def test_folders_are_searched_in_a_stable_order(tmp_path):
    for name in ("b.csv", "a.h5", "sub/c.mat", "sub/deeper/d.hdf5", "notes.txt", ".hidden/e.csv", "f.CSV"):
        touch(tmp_path / "runs" / name)
    paths, root = discover(tmp_path / "runs")
    assert [p.relative_to(tmp_path / "runs").as_posix() for p in paths] == [
        "a.h5", "b.csv", "f.CSV", "sub/c.mat", "sub/deeper/d.hdf5"]
    assert root == (tmp_path / "runs").resolve()


def test_patterns_files_and_the_output_folder(tmp_path):
    for name in ("x1.csv", "x2.csv", "y.csv", "out/runs/z.csv"):
        touch(tmp_path / name)
    paths, root = discover([str(tmp_path / "x*.csv"), tmp_path / "y.csv", tmp_path / "x1.csv"])
    assert [p.name for p in paths] == ["x1.csv", "x2.csv", "y.csv"]  # the repeated file is listed once
    assert root == tmp_path.resolve()
    paths, _ = discover(tmp_path, skip=tmp_path / "out")
    assert [p.name for p in paths] == ["x1.csv", "x2.csv", "y.csv"]


def test_problems(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(UsageError, match="no run files"):
        discover(tmp_path / "empty")
    with pytest.raises(UsageError, match="no run files match"):
        discover(str(tmp_path / "*.nothing"))
    with pytest.raises(SourceError, match="run file not found"):
        discover([tmp_path / "missing.csv"])
    with pytest.raises(UsageError, match="no runs"):
        discover([])


def test_run_ids_are_stems_unless_they_repeat(tmp_path):
    root = tmp_path.resolve()
    paths = [root / "a" / "run.csv", root / "b" / "run.csv", root / "c" / "other run!.csv", root / "d" / "X.csv",
             root / "e" / "x.csv"]
    assert run_ids(paths, root) == ["a__run", "b__run", "other_run", "X", "x-2"]
