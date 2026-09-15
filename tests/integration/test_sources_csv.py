"""CSV source adapter: delimiters, unit headers, time columns, string enums and empty cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baslt.errors import SourceError, UsageError
from baslt.sources import open_source
from baslt.sources.csv_src import CsvSource


def _write(tmp_path: Path, name: str, text: str | bytes) -> Path:
    path = tmp_path / name
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_unit_headers(tmp_path):
    path = _write(tmp_path, "run.csv",
                  "t [s],q_dyn [kPa],alpha (deg),count,note []\n"
                  "0.0,1.5,0.25,1,0.5\n0.5,2.5,-0.125,2,0.5\n1.0,3.5,0.5,3,0.5\n")
    src = open_source(path)
    assert isinstance(src, CsvSource)
    infos = {i.name: i for i in src.list_signals()}
    assert list(infos) == ["q_dyn", "alpha", "count", "note"]
    assert infos["q_dyn"].unit == "kPa" and infos["alpha"].unit == "deg"
    assert infos["count"].unit is None and infos["note"].unit is None
    assert infos["q_dyn"].time_ref == "t" and infos["q_dyn"].n == 3 and infos["q_dyn"].shape == (3,)
    assert infos["count"].kind == "discrete" and infos["count"].dtype == "<i8"
    run = src.load()
    q = run.signals["q_dyn"]
    assert q.t.tolist() == [0.0, 0.5, 1.0] and q.v.tolist() == [1.5, 2.5, 3.5]
    assert q.unit == "kPa" and q.path == "q_dyn" and q.kind == "continuous"
    assert run.signals["alpha"].v.tolist() == [0.25, -0.125, 0.5]
    assert run.signals["count"].v.dtype == np.int64
    assert np.shares_memory(q.t, run.signals["alpha"].t)
    assert run.meta.format == "csv" and run.meta.path == str(path)
    assert run.meta.size_bytes == path.stat().st_size


def test_float_text_round_trips_exactly(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.standard_normal(5000) * 10.0 ** rng.integers(-300, 300, 5000)
    x[:6] = [0.0, -0.0, 5e-324, 1.7976931348623157e308, 0.1, 1e16]
    t = np.arange(x.size, dtype=np.float64) * 1e-3
    lines = np.char.add(np.char.add(t.astype(str), ","), x.astype(str))
    path = _write(tmp_path, "exact.csv", "t,x\n" + "\n".join(lines.tolist()) + "\n")
    sig = open_source(path).load().signals["x"]
    assert sig.v.tobytes() == x.tobytes()
    assert sig.t.tobytes() == t.tobytes()


@pytest.mark.parametrize(
    "name, text",
    [
        ("semi.csv", "t [s];x [m];y\n0;1.5;2\n1;2.5;3\n2;3.5;4\n"),
        ("tabs.tsv", "t [s]\tx [m]\ty\n0\t1.5\t2\n1\t2.5\t3\n2\t3.5\t4\n"),
        ("spaces.txt", "t [s]   x [m]  y\n0   1.5  2\n1   2.5  3\n2 3.5 4\n"),
        ("comma.csv", "t [s], x [m], y\n0, 1.5, 2\n1, 2.5, 3\n2, 3.5, 4\n"),
    ],
)
def test_delimiters_are_sniffed(tmp_path, name, text):
    run = open_source(_write(tmp_path, name, text)).load()
    assert list(run.signals) == ["x", "y"]
    assert run.signals["x"].unit == "m"
    assert run.signals["x"].v.tolist() == [1.5, 2.5, 3.5]
    assert run.signals["y"].v.tolist() == [2, 3, 4]
    assert run.signals["y"].t.tolist() == [0.0, 1.0, 2.0]


def test_semicolon_file_with_commas_in_text(tmp_path):
    path = _write(tmp_path, "semi.csv", "t;label;x\n0;a,b;1.5\n1;c,d;2.5\n")
    run = open_source(path).load()
    assert run.signals["label"].labels == ["a,b", "c,d"]
    assert run.signals["x"].v.tolist() == [1.5, 2.5]


def test_delimiter_option(tmp_path):
    path = _write(tmp_path, "pipe.csv", "t|x\n0|1.5\n1|2.5\n")
    assert open_source(path, delimiter="|").load().signals["x"].v.tolist() == [1.5, 2.5]
    with pytest.raises(UsageError, match="delimiter"):
        open_source(path, delimiter="||")


def test_string_enum_column(tmp_path):
    path = _write(tmp_path, "modes.csv", "time,mode,x\n0,idle,0.1\n1,burn,0.2\n2, burn ,0.3\n3,coast,0.4\n")
    src = open_source(path)
    infos = {i.name: i for i in src.list_signals()}
    assert infos["mode"].kind == "discrete" and infos["mode"].dtype.startswith("<U")
    run = src.load()
    mode = run.signals["mode"]
    assert mode.kind == "discrete" and mode.source_dtype == "str"
    assert mode.labels == ["burn", "coast", "idle"]
    assert [mode.labels[c] for c in mode.v] == ["idle", "burn", "burn", "coast"]
    assert run.signals["x"].v.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert mode.t.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_empty_cells_become_nan(tmp_path):
    path = _write(tmp_path, "gaps.csv", "t,x,y\n0,1.5,\n1,,2.5\n2,3.5,4.5\n")
    run = open_source(path).load()
    x, y = run.signals["x"].v, run.signals["y"].v
    assert x[0] == 1.5 and np.isnan(x[1]) and x[2] == 3.5
    assert np.isnan(y[0]) and y[1:].tolist() == [2.5, 4.5]
    assert "x: 1 empty cells read as NaN" in run.meta.issues
    assert "y: 1 empty cells read as NaN" in run.meta.issues


def test_empty_time_cell_drops_the_sample(tmp_path):
    path = _write(tmp_path, "tgap.csv", "t,x\n0,1.5\n,2.5\n2,3.5\n")
    run = open_source(path).load()
    assert run.signals["x"].t.tolist() == [0.0, 2.0]
    assert run.signals["x"].v.tolist() == [1.5, 3.5]
    assert any("non-finite timestamps" in issue for issue in run.meta.issues)


def test_blank_lines_are_ignored(tmp_path):
    path = _write(tmp_path, "blank.csv", "t,x\n0,1.5\n\n   \n1,2.5\n")
    assert open_source(path).load().signals["x"].v.tolist() == [1.5, 2.5]


def test_integer_columns(tmp_path):
    path = _write(tmp_path, "ints.csv", "t,a,b,c,d\n0,1,1.0,1,-3\n1,2,2.0,2.5,+4\n")
    run = open_source(path).load()
    assert run.signals["a"].v.dtype == np.int64 and run.signals["a"].kind == "discrete"
    assert run.signals["b"].v.dtype == np.float64 and run.signals["b"].kind == "continuous"
    assert run.signals["c"].v.dtype == np.float64 and run.signals["c"].v.tolist() == [1.0, 2.5]
    assert run.signals["d"].v.tolist() == [-3, 4] and run.signals["d"].v.dtype == np.int64


def test_time_column_name_is_case_insensitive(tmp_path):
    path = _write(tmp_path, "case.csv", "x,TimeStamp,y\n1.5,0,2.5\n3.5,1,4.5\n")
    infos = open_source(path).list_signals()
    assert [i.name for i in infos] == ["x", "y"] and infos[0].time_ref == "TimeStamp"
    assert open_source(path).load().signals["y"].t.tolist() == [0.0, 1.0]


def test_first_column_is_the_time_fallback(tmp_path):
    path = _write(tmp_path, "clock.csv", "clock [ms],x\n0,1.5\n10,2.5\n20,3.5\n")
    src = open_source(path)
    assert [(i.name, i.time_ref) for i in src.list_signals()] == [("x", "clock")]
    run = src.load(time_scale=1e-3)
    assert run.signals["x"].t.tolist() == [0.0, 0.01, 0.02]


def test_global_time_column(tmp_path):
    path = _write(tmp_path, "global.csv", "a,clock,x\n1.5,0,2.5\n3.5,1,4.5\n")
    run = open_source(path).load(global_time="clock")
    assert list(run.signals) == ["a", "x"]
    assert run.signals["a"].t.tolist() == [0.0, 1.0]
    with pytest.raises(SourceError, match="global time signal 'nope'"):
        open_source(path).load(global_time="nope")


def test_named_time_column_comes_before_global_time(tmp_path):
    path = _write(tmp_path, "both.csv", "t,clock,x\n0,5,1.5\n1,6,2.5\n")
    run = open_source(path, global_time="clock").load()
    assert list(run.signals) == ["clock", "x"]
    assert run.signals["x"].t.tolist() == [0.0, 1.0]


def test_time_hints(tmp_path):
    path = _write(tmp_path, "hints.csv", "t,x,t_slow,y\n0,1.5,0,2.5\n1,2.5,10,3.5\n")
    src = open_source(path, time_hints={"y": "t_slow"})
    assert [(i.name, i.time_ref) for i in src.list_signals()] == [("x", "t"), ("y", "t_slow")]
    run = src.load()
    assert run.signals["y"].t.tolist() == [0.0, 10.0]
    assert run.signals["x"].t.tolist() == [0.0, 1.0]
    with pytest.raises(SourceError, match="time_hints names 'nope'"):
        open_source(path).load(time_hints={"y": "nope"})


def test_tout_is_a_time_column(tmp_path):
    path = _write(tmp_path, "tout.csv", "q,tout,x\n5.0,0,1.5\n6.0,1,2.5\n")
    src = open_source(path)
    assert [(i.name, i.time_ref) for i in src.list_signals()] == [("q", "tout"), ("x", "tout")]
    run = src.load()
    assert run.signals["q"].t.tolist() == [0.0, 1.0] and run.signals["q"].v.tolist() == [5.0, 6.0]
    assert run.signals["x"].t.tolist() == [0.0, 1.0]


def test_every_time_named_column_is_a_time_signal(tmp_path):
    path = _write(tmp_path, "many.csv", "t,time,x\n0,10,1.5\n1,11,2.5\n")
    src = open_source(path)
    assert [(i.name, i.time_ref) for i in src.list_signals()] == [("x", "t")]  # the preferred name is the clock
    assert src.load().signals["x"].t.tolist() == [0.0, 1.0]
    upper = open_source(_write(tmp_path, "upper.csv", "TOUT,x\n0,1.5\n1,2.5\n"))
    assert [(i.name, i.time_ref) for i in upper.list_signals()] == [("x", "TOUT")]


def test_bad_time_options_are_reported_by_list_signals(tmp_path):
    path = _write(tmp_path, "hint.csv", "x,t,y\n1.5,0,2.5\n3.5,1,4.5\n")
    assert [i.name for i in open_source(path).list_signals()] == ["x", "y"]
    with pytest.raises(SourceError, match="time_hints names 'nope'"):
        open_source(path, time_hints={"y": "nope"}).list_signals()
    other = _write(tmp_path, "gtime.csv", "x,clock,y\n1.5,0,2.5\n3.5,1,4.5\n")
    with pytest.raises(SourceError, match="global time signal 'nope'"):
        open_source(other, global_time="nope").list_signals()


def test_parse_issues_are_kept_when_names_are_given(tmp_path):
    path = _write(tmp_path, "gaps2.csv", "t,x,y\n0,1.5,\n1,,2.5\n2,3.5,4.5\n")
    assert open_source(path).load(["x"]).meta.issues == ["x: 1 empty cells read as NaN"]
    assert open_source(path).load(["y"]).meta.issues == ["y: 1 empty cells read as NaN"]
    assert open_source(path).load(["x", "y"]).meta.issues == ["x: 1 empty cells read as NaN",
                                                             "y: 1 empty cells read as NaN"]


def test_mixed_columns_are_parsed_without_the_python_fallback(tmp_path, monkeypatch):
    path = _write(tmp_path, "mixed.csv", "t,x,mode,y\n0,1.5,idle,\n1,,burn,2.5\n2,3.5,coast,4.5\n")

    def fail(*args, **kwargs):
        raise AssertionError("a file numpy can tokenize went through the csv-module fallback")

    monkeypatch.setattr(CsvSource, "_load_fallback", fail)
    run = open_source(path).load()
    assert run.signals["mode"].labels == ["burn", "coast", "idle"]
    assert [run.signals["mode"].labels[c] for c in run.signals["mode"].v] == ["idle", "burn", "coast"]
    assert run.signals["x"].v[0] == 1.5 and np.isnan(run.signals["x"].v[1])
    assert np.isnan(run.signals["y"].v[0]) and run.signals["y"].v[1:].tolist() == [2.5, 4.5]
    assert run.signals["x"].t.tolist() == [0.0, 1.0, 2.0]
    assert "x: 1 empty cells read as NaN" in run.meta.issues


def test_ragged_rows_are_rejected_whatever_the_column_types(tmp_path):
    with pytest.raises(SourceError, match="data row 2 has 3 fields but the header has 2"):
        open_source(_write(tmp_path, "numeric.csv", "t,x\n0,1\n1,2,3\n")).load()
    with pytest.raises(SourceError, match="data row 2 has 1 fields but the header has 2"):
        open_source(_write(tmp_path, "short.csv", "t,x\n0,1\n1\n")).load()
    with pytest.raises(SourceError, match="data row 3 has 4 fields but the header has 3"):
        open_source(_write(tmp_path, "text.csv", "t,x,mode\n0,1,idle\n1,2,burn\n2,3,coast,extra\n")).load()


def test_headerless_file_with_a_missing_value(tmp_path):
    path = _write(tmp_path, "holed.csv", "0,,7\n1,2.5,8\n2,3.5,9\n")
    src = open_source(path)
    assert [(i.name, i.n) for i in src.list_signals()] == [("column_2", 3), ("column_3", 3)]
    run = src.load()
    assert run.signals["column_3"].t.tolist() == [0.0, 1.0, 2.0]
    assert run.signals["column_3"].v.tolist() == [7, 8, 9]
    values = run.signals["column_2"].v
    assert np.isnan(values[0]) and values[1:].tolist() == [2.5, 3.5]
    assert "column_2: 1 empty cells read as NaN" in run.meta.issues


def test_headerless_numeric_file(tmp_path):
    path = _write(tmp_path, "raw.csv", "0,1.5,7\n1,2.5,8\n")
    run = open_source(path).load()
    assert list(run.signals) == ["column_2", "column_3"]
    assert run.signals["column_2"].t.tolist() == [0.0, 1.0]
    assert run.signals["column_3"].v.tolist() == [7, 8]


def test_comments_and_byte_order_mark(tmp_path):
    path = _write(tmp_path, "bom.csv", "﻿# exported run\n\nt [s],x [m]\n0,1.5\n1,2.5\n".encode())
    run = open_source(path).load()
    assert list(run.signals) == ["x"] and run.signals["x"].unit == "m"


def test_quoted_header_fields(tmp_path):
    path = _write(tmp_path, "quoted.csv", '"t [s]","x, y [m]"\n0,"1.5"\n1,2.5\n')
    run = open_source(path).load()
    assert list(run.signals) == ["x, y"]
    assert run.signals["x, y"].unit == "m" and run.signals["x, y"].v.tolist() == [1.5, 2.5]


def test_header_only_file(tmp_path):
    path = _write(tmp_path, "empty.csv", "t,x\n")
    infos = open_source(path).list_signals()
    assert [(i.name, i.n) for i in infos] == [("x", 0)]
    assert open_source(path).load().signals["x"].n == 0


def test_malformed_files(tmp_path):
    with pytest.raises(SourceError, match="data row 2 has 3 fields but the header has 2"):
        open_source(_write(tmp_path, "ragged.csv", "t,x\n0,a\n1,b,c\n")).load()
    with pytest.raises(SourceError, match="duplicate column 'x'"):
        open_source(_write(tmp_path, "dup.csv", "t,x,x [m]\n0,1,2\n")).list_signals()
    with pytest.raises(SourceError, match="time must be numeric"):
        open_source(_write(tmp_path, "strtime.csv", "t,x\nmorning,1\nnoon,2\n")).load()
    with pytest.raises(SourceError, match="not valid"):
        open_source(_write(tmp_path, "latin.csv", b"t,x\n0,\xe9\xe9\n"), encoding="utf-8").load()


def test_unknown_column_name(tmp_path):
    path = _write(tmp_path, "run.csv", "t,alpha\n0,1\n")
    with pytest.raises(SourceError, match="did you mean 'alpha'"):
        open_source(path).load(["alpah"])


def test_sniffed_without_csv_extension(tmp_path):
    path = _write(tmp_path, "run.dat", "t [s];x\n0;1.5\n1;2.5\n")
    src = open_source(path)
    assert isinstance(src, CsvSource)
    assert src.load().signals["x"].v.tolist() == [1.5, 2.5]
    assert isinstance(open_source(_write(tmp_path, "run.log", "t x\n0 1\n"), format="csv"), CsvSource)


def test_sniff_rejects_binary_and_single_column():
    assert not CsvSource.sniff(Path("x"), b"\x00\x01\x02,\n")
    assert not CsvSource.sniff(Path("x"), b"")
    assert not CsvSource.sniff(Path("x"), b"justoneword\nanother\n")
    assert CsvSource.sniff(Path("x"), b"t,x\n0,1\n1,2\n")
