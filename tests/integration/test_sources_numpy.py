"""In-memory source adapter: accepted inputs, time association and zero-copy loading."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.errors import SourceError, UsageError
from baslt.signals import normalize_signal
from baslt.sources import SourceAdapter, open_source
from baslt.sources.numpy_src import NumpySource


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 10.0, 101)
    return {"t": t, "x": np.sin(t), "nav/pos": rng.standard_normal((101, 3))}


def test_shared_time_key_is_not_copied(arrays):
    src = open_source(arrays)
    assert isinstance(src, NumpySource) and isinstance(src, SourceAdapter)
    run = src.load()
    assert list(run.signals) == ["x", "nav/pos"]
    for sig in run.signals.values():
        assert np.shares_memory(sig.t, arrays["t"])
    assert np.shares_memory(run.signals["x"].v, arrays["x"])
    assert np.shares_memory(run.signals["nav/pos"].v, arrays["nav/pos"])
    assert run.signals["nav/pos"].kind == "vector"
    assert run.signals["x"].path == "x"
    assert run.meta.format == "numpy" and run.meta.path is None and run.meta.issues == []
    assert run.meta.size_bytes == sum(a.nbytes for a in arrays.values())


def test_time_key_named_time():
    t = np.arange(5.0)
    x = t * 2.0
    run = open_source({"time": t, "x": x}).load()
    assert list(run.signals) == ["x"]
    assert np.shares_memory(run.signals["x"].t, t)


def test_list_signals_describes_inputs(arrays):
    arrays["cols"] = np.zeros((3, 101))  # (k, n) is transposed
    arrays["mode"] = np.zeros(101, dtype=np.int16)
    infos = {i.name: i for i in open_source(arrays, units={"x": "m", "/nav/pos": "km"}).list_signals()}
    assert list(infos) == ["x", "nav/pos", "cols", "mode"]
    assert infos["x"].unit == "m" and infos["x"].kind == "continuous" and infos["x"].time_ref == "t"
    assert infos["x"].dtype == "<f8" and infos["x"].n == 101 and infos["x"].shape == (101,)
    assert infos["nav/pos"].unit == "km" and infos["nav/pos"].shape == (101, 3) and infos["nav/pos"].kind == "vector"
    assert infos["cols"].shape == (101, 3)
    assert infos["mode"].kind == "discrete" and infos["mode"].dtype == "<i2"


def test_units_are_applied_on_load(arrays):
    run = open_source(arrays, units={"x": "Pa"}).load()
    assert run.signals["x"].unit == "Pa"
    assert run.signals["nav/pos"].unit is None


def test_pairs_have_their_own_time():
    ta, va = np.arange(4.0), np.array([1.0, 2.0, 3.0, 4.0])
    tb, vb = np.arange(7.0) * 0.5, np.ones(7)
    src = open_source({"a": (ta, va), "b": (tb, vb)})
    infos = {i.name: i for i in src.list_signals()}
    assert infos["a"].n == 4 and infos["b"].n == 7 and infos["a"].time_ref is None
    run = src.load()
    assert np.shares_memory(run.signals["a"].t, ta) and np.shares_memory(run.signals["a"].v, va)
    assert run.signals["b"].n == 7


def test_signal_values_are_reused():
    mode, _ = normalize_signal("mode", [0.0, 1.0, 2.0], np.array(["idle", "burn", "idle"]), path="gnc/mode")
    run = open_source({"flight_mode": mode}).load()
    sig = run.signals["flight_mode"]
    assert sig.labels == ["burn", "idle"] and sig.kind == "discrete"
    assert sig.source_dtype == "str" and sig.path == "gnc/mode"
    assert np.shares_memory(sig.t, mode.t) and np.shares_memory(sig.v, mode.v)
    info = open_source({"flight_mode": mode}).list_signals()[0]
    assert info.name == "flight_mode" and info.kind == "discrete"


def test_run_input(arrays):
    run = open_source(arrays).load()
    src = open_source(run)
    assert [i.name for i in src.list_signals()] == ["x", "nav/pos"]
    subset = src.load(["nav/pos"])
    assert list(subset.signals) == ["nav/pos"]
    assert subset.signals["nav/pos"] is run.signals["nav/pos"]
    scaled = src.load(["x"], time_scale=1e-3)
    assert scaled.signals["x"].t[-1] == pytest.approx(0.01)
    assert np.shares_memory(scaled.signals["x"].v, arrays["x"])


def test_sibling_time_beats_top_level_time():
    t_slow, t_fast = np.arange(3.0), np.arange(10.0) * 0.1
    src = open_source({"t": t_slow, "fast/t": t_fast, "fast/x": np.ones(10), "slow": np.ones(3)})
    infos = {i.name: i for i in src.list_signals()}
    assert set(infos) == {"fast/x", "slow"}
    assert infos["fast/x"].time_ref == "fast/t" and infos["slow"].time_ref == "t"
    run = src.load()
    assert np.shares_memory(run.signals["fast/x"].t, t_fast)
    assert np.shares_memory(run.signals["slow"].t, t_slow)


def test_time_hints_and_global_time():
    clock, t = np.arange(4.0) * 2.0, np.arange(4.0)
    data = {"clock": clock, "t": t, "x": np.ones(4), "y": np.zeros(4)}
    src = open_source(data, time_hints={"x": "clock"})
    assert [i.name for i in src.list_signals()] == ["x", "y"]
    run = src.load()
    assert np.shares_memory(run.signals["x"].t, clock)
    assert np.shares_memory(run.signals["y"].t, t)

    no_default = {"clock": clock, "x": np.ones(4)}
    infos = open_source(no_default).list_signals()
    assert [i.name for i in infos] == ["clock", "x"] and infos[1].time_ref is None
    with pytest.raises(SourceError, match="no time signal for 'clock'"):
        open_source(no_default).load()
    run = open_source(no_default).load(global_time="clock")
    assert list(run.signals) == ["x"]
    assert np.shares_memory(run.signals["x"].t, clock)
    assert [i.name for i in open_source(no_default, global_time="clock").list_signals()] == ["x"]


def test_time_errors():
    with pytest.raises(SourceError, match="no time signal for 'x'"):
        open_source({"x": np.ones(3)}).load()
    with pytest.raises(SourceError, match="time_hints names 'nope'"):
        open_source({"t": np.arange(3.0), "x": np.ones(3)}).load(time_hints={"x": "nope"})
    with pytest.raises(SourceError, match="global time signal 'clock'"):
        open_source({"x": np.ones(3)}).load(global_time="clock")


def test_time_scale_is_applied_once_to_shared_time():
    t_ms = np.array([0, 10, 20, 30], dtype=np.int64)
    run = open_source({"t": t_ms, "a": np.ones(4), "b": np.zeros(4)}).load(time_scale=1e-3)
    a, b = run.signals["a"], run.signals["b"]
    assert a.t.dtype == np.float64 and a.t.tolist() == [0.0, 0.01, 0.02, 0.03]
    assert a.t is b.t


def test_shared_integer_time_is_converted_once():
    t_counts = np.arange(1000, dtype=np.int64)
    run = open_source({"t": t_counts, "a": np.ones(1000), "b": np.zeros(1000)}).load()
    a, b = run.signals["a"], run.signals["b"]
    assert a.t.dtype == np.float64 and a.t is b.t  # one shared clock, not one copy per signal
    assert a.t[:3].tolist() == [0.0, 1.0, 2.0]


def test_unknown_name_suggests_close_match(arrays):
    with pytest.raises(SourceError, match="did you mean 'x'"):
        open_source(arrays).load(["xx"])


def test_names_select_and_order(arrays):
    run = open_source(arrays).load(["nav/pos", "/x", "x"])
    assert list(run.signals) == ["nav/pos", "x"]


def test_value_kinds():
    t = np.arange(4.0)
    run = open_source({"t": t, "count": np.arange(4), "phase": np.array(["a", "b", "b", "a"])}).load()
    assert run.signals["count"].kind == "discrete"
    assert run.signals["phase"].labels == ["a", "b"]
    assert run.signals["phase"].v.tolist() == [0, 1, 1, 0]


def test_non_monotonic_time():
    data = {"t": np.array([0.0, 2.0, 1.0]), "x": np.array([0.0, 2.0, 1.0])}
    with pytest.raises(SourceError, match="time decreases"):
        open_source(data).load()
    run = open_source(data).load(on_non_monotonic="sort")
    assert run.signals["x"].v.tolist() == [0.0, 1.0, 2.0]
    assert run.meta.issues


def test_non_contiguous_and_list_inputs():
    t = np.arange(10.0)
    x = np.arange(20.0)[::2]
    run = open_source({"t": t, "x": x, "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}).load()
    assert run.signals["x"].v.tolist() == x.tolist()
    assert run.signals["y"].kind == "discrete"


def test_mismatched_vector_shape():
    with pytest.raises(SourceError, match="do not match 10 timestamps"):
        open_source({"t": np.arange(10.0), "v": np.zeros((4, 5))}).load()


def test_leading_slash_keys_are_canonical():
    run = open_source({"/t": np.arange(3.0), "/aero/q": np.ones(3)}).load()
    assert list(run.signals) == ["aero/q"]


def test_bad_inputs_are_usage_errors(arrays):
    with pytest.raises(UsageError, match="format 'numpy'"):
        open_source(arrays, format="csv")
    with pytest.raises(UsageError, match="invalid option"):
        open_source(arrays, delimiter=",")
    with pytest.raises(UsageError, match="cannot open a source of type int"):
        open_source(42)
    with pytest.raises(UsageError, match="must be strings"):
        NumpySource({1: np.ones(3)})
    with pytest.raises(UsageError, match=r"\(t, v\)"):
        NumpySource({"x": (np.ones(3), np.ones(3), np.ones(3))})
    with pytest.raises(UsageError, match="duplicated"):
        NumpySource({"x": np.ones(3), "/x": np.ones(3)})
    assert open_source(arrays, format="numpy").list_signals()
