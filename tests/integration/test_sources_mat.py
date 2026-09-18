"""MAT-file sources: version 5 through scipy, version 7.3 through h5py, one flattening for both."""

from __future__ import annotations

import importlib.util
import mmap
import sys
from pathlib import Path

import numpy as np
import pytest

from baslt.api import inspect
from baslt.errors import SourceError, UsageError
from baslt.sources import open_source
from baslt.sources.mat_src import MatSource

sio = pytest.importorskip("scipy.io")
h5py = pytest.importorskip("h5py")
sparse = pytest.importorskip("scipy.sparse")

from reference.mat73_layout import Mat73Writer  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "rocket_sim.py"
N = 50


def _rocket():
    if "rocket_sim" in sys.modules:
        return sys.modules["rocket_sim"]
    spec = importlib.util.spec_from_file_location("rocket_sim", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rocket_sim"] = module
    spec.loader.exec_module(module)
    return module


def _memmapped(arr: np.ndarray) -> bool:
    base = arr
    while base is not None:
        if isinstance(base, (np.memmap, mmap.mmap)):
            return True
        base = getattr(base, "base", None)
    return False


@pytest.fixture
def t():
    return np.linspace(0.0, 4.9, N)


def _savemat(path, variables):
    sio.savemat(path, variables, oned_as="column", long_field_names=True)
    return path


# --------------------------------------------------------------------------------------------------------------
# Version 5


def test_v5_structs_vectors_and_matrix_orientation(tmp_path, t):
    q = np.sin(t)
    pos = np.column_stack([t, 2 * t, 3 * t])
    path = _savemat(tmp_path / "run.mat", {
        "tout": t,
        "aero": {"q": q, "wide": pos.T},  # a 3-by-N matrix: one row per component
        "nav": {"pos": pos},
        "row": q[np.newaxis, :],
    })
    src = open_source(path)
    assert isinstance(src, MatSource) and src.version == "5"
    infos = {info.name: info for info in src.list_signals()}
    assert list(infos) == ["aero/q", "aero/wide", "nav/pos", "row"]
    assert infos["nav/pos"].shape == (N, 3) and infos["nav/pos"].kind == "vector"
    assert infos["aero/wide"].shape == (N, 3)
    assert all(info.time_ref == "tout" and info.unit is None for info in infos.values())
    run = src.load()
    assert run.meta.format == "mat" and run.meta.issues == []
    assert run.meta.size_bytes == path.stat().st_size
    assert run.signals["aero/q"].v.tobytes() == q.tobytes()
    assert run.signals["row"].v.tobytes() == q.tobytes()
    assert np.array_equal(run.signals["nav/pos"].v, pos)
    assert np.array_equal(run.signals["aero/wide"].v, pos)
    assert run.signals["aero/q"].t.tobytes() == t.tobytes()
    assert np.shares_memory(run.signals["aero/q"].t, run.signals["nav/pos"].t)  # one clock in memory


def test_v5_parameters_text_and_empties_are_not_signals(tmp_path, t):
    path = _savemat(tmp_path / "p.mat", {
        "tout": t, "q": np.cos(t), "gain": 3.5, "label": "flight 7", "unused": np.zeros((0, 0)),
        "cfg": {"mass": 1200.0, "name": "vehicle"},
    })
    run = open_source(path).load()
    assert list(run.signals) == ["q"]
    assert run.meta.issues == []


def test_v5_logical_integers_and_single_keep_their_types(tmp_path, t):
    flags = t > 2.0
    path = _savemat(tmp_path / "types.mat", {
        "tout": t, "armed": flags, "mode": np.arange(N, dtype=np.int16) // 10,
        "temp": np.linspace(280, 300, N, dtype=np.float32),
    })
    run = open_source(path).load()
    assert run.signals["armed"].v.dtype == np.bool_ and np.array_equal(run.signals["armed"].v, flags)
    assert run.signals["armed"].kind == "discrete"
    assert run.signals["mode"].v.dtype == np.int16 and run.signals["mode"].kind == "discrete"
    assert run.signals["temp"].v.dtype == np.float32 and run.signals["temp"].kind == "continuous"


def test_v5_cell_of_strings_is_a_discrete_signal(tmp_path, t):
    names = np.array(["idle", "burn", "burn", "coast"] * (N // 4) + ["coast"] * (N % 4), dtype=object)
    path = _savemat(tmp_path / "mode.mat", {"tout": t, "phase": names})
    info = open_source(path).list_signals()[0]
    assert info.name == "phase" and info.kind == "discrete"
    sig = open_source(path).load().signals["phase"]
    assert sig.labels == ["burn", "coast", "idle"]
    assert [sig.labels[c] for c in sig.v] == list(names)


def test_v5_struct_array_elements_are_numbered_from_one(tmp_path, t):
    runs = np.empty(2, dtype=object)
    runs[0] = {"x": t * 1.0}
    runs[1] = {"x": t * 2.0}
    path = _savemat(tmp_path / "sa.mat", {"tout": t, "legs": runs})
    run = open_source(path).load()
    assert list(run.signals) == ["legs/1/x", "legs/2/x"]
    assert np.array_equal(run.signals["legs/2/x"].v, t * 2.0)


def test_v5_complex_and_sparse_are_skipped_with_notes(tmp_path, t):
    path = _savemat(tmp_path / "skip.mat", {
        "tout": t, "q": t, "z": t + 1j * t, "S": sparse.csc_matrix(np.eye(3)),
    })
    run = open_source(path).load()
    assert list(run.signals) == ["q"]
    notes = " | ".join(run.meta.issues)
    assert "z: complex values are not supported" in notes
    assert run.signals["q"].v.tobytes() == t.tobytes()
    assert "S: sparse arrays are not supported" in notes
    assert open_source(path).load(["q"]).meta.issues == []


def test_v5_integer_valued_doubles_stay_doubles(tmp_path, t):
    # MATLAB stores doubles in the smallest integer type that holds them; the reader must return doubles.
    import scipy.io.matlab._mio5_params as mio5

    path = tmp_path / "compact.mat"
    counts = np.arange(N, dtype=np.float64)
    _savemat(path, {"tout": t, "count": counts.astype(np.uint8)})
    data = bytearray(path.read_bytes())
    # rewrite the array class of "count" from uint8 (mxUINT8_CLASS) to double (mxDOUBLE_CLASS), keeping miUINT8 data
    marker = data.find(b"count")
    flags = data.rfind(bytes([mio5.mxUINT8_CLASS, 0, 0, 0]), 0, marker)
    if flags < 0:
        pytest.skip("uncompressed layout not found")
    data[flags] = mio5.mxDOUBLE_CLASS
    path.write_bytes(bytes(data))
    sig = open_source(path).load(["count"]).signals["count"]
    assert sig.v.dtype == np.float64 and sig.kind == "continuous"
    assert sig.v.tobytes() == counts.tobytes()


def test_v5_complex_inside_structs_is_skipped_without_losing_data(tmp_path, t):
    path = _savemat(tmp_path / "nested.mat", {
        "tout": t, "s": {"z": t * 1j, "x": t}, "cells": np.array([{"z": t + 1j}, {"z": t}], dtype=object),
    })
    run = open_source(path).load()
    assert list(run.signals) == ["s/x", "cells/2/z"]
    assert run.signals["s/x"].v.tobytes() == t.tobytes()
    assert set(run.meta.issues) == {"s/z: complex values are not supported",
                                    "cells/1/z: complex values are not supported"}


def test_v5_simulink_structure_with_time(tmp_path, t):
    signals = np.empty(3, dtype=object)
    signals[0] = {"values": np.sin(t), "dimensions": 1, "label": "q_dyn", "blockName": "rocket/Aero"}
    signals[1] = {"values": np.column_stack([t, -t]), "dimensions": 2, "label": "", "blockName": "rocket/Gain 2"}
    signals[2] = {"values": np.cos(t), "dimensions": 1, "label": "q_dyn", "blockName": "rocket/Aero2"}
    path = _savemat(tmp_path / "yout.mat", {"yout": {"time": t, "signals": signals, "blockName": "rocket"}})
    src = open_source(path)
    infos = {info.name: info for info in src.list_signals()}
    assert list(infos) == ["yout/q_dyn", "yout/Gain 2", "yout/q_dyn_3"]
    assert {info.time_ref for info in infos.values()} == {"yout/time"}
    run = src.load()
    assert run.signals["yout/Gain 2"].v.shape == (N, 2)
    assert np.array_equal(run.signals["yout/q_dyn_3"].v, np.cos(t))


def test_v5_single_signal_structure_with_time(tmp_path, t):
    path = _savemat(tmp_path / "one.mat", {"logs": {"time": t, "signals": {"values": t * 3, "label": ""}}})
    run = open_source(path).load()
    assert list(run.signals) == ["logs/signal1"]


def test_v5_missing_time_raises_with_guidance(tmp_path, t):
    path = _savemat(tmp_path / "notime.mat", {"aero": {"q": t}})
    info = open_source(path).list_signals()[0]
    assert info.time_ref is None
    with pytest.raises(SourceError, match="no time signal for 'aero/q'"):
        open_source(path).load()


def test_v5_time_hints_and_global_time(tmp_path, t):
    path = _savemat(tmp_path / "clock.mat", {"clock": t, "aero": {"q": np.sin(t), "stamp": t}})
    run = open_source(path, global_time="clock").load()
    assert list(run.signals) == ["aero/q", "aero/stamp"]
    run = open_source(path).load(["aero/q"], time_hints={"aero/q": "stamp"})
    assert run.signals["aero/q"].t.tobytes() == t.tobytes()


def test_v5_mismatched_matrix_is_reported(tmp_path, t):
    path = _savemat(tmp_path / "bad.mat", {"tout": t, "m": np.zeros((3, 4))})
    with pytest.raises(SourceError, match="3-by-4 does not match the 50 samples"):
        open_source(path).load()


def test_missing_scipy_is_a_usage_error(tmp_path, t, monkeypatch):
    path = _savemat(tmp_path / "x.mat", {"tout": t, "q": t})
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.io", None)
    with pytest.raises(UsageError, match=r"install baslt\[mat\]"):
        open_source(path).list_signals()


def test_damaged_file_is_a_source_error(tmp_path):
    path = tmp_path / "broken.mat"
    path.write_bytes(b"MATLAB 5.0 MAT-file, Platform: GLNXA64" + b"\x00" * 90 + b"\x00\x01IM" + b"\xff" * 64)
    with pytest.raises(SourceError, match="cannot read MAT-file"):
        open_source(path).list_signals()


# --------------------------------------------------------------------------------------------------------------
# Version 7.3


def _v73_basic(path, t, **options):
    with Mat73Writer(path) as w:
        w.numeric(w.root, "tout", t)
        aero = w.struct(w.root, "aero", ["q", "pos"])
        w.numeric(aero, "q", np.sin(t), **options)
        w.numeric(aero, "pos", np.column_stack([t, 2 * t, 3 * t]), **options)
        w.numeric(w.root, "row", np.cos(t)[np.newaxis, :], **options)
    return path


def test_v73_numeric_layout_and_orientation(tmp_path, t):
    path = _v73_basic(tmp_path / "run.mat", t)
    src = open_source(path)
    assert isinstance(src, MatSource) and src.version == "7.3"
    infos = {info.name: info for info in src.list_signals()}
    assert list(infos) == ["aero/q", "aero/pos", "row"]
    assert infos["aero/pos"].shape == (N, 3) and infos["aero/pos"].kind == "vector"
    run = src.load()
    assert run.meta.format == "mat73" and run.meta.issues == []
    assert run.signals["aero/q"].v.tobytes() == np.sin(t).tobytes()
    assert np.array_equal(run.signals["aero/pos"].v, np.column_stack([t, 2 * t, 3 * t]))
    assert run.signals["row"].v.tobytes() == np.cos(t).tobytes()
    assert run.signals["aero/q"].t.tobytes() == t.tobytes()
    assert _memmapped(run.signals["aero/q"].v) and _memmapped(run.signals["aero/q"].t)


def test_v73_compressed_datasets_are_read_directly(tmp_path, t):
    path = _v73_basic(tmp_path / "gz.mat", t, compression="gzip", chunks=True)
    run = open_source(path).load()
    assert not _memmapped(run.signals["aero/q"].v)
    assert run.signals["aero/q"].v.tobytes() == np.sin(t).tobytes()
    assert np.array_equal(run.signals["aero/pos"].v, np.column_stack([t, 2 * t, 3 * t]))


def test_v73_char_logical_empty_and_cell_strings(tmp_path, t):
    flags = t > 2.0
    phases = ["idle", "burn"] * (N // 2)
    with Mat73Writer(tmp_path / "types.mat") as w:
        w.numeric(w.root, "tout", t)
        w.char(w.root, "title", "flight ü")
        w.char(w.root, "blank", "")
        w.logical(w.root, "armed", flags)
        w.empty(w.root, "unused")
        w.numeric(w.root, "mode", (np.arange(N) // 10).astype(np.int32))
        w.cell(w.root, "phase", phases, shape=(N, 1))
        w.cell(w.root, "mixed", ["a", np.arange(3.0)])
    run = open_source(tmp_path / "types.mat").load()
    assert list(run.signals) == ["armed", "mode", "phase"]
    assert run.signals["armed"].v.dtype == np.bool_ and np.array_equal(run.signals["armed"].v, flags)
    assert run.signals["mode"].v.dtype == np.int32
    phase = run.signals["phase"]
    assert [phase.labels[c] for c in phase.v] == phases
    assert run.meta.issues == ["mixed: cell arrays other than lists of character vectors are not supported"]


def test_v73_struct_array_and_structure_with_time(tmp_path, t):
    with Mat73Writer(tmp_path / "yout.mat") as w:
        yout = w.struct(w.root, "yout", ["time", "signals", "blockName"])
        w.numeric(yout, "time", t)
        w.struct_array(yout, "signals", [
            {"values": np.sin(t), "label": "q_dyn", "blockName": "rocket/Aero"},
            {"values": np.column_stack([t, -t]), "label": "", "blockName": "rocket/Gain"},
        ])
        w.char(yout, "blockName", "rocket")
        w.numeric(w.root, "tout", t)
        w.struct_array(w.root, "legs", [{"x": t * 1.0}, {"x": t * 2.0}])
    src = open_source(tmp_path / "yout.mat")
    infos = {info.name: info for info in src.list_signals()}
    assert list(infos) == ["yout/q_dyn", "yout/Gain", "legs/1/x", "legs/2/x"]
    assert infos["yout/q_dyn"].time_ref == "yout/time" and infos["legs/1/x"].time_ref == "tout"
    run = src.load()
    assert np.array_equal(run.signals["yout/Gain"].v, np.column_stack([t, -t]))
    assert np.array_equal(run.signals["legs/2/x"].v, t * 2.0)


def test_v73_objects_sparse_and_complex_are_skipped_with_notes(tmp_path, t):
    with Mat73Writer(tmp_path / "skip.mat") as w:
        w.numeric(w.root, "tout", t)
        w.numeric(w.root, "q", t)
        w.object(w.root, "ts", "timeseries")
        w.sparse(w.root, "S")
        w.complex(w.root, "z", t)
        w.f.create_group("#subsystem#").create_dataset("MCOS", data=np.zeros(4))
        odd = w.f.create_dataset("tbl", data=np.zeros((2, 2)))
        odd.attrs["MATLAB_class"] = np.bytes_("table")
    run = open_source(tmp_path / "skip.mat").load()
    assert list(run.signals) == ["q"]
    assert run.meta.issues == [
        "ts: MATLAB timeseries objects cannot be read without MATLAB",
        "S: sparse arrays are not supported",
        "z: complex values are not supported",
        "tbl: MATLAB table values cannot be read",
    ]


def test_v73_field_order_follows_matlab_fields(tmp_path, t):
    with Mat73Writer(tmp_path / "order.mat") as w:
        w.numeric(w.root, "tout", t)
        s = w.struct(w.root, "s")
        w.numeric(s, "a", t)
        w.numeric(s, "b", t)
        w.numeric(s, "c", t)
        w.set_fields(s, ["c", "a", "b"])
    assert [i.name for i in open_source(tmp_path / "order.mat").list_signals()] == ["s/c", "s/a", "s/b"]


def test_v73_three_dimensional_arrays_are_skipped(tmp_path, t):
    with Mat73Writer(tmp_path / "cube.mat") as w:
        w.numeric(w.root, "tout", t)
        w.f.create_dataset("cube", data=np.zeros((2, 3, N))).attrs["MATLAB_class"] = np.bytes_("double")
        single = w.f.create_dataset("squeezed", data=np.arange(float(N)).reshape(N, 1, 1))
        single.attrs["MATLAB_class"] = np.bytes_("double")
    run = open_source(tmp_path / "cube.mat").load()
    assert list(run.signals) == ["squeezed"]
    assert run.meta.issues == ["cube: 3-D arrays are not supported"]


def test_v73_broken_and_null_references_are_skipped(tmp_path, t):
    with Mat73Writer(tmp_path / "refs.mat") as w:
        w.numeric(w.root, "tout", t)
        cell = w.cell(w.root, "names", ["a", "b"])
        data = cell[()]
        data[1, 0] = h5py.Reference()
        cell[...] = data
    run = open_source(tmp_path / "refs.mat").load()
    assert list(run.signals) == []
    assert run.meta.issues == ["names: cell arrays other than lists of character vectors are not supported"]


def test_v73_is_recognized_without_the_extension(tmp_path, t):
    path = _v73_basic(tmp_path / "run.mat", t)
    renamed = tmp_path / "run.h5"
    renamed.write_bytes(path.read_bytes())
    plain = tmp_path / "logfile"
    plain.write_bytes(path.read_bytes())
    assert isinstance(open_source(plain), MatSource)
    # save('run.h5', ..., '-v7.3') writes a MAT-file with an .h5 name: the header text wins over the extension
    assert isinstance(open_source(renamed), MatSource)
    assert open_source(renamed, format="hdf5").format == "hdf5"  # plain HDF5 only when asked for
    assert isinstance(open_source(renamed, format="mat73"), MatSource)
    assert isinstance(open_source(renamed, format="matlab"), MatSource)


def test_v73_without_the_header_text_is_still_version_73(tmp_path, t):
    path = _v73_basic(tmp_path / "noheader.mat", t)
    data = bytearray(path.read_bytes())
    data[:128] = b"\x00" * 128
    path.write_bytes(bytes(data))
    src = open_source(path)
    assert isinstance(src, MatSource) and src.version == "7.3"
    assert list(src.load().signals) == ["aero/q", "aero/pos", "row"]


def test_missing_h5py_is_a_usage_error(tmp_path, t, monkeypatch):
    path = _v73_basic(tmp_path / "x.mat", t)
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(UsageError, match=r"install baslt\[mat\]"):
        open_source(path).list_signals()


# --------------------------------------------------------------------------------------------------------------
# Equivalence and compilation


@pytest.mark.parametrize("version", ["5", "7.3"])
def test_rocket_run_is_identical_in_every_format(tmp_path, version):
    rs = _rocket()
    data = rs.simulate(duration=6.0, rate=100.0)
    reference = open_source(data).load()
    from_h5 = open_source(rs.write_h5(tmp_path / "run.h5", data)).load()
    from_mat = open_source(rs.write_mat(tmp_path / "run.mat", data, version=version)).load()
    assert set(from_mat.signals) == set(reference.signals) == set(from_h5.signals)
    for name, sig in reference.signals.items():
        for other in (from_mat.signals[name], from_h5.signals[name]):
            assert other.v.dtype == sig.v.dtype and other.v.shape == sig.v.shape
            assert other.v.tobytes() == sig.v.tobytes(), name
            assert other.t.tobytes() == sig.t.tobytes(), name


@pytest.mark.parametrize("version", ["5", "7.3"])
def test_mat_run_compiles_and_verifies_with_declared_units(tmp_path, version):
    from baslt.api import compile, verify

    rs = _rocket()
    data = rs.simulate(duration=70.0, rate=200.0)
    path = rs.write_mat(tmp_path / "run.mat", data, version=version)
    policy = {
        "version": 1,
        "signals": {"decl": {"q": {"path": "aero/q", "unit": "Pa"}}},
        "hard": {"q": {"global_extrema": {}, "threshold_crossing": [{"value": "65 kPa", "hysteresis": "1 kPa"}]}},
    }
    result = compile(path, policy, output=tmp_path / "run.baslt")
    assert result.status in ("pass", "pass_with_warnings"), result.to_json()
    assert result.manifest["source"]["format"] == ("mat73" if version == "7.3" else "mat")
    checked = verify(tmp_path / "run.baslt", source=path)
    assert checked.status in ("pass", "pass_with_warnings")


def test_a_file_with_nothing_to_plot_says_so(tmp_path):
    """MATLAB objects (an inline function, a timeseries) arrive as structs of scalars and text."""
    sio = pytest.importorskip("scipy.io")
    path = tmp_path / "object.mat"
    sio.savemat(path, {"obj": {"expr": "x", "numArgs": 1.0, "isEmpty": 0.0}})
    result = inspect(path)
    assert result["signals"] == []
    assert result["issues"] == ["obj: no arrays that can become signals (only scalars, text or empty values)"]
    sio.savemat(tmp_path / "mixed.mat", {"t": np.arange(3.0), "x": np.arange(3.0), "gain": 2.0, "name": "rig"})
    mixed = inspect(tmp_path / "mixed.mat")
    assert [s["name"] for s in mixed["signals"]] == ["x"] and "issues" not in mixed
