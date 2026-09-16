"""HDF5 source adapter: memory-mapped and chunked reads, 2-D orientation, units and time association."""

from __future__ import annotations

import mmap
import sys

import numpy as np
import pytest

from baslt.errors import SourceError, UsageError
from baslt.sources import open_source
from baslt.sources.hdf5_src import Hdf5Source

h5py = pytest.importorskip("h5py")


def _memmapped(arr: np.ndarray) -> bool:
    base = arr
    while base is not None:
        if isinstance(base, (np.memmap, mmap.mmap)):
            return True
        base = getattr(base, "base", None)
    return False


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    n = 1000
    return {"t": np.linspace(0.0, 10.0, n), "x": rng.standard_normal(n), "pos": rng.standard_normal((n, 3))}


def _write_basic(path, data, **dataset_options):
    with h5py.File(path, "w", **({"userblock_size": dataset_options.pop("userblock")} if "userblock" in dataset_options
                                 else {})) as f:
        f.create_dataset("t", data=data["t"])
        f.create_dataset("aero/x", data=data["x"], **dataset_options).attrs["units"] = "Pa"
        f.create_dataset("nav/pos", data=data["pos"], **dataset_options).attrs["units"] = "m"
        f["aero/x"].attrs["time"] = "/t"
        f["nav/pos"].attrs["time"] = "/t"
    return path


def test_contiguous_datasets_are_memory_mapped(tmp_path, data):
    path = _write_basic(tmp_path / "run.h5", data)
    src = open_source(path)
    assert isinstance(src, Hdf5Source)
    run = src.load()
    assert list(run.signals) == ["aero/x", "nav/pos"]
    x, pos = run.signals["aero/x"], run.signals["nav/pos"]
    assert _memmapped(x.v) and _memmapped(x.t) and _memmapped(pos.v)
    assert x.v.tobytes() == data["x"].tobytes() and x.t.tobytes() == data["t"].tobytes()
    assert pos.v.tobytes() == data["pos"].tobytes() and pos.kind == "vector"
    assert x.t is pos.t
    assert x.unit == "Pa" and x.path == "aero/x"
    assert run.meta.format == "hdf5" and run.meta.path == str(path)
    assert run.meta.size_bytes == path.stat().st_size and run.meta.issues == []


@pytest.mark.parametrize("userblock", [512, 1024, 4096])
def test_memory_map_offsets_include_the_user_block(tmp_path, data, userblock):
    path = _write_basic(tmp_path / "ub.h5", data, userblock=userblock)
    x = open_source(path).load(["aero/x"]).signals["aero/x"]
    assert _memmapped(x.v)
    assert x.v.tobytes() == data["x"].tobytes()


def test_chunked_gzip_datasets_are_read_directly(tmp_path, data):
    path = _write_basic(tmp_path / "gz.h5", data, chunks=True, compression="gzip", shuffle=True, fletcher32=True)
    run = open_source(path).load()
    x, pos = run.signals["aero/x"], run.signals["nav/pos"]
    assert not _memmapped(x.v) and not _memmapped(pos.v)
    assert x.v.flags.c_contiguous
    assert x.v.tobytes() == data["x"].tobytes() and pos.v.tobytes() == data["pos"].tobytes()


def test_big_endian_dataset_is_converted(tmp_path, data):
    path = tmp_path / "be.h5"
    with h5py.File(path, "w") as f:
        f["t"] = data["t"]
        f.create_dataset("x", data=data["x"].astype(">f8"), dtype=">f8")
    src = open_source(path)
    assert src.list_signals()[0].dtype == ">f8"
    x = src.load().signals["x"]
    assert not _memmapped(x.v)
    assert x.v.dtype == np.dtype("<f8") and x.v.tobytes() == data["x"].tobytes()


def test_integer_bool_and_unallocated_datasets(tmp_path):
    path = tmp_path / "kinds.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.arange(4.0)
        f["mode"] = np.array([0, 1, 1, 3], dtype=np.int16)
        f["flag"] = np.array([True, False, True, True])
        f.create_dataset("unset", shape=(4,), dtype="f4")
    run = open_source(path).load()
    assert run.signals["mode"].kind == "discrete" and run.signals["mode"].v.dtype == np.int16
    assert run.signals["mode"].v.tolist() == [0, 1, 1, 3] and _memmapped(run.signals["mode"].v)
    assert run.signals["flag"].v.tolist() == [True, False, True, True] and run.signals["flag"].kind == "discrete"
    assert run.signals["unset"].v.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_empty_datasets(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("t", shape=(0,), dtype="f8")
        f.create_dataset("x", shape=(0,), dtype="f8")
    assert open_source(path).load().signals["x"].n == 0


def test_two_dimensional_orientation(tmp_path):
    n = 50
    rows = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    path = tmp_path / "orient.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.arange(float(n))
        f["rows"] = rows
        f["cols"] = np.ascontiguousarray(rows.T)
        f["col1"] = np.arange(float(n)).reshape(n, 1)
        f["row1"] = np.arange(float(n)).reshape(1, n)
        f["bad"] = np.zeros((4, 7))
    src = open_source(path)
    infos = {i.name: i for i in src.list_signals()}
    assert infos["rows"].shape == (n, 3) and infos["cols"].shape == (n, 3)
    assert infos["col1"].shape == (n,) and infos["row1"].shape == (n,)
    assert infos["rows"].kind == "vector" and infos["col1"].kind == "continuous"
    run = src.load(["rows", "cols", "col1", "row1"])
    np.testing.assert_array_equal(run.signals["rows"].v, rows)
    np.testing.assert_array_equal(run.signals["cols"].v, rows)
    assert run.signals["col1"].v.shape == (n,) and run.signals["row1"].v.shape == (n,)
    with pytest.raises(SourceError, match=r"shape \(4, 7\) does not match the 50 timestamps"):
        src.load(["bad"])


def test_units_attributes(tmp_path):
    path = tmp_path / "units.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.arange(3.0)
        f["a"] = np.ones(3)
        f["a"].attrs["units"] = "kPa"
        f["b"] = np.ones(3)
        f["b"].attrs["unit"] = np.bytes_(b"deg")
        f["c"] = np.ones(3)
        f["c"].attrs.create("units", np.array(b"m/s", dtype="S3"))
        f["d"] = np.ones(3)
        f["d"].attrs["units"] = ""
        f["e"] = np.ones(3)
        f["e"].attrs["units"] = np.array([b"N"])
    units = {i.name: i.unit for i in open_source(path).list_signals()}
    assert units == {"a": "kPa", "b": "deg", "c": "m/s", "d": None, "e": "N"}
    assert open_source(path).load(["b"]).signals["b"].unit == "deg"


def test_time_attribute_rules(tmp_path):
    path = tmp_path / "attrs.h5"
    with h5py.File(path, "w") as f:
        f["clock"] = np.arange(4.0) * 2.0
        f["g/t"] = np.arange(4.0) * 3.0
        f["g/x"] = np.ones(4)
        f["g/x"].attrs["time"] = "/clock"  # attribute wins over the sibling g/t
        f["g/y"] = np.ones(4)
        f["g/y"].attrs["t"] = b"t"  # relative to the group
        f["h/z"] = np.ones(4)
        f["h/z"].attrs["time"] = f["clock"].ref
        f["h/w"] = np.ones(4)
        f["h/w"].attrs["time"] = "clock"  # not in h, so the top-level name
    src = open_source(path)
    refs = {i.name: i.time_ref for i in src.list_signals()}
    assert refs == {"g/x": "clock", "g/y": "g/t", "h/w": "clock", "h/z": "clock"}
    run = src.load()
    assert run.signals["g/x"].t.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert run.signals["g/y"].t.tolist() == [0.0, 3.0, 6.0, 9.0]
    assert run.signals["h/z"].t is run.signals["g/x"].t


@pytest.mark.parametrize("leaf", ["t", "time", "Time", "timestamp", "tout"])
def test_sibling_time_names(tmp_path, leaf):
    path = tmp_path / "sib.h5"
    with h5py.File(path, "w") as f:
        f[f"sim/{leaf}"] = np.arange(5.0)
        f["sim/q"] = np.ones(5)
    infos = open_source(path).list_signals()
    assert [(i.name, i.time_ref) for i in infos] == [("sim/q", f"sim/{leaf}")]


def test_time_order_sibling_global_and_top_level_tout(tmp_path):
    path = tmp_path / "order.h5"
    with h5py.File(path, "w") as f:
        f["tout"] = np.arange(3.0)
        f["clock"] = np.arange(3.0) + 100.0
        f["a/time"] = np.arange(3.0) + 10.0
        f["a/x"] = np.ones(3)
        f["b/y"] = np.ones(3)
    src = open_source(path)
    assert {i.name: i.time_ref for i in src.list_signals()} == {"a/x": "a/time", "b/y": "tout", "clock": "tout"}
    run = src.load(global_time="/clock")
    assert list(run.signals) == ["a/x", "b/y"]
    assert run.signals["a/x"].t[0] == 10.0  # sibling beats global_time
    assert run.signals["b/y"].t[0] == 100.0  # global_time beats top-level tout
    hinted = src.load(["a/x"], time_hints={"/a/x": "tout"})
    assert hinted.signals["a/x"].t[0] == 0.0  # hints beat everything


def test_time_hint_overrides_attribute(tmp_path):
    path = tmp_path / "hint.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.arange(3.0)
        f["t2"] = np.arange(3.0) * 5.0
        f["x"] = np.ones(3)
        f["x"].attrs["time"] = "/t"
    src = open_source(path, time_hints={"x": "t2"})
    assert [(i.name, i.time_ref) for i in src.list_signals()] == [("x", "t2")]
    assert src.load().signals["x"].t.tolist() == [0.0, 5.0, 10.0]


def test_time_errors(tmp_path):
    path = tmp_path / "notime.h5"
    with h5py.File(path, "w") as f:
        f["g/x"] = np.ones(3)
        f["g/y"] = np.ones(3)
        f["g/y"].attrs["time"] = "/missing"
    src = open_source(path)
    infos = {i.name: i for i in src.list_signals()}
    assert infos["g/x"].time_ref is None and infos["g/x"].n == 3
    with pytest.raises(SourceError, match="no time signal for 'g/x'"):
        src.load(["g/x"])
    with pytest.raises(SourceError, match="time attribute names '/missing'"):
        src.load(["g/y"])
    with pytest.raises(SourceError, match="time_hints names 'nope'"):
        src.load(["g/x"], time_hints={"g/x": "nope"})
    with pytest.raises(SourceError, match="global time signal 'nope'"):
        src.load(["g/x"], global_time="nope")
    with pytest.raises(SourceError, match="unknown signal 'g/z'"):
        src.load(["g/z"])


def test_non_signal_datasets_are_skipped_and_reported(tmp_path):
    path = tmp_path / "mixed.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.arange(3.0)
        f["x"] = np.ones(3)
        f["label"] = np.array([b"a", b"b", b"c"])
        f["param"] = 3.0
        f["cube"] = np.zeros((3, 2, 2))
    src = open_source(path)
    assert [i.name for i in src.list_signals()] == ["x"]
    run = src.load()
    assert list(run.signals) == ["x"]
    assert run.meta.issues == ["skipped 3 datasets that are not numeric 1-D or 2-D arrays: cube, label, param"]
    assert open_source(path).load(["x"]).meta.issues == []


def test_pathological_chunking_is_reported(tmp_path):
    n = 1_100_000
    path = tmp_path / "tiny_chunks.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.arange(n, dtype=np.float64)
        f.create_dataset("x", data=np.zeros(n, dtype=np.float32), chunks=(512,))
        f.create_dataset("y", data=np.zeros(n, dtype=np.float32), chunks=(65536,))
    run = open_source(path).load()
    chunk_issues = [i for i in run.meta.issues if "chunks" in i]
    assert len(chunk_issues) == 1 and chunk_issues[0].startswith("x: 1100000 elements stored in chunks of 512")


def test_time_scale_shares_the_scaled_time(tmp_path):
    path = tmp_path / "ms.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.array([0, 10, 20], dtype=np.int32)
        f["a"] = np.ones(3)
        f["b"] = np.zeros(3)
    run = open_source(path).load(time_scale=1e-3)
    assert run.signals["a"].t.tolist() == [0.0, 0.01, 0.02]
    assert run.signals["a"].t is run.signals["b"].t


def _mapping(arr: np.ndarray):
    """The memory map an array is a view of, or None."""
    base = arr
    while base is not None:
        if isinstance(base, mmap.mmap):
            return base
        if isinstance(base, np.memmap):
            return base._mmap
        base = getattr(base, "base", None)
    return None


def test_one_memory_map_serves_every_dataset(tmp_path):
    path = tmp_path / "many.h5"
    with h5py.File(path, "w") as f:
        f["t"] = np.arange(50.0)
        for i in range(12):
            f[f"s{i}"] = np.arange(50.0) + i
    run = open_source(path).load()
    assert len(run.signals) == 12
    maps = {id(_mapping(sig.v)) for sig in run.signals.values()}
    assert None not in {_mapping(sig.v) for sig in run.signals.values()}
    assert len(maps) == 1  # one descriptor for the file, not one per signal
    assert run.signals["s7"].v.tolist() == (np.arange(50.0) + 7).tolist()


def test_read_failures_become_source_errors(tmp_path, data, monkeypatch):
    path = _write_basic(tmp_path / "run.h5", data)
    real_memmap = np.memmap  # _memmapped cannot be used while np.memmap is not a class

    def no_memmap(*args, **kwargs):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(np, "memmap", no_memmap)
    values = open_source(path).load(["aero/x"]).signals["aero/x"].v  # mapping failed, so it is read instead
    assert not isinstance(values, real_memmap) and values.base is None
    assert values.tobytes() == data["x"].tobytes()

    def no_read(self, dest, *args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(h5py.Dataset, "read_direct", no_read)
    with pytest.raises(SourceError, match=r"cannot read dataset 't' of .*run\.h5"):
        open_source(path).load(["aero/x"])  # the clock is read first, so it is the one that fails


@pytest.mark.parametrize("userblock", [4096, 8192, 16384])
def test_large_user_blocks_are_sniffed(tmp_path, data, userblock):
    path = tmp_path / f"ub{userblock}"  # no extension, so only sniffing can find the format
    with h5py.File(path, "w", userblock_size=userblock) as f:
        f["t"] = data["t"]
        f["x"] = data["x"]
    src = open_source(path)
    assert isinstance(src, Hdf5Source)
    assert src.load().signals["x"].v.tobytes() == data["x"].tobytes()


def test_soft_linked_time_array_is_visible(tmp_path):
    path = tmp_path / "soft.h5"
    with h5py.File(path, "w") as f:
        f["sim/clock"] = np.arange(3.0)
        f["t"] = h5py.SoftLink("/sim/clock")
        f["x"] = np.ones(3)
    src = open_source(path)
    assert ("x", "t") in [(i.name, i.time_ref) for i in src.list_signals()]
    assert src.load(["x"]).signals["x"].t.tolist() == [0.0, 1.0, 2.0]


def test_link_cycles_and_external_links_are_not_followed(tmp_path):
    path = tmp_path / "links.h5"
    with h5py.File(path, "w") as f:
        f["g/t"] = np.arange(3.0)
        f["g/x"] = np.ones(3)
        f["g/up"] = h5py.SoftLink("/")  # a cycle back to the root
        f["ext"] = h5py.ExternalLink("elsewhere.h5", "/y")
        f["dead"] = h5py.SoftLink("/missing")
    infos = open_source(path).list_signals()
    assert [(i.name, i.time_ref) for i in infos] == [("g/x", "g/t")]


def test_missing_h5py_is_a_usage_error(tmp_path, monkeypatch, data):
    path = _write_basic(tmp_path / "run.h5", data)
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(UsageError, match=r"install baslt\[hdf5\]"):
        open_source(path).list_signals()


@pytest.mark.parametrize("name, userblock", [("run.h5", 0), ("run.hdf5", 0), ("run.he5", 0), ("run.bin", 0),
                                             ("run.bin", 512), ("run", 1024), ("run.dat", 2048)])
def test_dispatch_by_extension_and_signature(tmp_path, data, name, userblock):
    path = tmp_path / name
    with h5py.File(path, "w", **({"userblock_size": userblock} if userblock else {})) as f:
        f["t"] = data["t"]
        f["x"] = data["x"]
    src = open_source(path)
    assert isinstance(src, Hdf5Source)
    assert src.load().signals["x"].v.tobytes() == data["x"].tobytes()


def test_explicit_format_and_corrupt_file(tmp_path, data):
    path = tmp_path / "run.data"
    with h5py.File(path, "w") as f:
        f["t"] = data["t"]
        f["x"] = data["x"]
    assert isinstance(open_source(path, format="h5"), Hdf5Source)
    broken = tmp_path / "broken.h5"
    broken.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    with pytest.raises(SourceError, match="cannot open HDF5 source"):
        open_source(broken).list_signals()


def test_a_clock_in_a_parent_group_serves_the_groups_below(tmp_path):
    t = np.linspace(0.0, 2.0, 21)
    path = tmp_path / "simout.h5"
    with h5py.File(path, "w") as f:
        f["time"] = t
        f["simout/q_dyn"] = np.sin(t)
        f["simout/deep/mode"] = (t > 1).astype(np.int32)
        f["fast/t"] = np.linspace(0.0, 2.0, 41)
        f["fast/x"] = np.cos(np.linspace(0.0, 2.0, 41))
        f["fast/inner/y"] = np.zeros(41)
    infos = {info.name: info.time_ref for info in open_source(path).list_signals()}
    assert infos == {"simout/q_dyn": "time", "simout/deep/mode": "time", "fast/x": "fast/t",
                     "fast/inner/y": "fast/t"}  # the nearest clock wins
    run = open_source(path).load()
    assert np.array_equal(run.signals["simout/deep/mode"].t, t)


def test_an_explicit_global_clock_beats_a_parent_group_clock(tmp_path):
    t = np.linspace(0.0, 2.0, 21)
    path = tmp_path / "simout.h5"
    with h5py.File(path, "w") as f:
        f["time"] = t
        f["clock"] = t * 1000.0
        f["simout/q_dyn"] = np.sin(t)
    source = open_source(path, global_time="clock")
    assert {info.name: info.time_ref for info in source.list_signals()}["simout/q_dyn"] == "clock"
