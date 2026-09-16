"""The same simulated run loads to identical arrays from memory, CSV and HDF5; open_source dispatch rules."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from baslt.errors import SourceError, UsageError
from baslt.signals import Signal
from baslt.sources import base, open_source, register_adapter

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "rocket_sim.py"


def _load_example():
    if "rocket_sim" in sys.modules:
        return sys.modules["rocket_sim"]
    spec = importlib.util.spec_from_file_location("rocket_sim", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rocket_sim"] = module
    spec.loader.exec_module(module)
    return module


rs = _load_example()


@pytest.fixture(scope="module")
def sim():
    return rs.simulate(duration=50.0, rate=100.0)


def _assert_identical(got: Signal, want: Signal) -> None:
    name = want.name
    assert got.name == name and got.path == want.path, name
    assert (got.kind, got.unit, got.labels, got.source_dtype) == (want.kind, want.unit, want.labels,
                                                                   want.source_dtype), name
    assert got.t.dtype == want.t.dtype and got.t.shape == want.t.shape, name
    assert got.v.dtype == want.v.dtype and got.v.shape == want.v.shape, name
    assert np.ascontiguousarray(got.t).tobytes() == np.ascontiguousarray(want.t).tobytes(), name
    assert np.ascontiguousarray(got.v).tobytes() == np.ascontiguousarray(want.v).tobytes(), name


def test_fixture_covers_nan_gap_enum_and_vectors(sim):
    assert np.isnan(sim["thermal/skin_temp"]).any()
    assert sim["gnc/mode"].dtype == np.int64
    assert sim["nav/position"].ndim == 2


@pytest.mark.parametrize("chunked, compression", [(False, None), (True, None), (True, "gzip")])
def test_hdf5_matches_numpy(sim, tmp_path, chunked, compression):
    pytest.importorskip("h5py")
    path = rs.write_h5(tmp_path / "run.h5", sim, chunked=chunked, compression=compression)
    want = open_source(sim, units=rs.UNITS).load()
    got = open_source(path).load()
    assert sorted(got.signals) == sorted(want.signals)
    for name, sig in want.signals.items():
        _assert_identical(got.signals[name], sig)
    assert got.meta.issues == want.meta.issues == []


def test_csv_matches_numpy(sim, tmp_path):
    path = rs.write_csv(tmp_path / "run.csv", sim)
    columns = rs.csv_columns(sim)
    flat = {name: values for name, _, values in columns}
    want = open_source(flat, units={name: unit for name, unit, _ in columns}).load()
    got = open_source(path).load()
    assert list(got.signals) == list(want.signals)
    for name, sig in want.signals.items():
        _assert_identical(got.signals[name], sig)


def test_csv_fallback_parser_matches_numpy(sim, tmp_path):
    path = rs.write_csv(tmp_path / "run.csv", sim)
    lines = path.read_text().splitlines()
    phases = np.where(sim["t"] < 25.0, "boost", "cruise")
    text = "\n".join([lines[0] + ",phase"] + [f"{line},{p}" for line, p in zip(lines[1:], phases.tolist())])
    path.write_text(text + "\n")
    columns = rs.csv_columns(sim)
    flat = {name: values for name, _, values in columns}
    want = open_source(flat, units={name: unit for name, unit, _ in columns}).load()
    got = open_source(path).load()
    assert got.signals["phase"].labels == ["boost", "cruise"]
    for name, sig in want.signals.items():
        _assert_identical(got.signals[name], sig)


def test_three_formats_agree_on_vectors(sim, tmp_path):
    pytest.importorskip("h5py")
    h5 = open_source(rs.write_h5(tmp_path / "run.h5", sim)).load(["nav/position"]).signals["nav/position"]
    csv_run = open_source(rs.write_csv(tmp_path / "run.csv", sim)).load(
        ["nav/position_x", "nav/position_y", "nav/position_z"])
    stacked = np.column_stack([csv_run.signals[f"nav/position_{axis}"].v for axis in "xyz"])
    assert stacked.tobytes() == np.ascontiguousarray(h5.v).tobytes() == sim["nav/position"].tobytes()


def test_time_column_names_agree_across_formats(tmp_path):
    h5py = pytest.importorskip("h5py")
    arrays = {"q": np.array([5.0, 6.0, 7.0]), "tout": np.array([0.0, 1.0, 2.0]),
              "x": np.array([1.5, 2.5, 3.5])}
    csv_path = tmp_path / "run.csv"
    rows = "\n".join(f"{q},{t},{x}" for q, t, x in zip(arrays["q"], arrays["tout"], arrays["x"]))
    csv_path.write_text("q,tout,x\n" + rows + "\n")
    h5_path = tmp_path / "run.h5"
    with h5py.File(h5_path, "w") as f:
        for name, values in arrays.items():
            f[name] = values
    from_csv = open_source(csv_path).load()
    from_h5 = open_source(h5_path).load()
    assert list(from_csv.signals) == list(from_h5.signals) == ["q", "x"]
    for name in ("q", "x"):
        _assert_identical(from_csv.signals[name], from_h5.signals[name])
        assert from_csv.signals[name].t.tolist() == arrays["tout"].tolist()


# --------------------------------------------------------------------------------------------------------------
# open_source dispatch


@pytest.mark.parametrize("name", ["run.mat", "RUN.MAT"])
def test_mat_extension_selects_the_mat_reader(tmp_path, name):
    from baslt.sources.mat_src import MatSource

    path = tmp_path / name
    path.write_bytes(b"MATLAB 5.0 MAT-file" + b"\x00" * 200)
    assert isinstance(open_source(path), MatSource)
    with pytest.raises(SourceError, match="source file not found"):
        open_source(tmp_path / "missing.mat")


def test_mat_by_format_alias_or_header(tmp_path):
    from baslt.sources.mat_src import MatSource

    path = tmp_path / "run.bin"
    path.write_bytes(b"MATLAB 7.3 MAT-file, Platform: GLNXA64" + b"\x00" * 600)
    assert isinstance(open_source(path), MatSource)
    for alias in ("mat", "mat73", "MATLAB"):
        assert isinstance(open_source(path, format=alias), MatSource)
    assert base.available_formats()[:2] == ["numpy", "mat"]  # MAT is sniffed before plain HDF5


def test_missing_files_and_directories(tmp_path):
    for path in (tmp_path / "nope.csv", tmp_path / "nope.h5", tmp_path / "nope"):
        with pytest.raises(SourceError, match="source file not found"):
            open_source(path)
    with pytest.raises(SourceError, match="source file not found"):
        open_source(str(tmp_path / "nope.log"), format="csv")
    with pytest.raises(SourceError, match="is a directory"):
        open_source(tmp_path)


def test_unknown_and_undetectable_formats(tmp_path):
    path = tmp_path / "run.bin"
    path.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(UsageError, match="unknown source format 'parquet'"):
        open_source(path, format="parquet")
    with pytest.raises(SourceError, match="cannot determine the format"):
        open_source(path)
    with pytest.raises(UsageError, match="needs an in-memory mapping"):
        open_source(path, format="numpy")


def test_invalid_options_are_usage_errors(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("t,x\n0,1\n")
    with pytest.raises(UsageError, match="invalid option for csv sources"):
        open_source(path, compression="gzip")


def test_registered_adapter_is_used_for_its_extension(tmp_path):
    class DemoSource:
        format = "demo"
        extensions = (".demo",)

        def __init__(self, path):
            self.path = path

        @classmethod
        def sniff(cls, path, head):
            return head.startswith(b"DEMO")

        def list_signals(self):
            return []

        def load(self, names=None, **options):
            raise NotImplementedError

    register_adapter("demo", DemoSource)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_adapter("demo", DemoSource)
        (tmp_path / "a.demo").write_text("anything")
        assert isinstance(open_source(tmp_path / "a.demo"), DemoSource)
        (tmp_path / "b.raw").write_bytes(b"DEMO\x00\x00")
        assert isinstance(open_source(tmp_path / "b.raw"), DemoSource)
        assert "demo" in base.available_formats()
    finally:
        base._REGISTRY.pop("demo", None)


def test_importing_sources_does_not_import_numpy():
    code = "import sys, baslt.sources; print('numpy' in sys.modules, 'h5py' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"
