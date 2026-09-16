"""The rocket example writes MATLAB MAT-files in both versions, in the layout MATLAB uses."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
sio = pytest.importorskip("scipy.io")

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
def data():
    return rs.simulate(duration=3.0, rate=50.0)


def test_mat_tree_nests_groups_and_renames_the_clock(data):
    tree = rs.mat_tree(data)
    assert list(tree)[0] == "tout" and "t" not in tree
    assert set(tree["aero"]) == {"q", "alpha"}
    assert tree["nav"]["position"].shape == (data["t"].size, 3)


def test_version_5_is_readable_by_scipy(tmp_path, data):
    path = rs.write_mat(tmp_path / "run.mat", data, version="5")
    assert path.read_bytes().startswith(b"MATLAB 5.0 MAT-file")
    loaded = sio.loadmat(path, simplify_cells=True)
    assert np.array_equal(loaded["tout"], data["t"])
    assert np.array_equal(loaded["aero"]["q"], data["aero/q"])
    assert loaded["gnc"]["mode"].dtype == data["gnc/mode"].dtype
    info = {name: (shape, cls) for name, shape, cls in sio.whosmat(path)}
    assert info["tout"] == ((data["t"].size, 1), "double")  # a MATLAB column vector


def test_version_73_follows_the_matlab_layout(tmp_path, data):
    path = rs.write_mat(tmp_path / "run.mat", data, version="7.3")
    raw = path.read_bytes()
    assert raw.startswith(b"MATLAB 7.3 MAT-file")
    assert raw[124:128] == b"\x00\x02IM"
    assert raw[512:520] == b"\x89HDF\r\n\x1a\n"
    n = data["t"].size
    with h5py.File(path, "r") as f:
        assert f["tout"].shape == (1, n)  # n-by-1 in MATLAB, dimensions reversed in HDF5
        assert f["tout"].attrs["MATLAB_class"] == b"double"
        assert f["nav/position"].shape == (3, n)
        assert np.array_equal(f["nav/position"][()].T, data["nav/position"])
        assert f["aero"].attrs["MATLAB_class"] == b"struct"
        fields = [b"".join(x.tolist()).decode() for x in f["aero"].attrs["MATLAB_fields"]]
        assert fields == list(rs.mat_tree(data)["aero"])
        assert f["gnc/mode"].attrs["MATLAB_class"] == b"int64"


def test_logical_values_are_written_as_matlab_logical(tmp_path):
    t = np.arange(4.0)
    path = rs.write_mat(tmp_path / "flags.mat", {"t": t, "gnc/armed": t > 1}, version="7.3")
    with h5py.File(path, "r") as f:
        assert f["gnc/armed"].attrs["MATLAB_class"] == b"logical"
        assert f["gnc/armed"].dtype == np.uint8


def test_unknown_version_is_rejected(tmp_path, data):
    with pytest.raises(ValueError, match="unknown MAT-file version"):
        rs.write_mat(tmp_path / "x.mat", data, version="7")


@pytest.mark.parametrize("fmt, header", [("mat", b"MATLAB 5.0"), ("mat73", b"MATLAB 7.3")])
def test_command_line_writes_mat_files(tmp_path, fmt, header):
    out = tmp_path / f"{fmt}.mat"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--format", fmt, "--out", str(out), "--duration", "2", "--rate", "20"],
        capture_output=True, text=True, check=True,
    )
    assert "41 samples" in result.stdout
    assert out.read_bytes().startswith(header)
