"""Run parameters stored in the run file itself: HDF5 scalars and attributes, MAT-file scalars and texts."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.sources import open_source
from reference.reqs_check import load_rows
from reference.reqs_runs import run as make_run

T = np.linspace(0.0, 10.0, 101)
X = 6.0 * np.sin(np.pi * T / 10.0)
ROWS = [{"id": "R", "check": "x", "case": "heavy", "applies_to": "vehicle == 'heavy'", "limit": "<= 5"},
        {"id": "R", "case": "light", "limit": "<= 10"},
        {"id": "M", "check": "param.mass", "limit": "<= 2 t"}]
MAPPING = {"params": {"from_source": ["mass", "vehicle"], "units": {"mass": "kg"}}}


def check_file(tmp_path, path, params=None):
    from baslt.reqs.run import check_run

    reqset = load_rows(ROWS, MAPPING, folder=tmp_path)
    result, _ = check_run(path, reqset, params=params)
    assert result.error is None, result.error
    return result


def test_hdf5_parameters(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as f:
        f["t"] = T
        f["x"] = X
        f["params/mass"] = 1500.0
        f["params/count"] = np.int32(3)
        f["params/vehicle"] = "heavy"
        f["params/flag"] = np.bool_(True)
        f["params/vector"] = np.arange(3.0)
        f.attrs["vehicle_code"] = "LV-3"
        f.attrs["stage"] = 2
        f.attrs["matrix"] = np.ones((2, 2))
    params = open_source(path).parameters()
    assert params == {"vehicle_code": "LV-3", "stage": 2, "params/mass": 1500.0, "params/count": 3,
                      "params/vehicle": "heavy", "params/flag": True}
    result = check_file(tmp_path, path)
    assert result.params == {"mass": 1500.0, "vehicle": "heavy"}
    by_id = {r.id: r for r in result.results}
    assert by_id["R"].verdict == "fail" and by_id["R"].case == "heavy"
    assert by_id["M"].verdict == "pass" and by_id["M"].value == 1500.0
    table_wins = check_file(tmp_path, path, params={"vehicle": "light"})
    assert {r.id: r.verdict for r in table_wins.results}["R"] == "pass"


def test_mat_parameters(tmp_path):
    sio = pytest.importorskip("scipy.io")
    path = tmp_path / "run.mat"
    sio.savemat(path, {"t": T, "x": X, "mass": 1500.0, "vehicle": "heavy", "cfg": {"gain": 2.5, "name": "fast"},
                       "flags": np.array([[True]])})
    params = open_source(path).parameters()
    assert params == {"mass": 1500.0, "vehicle": "heavy", "cfg/gain": 2.5, "cfg/name": "fast", "flags": True}
    result = check_file(tmp_path, path)
    assert {r.id: r.verdict for r in result.results} == {"R": "fail", "M": "pass"}


@pytest.mark.minimal
def test_in_memory_parameters_and_repeated_leaves(tmp_path):
    from baslt.reqs.run import source_parameters

    adapter = open_source({"t": T, "x": X, "mass": 1500.0, "vehicle": "heavy"})
    assert adapter.parameters() == {"mass": 1500.0, "vehicle": "heavy"}

    class Fake:
        def parameters(self):
            return {"a/mass": 1.0, "b/mass": 2.0, "c/vehicle": "x", "c/other-name": 3}

    assert source_parameters(Fake(), ["mass", "c/*"]) == {"a_mass": 1.0, "b_mass": 2.0, "vehicle": "x",
                                                          "other_name": 3}
    assert source_parameters(Fake(), []) == {}
    assert source_parameters(object(), ["*"]) == {}


@pytest.mark.minimal
def test_csv_runs_have_no_parameters(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("t,x\n0,1\n1,2\n", encoding="utf-8")
    assert open_source(path).parameters() == {}


@pytest.mark.minimal
def test_lint_knows_source_parameters(tmp_path):
    from baslt.reqs.bind import lint_against_source

    reqset = load_rows(ROWS, MAPPING, folder=tmp_path)
    source = {"t": T, "x": X, "mass": 1500.0, "vehicle": "heavy"}
    items = lint_against_source(reqset, make_run(x=(T, X)), param_names={"other"})
    assert any("unknown run parameter 'vehicle'" in item.message for item in items)
    items = lint_against_source(reqset, source, param_names={"other"})
    assert not [item for item in items if item.level == "error"]
