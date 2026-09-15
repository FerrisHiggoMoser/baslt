"""The synthetic rocket ascent is deterministic and has the documented flight features."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "rocket_sim.py"
KEYS = {
    "t", "prop/thrust", "aero/q", "aero/alpha", "nav/position", "nav/velocity", "gnc/mode", "thermal/skin_temp",
    "gnc/elevon_cmd",
}


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
def nominal():
    return rs.simulate()


@pytest.fixture(scope="module")
def q_spike():
    return rs.simulate(anomaly="q_spike")


@pytest.fixture(scope="module")
def alpha_chatter():
    return rs.simulate(anomaly="alpha_chatter")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of each run of True values."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), stops.tolist()))


def _crossings(x: np.ndarray, level: float) -> int:
    return int(np.count_nonzero(np.diff((x >= level).astype(np.int8))))


def test_same_seed_gives_identical_arrays():
    a = rs.simulate(seed=3, duration=20.0, rate=200.0)
    b = rs.simulate(seed=3, duration=20.0, rate=200.0)
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].dtype == b[key].dtype
        assert a[key].tobytes() == b[key].tobytes(), key
    c = rs.simulate(seed=4, duration=20.0, rate=200.0)
    assert not np.array_equal(a["aero/alpha"], c["aero/alpha"])


def test_default_run_layout(nominal):
    assert set(nominal) == KEYS
    assert set(rs.UNITS) == KEYS
    t = nominal["t"]
    assert t.shape == (120_001,) and t[0] == 0.0 and t[-1] == 120.0
    assert np.all(np.diff(t) > 0)
    for key, values in nominal.items():
        assert values.flags.c_contiguous, key
        assert values.shape[0] == t.shape[0], key
        expected = np.int64 if key == "gnc/mode" else np.float64
        assert values.dtype == expected, key
    assert nominal["nav/position"].shape == (120_001, 3)
    assert nominal["nav/velocity"].shape == (120_001, 3)


def test_thrust_and_meco(nominal):
    t, thrust = nominal["t"], nominal["prop/thrust"]
    burn = thrust[(t > 2.0) & (t < 99.0)]
    assert 1.8e6 < np.median(burn) < 2.5e6
    below = np.flatnonzero(thrust < 100.0)
    meco = t[below[0]]
    assert 100.0 <= meco <= 102.5
    assert np.all(thrust[: below[0]] >= 100.0)
    assert np.all(thrust[below[0]:] < 100.0)


def test_max_q_and_crossings(nominal):
    t, q = nominal["t"], nominal["aero/q"]
    assert 60_000.0 <= q.max() <= 70_000.0
    assert 55.0 <= t[np.argmax(q)] <= 75.0
    assert _crossings(q, 65_000.0) >= 2
    assert np.all(q >= 0.0)


def test_alpha_has_a_few_short_excursions(nominal):
    t, alpha = nominal["t"], nominal["aero/alpha"]
    beyond = np.flatnonzero(np.abs(alpha) > 7.0)
    assert beyond.size > 0
    groups = np.split(beyond, np.flatnonzero(np.diff(t[beyond]) > 1.0) + 1)
    assert 2 <= len(groups) <= 5
    for group in groups:
        assert t[group[-1]] - t[group[0]] < 0.25
    signs = {bool(np.sign(alpha[g[0]]) > 0) for g in groups}
    assert signs == {True, False}
    assert np.std(np.diff(alpha)) > 0.3  # sample-to-sample noise


def test_navigation_is_a_smooth_ascent(nominal):
    t, pos, vel = nominal["t"], nominal["nav/position"], nominal["nav/velocity"]
    burn = t < 100.0
    assert np.all(np.diff(pos[burn, 2]) >= 0.0)
    assert pos[-1, 2] > 30_000.0 and pos[-1, 0] > 30_000.0
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))
    derivative = np.gradient(pos[:, 2], t)
    assert np.max(np.abs(derivative[1:-1] - vel[1:-1, 2])) < 1.0


def test_mode_codes(nominal):
    mode = nominal["gnc/mode"]
    assert set(np.unique(mode).tolist()) <= {0, 1, 2, 3}
    assert mode[0] == 0 and mode[-1] == 3
    assert 3 <= np.count_nonzero(np.diff(mode)) <= 6


def test_skin_temperature_has_one_short_gap(nominal):
    t, skin = nominal["t"], nominal["thermal/skin_temp"]
    gaps = _runs(np.isnan(skin))
    assert len(gaps) == 1
    first, last = gaps[0]
    assert 0.1 <= t[last] - t[first] <= 1.0
    finite = skin[np.isfinite(skin)]
    assert 250.0 < finite.min() and finite.max() < 1000.0


def test_elevon_command(nominal):
    elevon = nominal["gnc/elevon_cmd"]
    assert np.all(np.isfinite(elevon))
    assert np.max(np.abs(elevon)) <= 20.5


def test_q_spike_anomaly(nominal, q_spike):
    t, q = q_spike["t"], q_spike["aero/q"]
    assert q.max() >= 70_000.0
    runs = _runs(q > 70_000.0)
    assert len(runs) == 1
    first, last = runs[0]
    assert 0.15 <= t[last] - t[first] + 0.001 <= 0.25
    for key in KEYS - {"aero/q"}:
        assert q_spike[key].tobytes() == nominal[key].tobytes(), key


def test_alpha_chatter_anomaly(nominal, alpha_chatter):
    t, alpha = alpha_chatter["t"], alpha_chatter["aero/alpha"]
    window = (t >= 69.0) & (t < 73.0)
    assert _crossings(alpha[window], 7.0) >= 20
    assert _crossings(nominal["aero/alpha"][window], 7.0) == 0
    for key in KEYS - {"aero/alpha"}:
        assert alpha_chatter[key].tobytes() == nominal[key].tobytes(), key


def test_runs_shorter_than_the_smoothing_window(tmp_path):
    data = rs.simulate(duration=0.04, rate=1000.0)  # 41 samples, below the 50-sample smoothing window
    assert set(data) == KEYS
    for key, values in data.items():
        assert values.shape[0] == 41, key
        assert np.all(np.isfinite(values)), key
    assert rs.simulate(duration=0.001, rate=1000.0)["t"].shape == (2,)
    out = tmp_path / "short.csv"
    assert rs.main(["--format", "csv", "--out", str(out), "--duration", "0.04", "--rate", "1000"]) == 0
    assert len(out.read_text().splitlines()) == 42


def test_invalid_arguments():
    with pytest.raises(ValueError, match="unknown anomaly"):
        rs.simulate(anomaly="engine_fire")
    with pytest.raises(ValueError):
        rs.simulate(duration=0.0)


def test_write_csv_headers(tmp_path):
    data = rs.simulate(duration=1.0, rate=10.0)
    path = rs.write_csv(tmp_path / "run.csv", data)
    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    assert header[0] == "t [s]"
    assert "nav/position_x [m]" in header and "nav/position_z [m]" in header
    assert "nav/velocity_y [m/s]" in header
    assert "gnc/mode" in header
    assert len(lines) == 1 + data["t"].size
    assert len(header) == 13  # t, thrust, q, alpha, 3 position, 3 velocity, mode, skin_temp, elevon


def test_write_h5_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    data = rs.simulate(duration=1.0, rate=10.0)
    path = rs.write_h5(tmp_path / "run.h5", data, chunked=True, compression="gzip")
    with h5py.File(path, "r") as f:
        assert f["t"].attrs["units"] == "s"
        assert f["aero/q"].attrs["units"] == "Pa"
        assert f["aero/q"].attrs["time"] == "/t"
        assert "units" not in f["gnc/mode"].attrs
        assert f["nav/position"].shape == (11, 3)
        assert f["aero/q"].compression == "gzip" and f["aero/q"].chunks is not None
        np.testing.assert_array_equal(f["gnc/mode"][()], data["gnc/mode"])


def test_cli_writes_csv(tmp_path, capsys):
    out = tmp_path / "sub" / "run.csv"
    assert rs.main(["--format", "csv", "--out", str(out), "--duration", "2", "--rate", "50", "--seed", "1"]) == 0
    assert out.exists()
    assert "101 samples" in capsys.readouterr().out
    assert out.read_text().startswith("t [s],")


def test_cli_script_writes_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    out = tmp_path / "run.h5"
    subprocess.run(
        [sys.executable, str(EXAMPLE), "--format", "h5", "--out", str(out), "--duration", "1", "--rate", "20",
         "--anomaly", "q_spike", "--chunked", "--compression", "gzip"],
        check=True, capture_output=True, text=True,
    )
    with h5py.File(out, "r") as f:
        assert f["t"].shape == (21,)
        assert f["prop/thrust"].chunks is not None
