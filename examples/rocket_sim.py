"""Deterministic synthetic rocket ascent for examples, benchmarks and tests.

`simulate()` returns a dict of NumPy arrays keyed by signal path ("t", "prop/thrust", "aero/q", ...). The flight has
a hold-down release, a pitch program, max-q near 65 s, main engine cut-off (MECO) at 100 s and a coast phase.
The same seed and arguments always give identical arrays.

Write a run to disk:

    python examples/rocket_sim.py --format h5 --out examples/out/run.h5
    python examples/rocket_sim.py --format csv --out examples/out/run.csv --anomaly q_spike
    python examples/rocket_sim.py --format mat73 --out examples/out/run.mat
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

UNITS: dict[str, str | None] = {
    "t": "s",
    "prop/thrust": "N",
    "aero/q": "Pa",
    "aero/alpha": "deg",
    "nav/position": "m",
    "nav/velocity": "m/s",
    "gnc/mode": None,
    "thermal/skin_temp": "K",
    "gnc/elevon_cmd": "deg",
}

ANOMALIES: tuple[str, ...] = ("q_spike", "alpha_chatter")

# gnc/mode codes
MODE_VERTICAL_RISE = 0
MODE_PITCH_PROGRAM = 1
MODE_LOAD_RELIEF = 2
MODE_COAST = 3

G0 = 9.80665
THRUST_SEA_LEVEL = 2.0e6        # N
THRUST_VACUUM_GAIN = 0.12       # fraction gained as the ambient pressure drops
LIFTOFF_FRACTION = 0.6          # thrust fraction at hold-down release, full after RAMP_TIME
RAMP_TIME = 1.0                 # s
MECO_TIME = 100.0               # s
SHUTDOWN_TAU = 0.15             # s
MASS_LIFTOFF = 1.3e5            # kg
MASS_FLOW = 800.0               # kg/s
PITCH_START = 8.0               # s
PITCH_TOTAL = 72.0              # deg of pitch-over at MECO
RHO0 = 1.4357                   # kg/m^3, effective sea-level density of the aero model
SCALE_HEIGHT = 9000.0           # m
MAX_Q_TARGET = 66_800.0         # Pa, approximate nominal max-q

Q_SPIKE_START = 61.895          # s
Q_SPIKE_END = 62.105            # s
Q_SPIKE_RAMP = 0.010            # s
Q_SPIKE_AMPLITUDE = 6_000.0     # Pa

ALPHA_EXCURSIONS: tuple[tuple[float, float, float], ...] = (
    (23.4, 9.0, 0.030),         # (center s, peak deg, width s)
    (58.2, -8.6, 0.025),
    (87.9, 8.2, 0.035),
)
ALPHA_CHATTER_START = 70.0      # s
ALPHA_CHATTER_END = 71.5        # s
ALPHA_CHATTER_HZ = 30.0

SKIN_GAP_START = 45.0           # s
SKIN_GAP_END = 45.35            # s


def simulate(seed: int = 7, duration: float = 120.0, rate: float = 1000.0,
             anomaly: str | None = None, *, thrust_scale: float = 1.0,
             payload: float = 0.0) -> dict[str, np.ndarray]:
    """Simulate an ascent sampled at `rate` Hz from 0 to `duration` s (both ends included).

    `thrust_scale` multiplies the engine thrust and `payload` (kg) adds to the lift-off mass; the defaults give the
    nominal vehicle.
    """
    if anomaly is not None and anomaly not in ANOMALIES:
        raise ValueError(f"unknown anomaly {anomaly!r}; expected one of {', '.join(ANOMALIES)}")
    if not duration > 0 or not rate > 0:
        raise ValueError("duration and rate must be positive")
    if not thrust_scale > 0 or not payload >= 0:
        raise ValueError("thrust_scale must be positive and payload not negative")
    n = int(round(duration * rate)) + 1
    t = np.arange(n, dtype=np.float64) / rate
    dt = 1.0 / rate

    # All random draws happen here, in a fixed order, so anomalies never change the other signals.
    rng = np.random.default_rng(seed)
    thrust_noise = _smooth(rng.standard_normal(n), rate, 0.05, keep_std=True)
    q_noise = rng.standard_normal(n)
    alpha_noise = rng.standard_normal(n)
    elevon_noise = rng.standard_normal(n)
    temp_noise = rng.standard_normal(n)

    # Propulsion
    burning = t < MECO_TIME
    ramp = np.minimum(LIFTOFF_FRACTION + (1.0 - LIFTOFF_FRACTION) * t / RAMP_TIME, 1.0)
    vacuum = THRUST_SEA_LEVEL * (1.0 + THRUST_VACUUM_GAIN * (1.0 - np.exp(-t / 40.0)))
    shutdown = np.exp(-np.maximum(t - MECO_TIME, 0.0) / SHUTDOWN_TAU)
    thrust = thrust_scale * vacuum * np.where(burning, ramp, shutdown) * (1.0 + 0.002 * thrust_noise)

    # Point-mass ascent along a pitch program in the x-z plane with a slight cross-range drift
    mass = MASS_LIFTOFF + payload - MASS_FLOW * np.minimum(t, MECO_TIME)
    w = np.clip((t - PITCH_START) / (MECO_TIME - PITCH_START), 0.0, 1.0)
    gamma = np.deg2rad(90.0 - PITCH_TOTAL * (3.0 * w**2 - 2.0 * w**3))
    accel = thrust / mass - G0 * np.sin(gamma)
    speed = np.maximum(_integrate(accel, dt), 0.0)
    velocity = np.column_stack([speed * np.cos(gamma), 0.002 * speed * np.cos(gamma), speed * np.sin(gamma)])
    position = np.column_stack([_integrate(velocity[:, k], dt) for k in range(3)])
    altitude = position[:, 2]

    # Aerodynamics
    rho = RHO0 * np.exp(-altitude / SCALE_HEIGHT)
    q_nominal = 0.5 * rho * speed**2
    q = np.maximum(q_nominal + 40.0 * q_noise, 0.0)

    alpha_nominal = 1.2 * np.sin(2.0 * np.pi * t / 17.0) + 0.5 * alpha_noise
    for center, peak, width in ALPHA_EXCURSIONS:
        alpha_nominal = alpha_nominal + peak * np.exp(-0.5 * ((t - center) / width) ** 2)
    alpha = alpha_nominal

    # Guidance and control
    mode = np.full(n, MODE_VERTICAL_RISE, dtype=np.int64)
    mode[t >= 6.0] = MODE_PITCH_PROGRAM
    mode[(t >= 55.0) & (t < 75.0)] = MODE_LOAD_RELIEF
    mode[t >= MECO_TIME] = MODE_COAST
    elevon = np.clip(-0.8 * _smooth(alpha_nominal, rate, 0.05) + 0.5 * np.sin(2.0 * np.pi * t / 9.0), -20.0, 20.0)
    elevon = elevon + 0.05 * elevon_noise

    # Thermal: first-order lag of an aerodynamic heating target, with a sensor dropout
    heating_target = 288.15 + 0.3 * speed**2 / 2009.0 * np.exp(-altitude / 20000.0)
    skin_temp = _lag(heating_target, t, 15.0, 288.15) + 0.2 * temp_noise
    skin_temp[(t >= SKIN_GAP_START) & (t < SKIN_GAP_END)] = np.nan

    if anomaly == "q_spike":
        edge = np.minimum(t - Q_SPIKE_START, Q_SPIKE_END - t) / Q_SPIKE_RAMP
        q = q + Q_SPIKE_AMPLITUDE * np.clip(edge, 0.0, 1.0)
    elif anomaly == "alpha_chatter":
        window = (t >= ALPHA_CHATTER_START) & (t < ALPHA_CHATTER_END)
        chatter = 7.0 + 0.35 * np.sin(2.0 * np.pi * ALPHA_CHATTER_HZ * t) + 0.05 * alpha_noise
        alpha = np.where(window, chatter, alpha_nominal)

    return {
        "t": t,
        "prop/thrust": thrust,
        "aero/q": q,
        "aero/alpha": alpha,
        "nav/position": np.ascontiguousarray(position),
        "nav/velocity": np.ascontiguousarray(velocity),
        "gnc/mode": mode,
        "thermal/skin_temp": skin_temp,
        "gnc/elevon_cmd": elevon,
    }


def _integrate(y: np.ndarray, dt: float) -> np.ndarray:
    """Cumulative trapezoidal integral starting at 0."""
    out = np.empty_like(y)
    out[0] = 0.0
    np.cumsum(0.5 * (y[1:] + y[:-1]) * dt, out=out[1:])
    return out


def _smooth(x: np.ndarray, rate: float, window_s: float, *, keep_std: bool = False) -> np.ndarray:
    """Centered moving average; `keep_std` rescales white noise back to unit variance.

    The window is clamped to the sample count so that runs shorter than the window still simulate.
    """
    width = max(1, min(x.size, int(round(window_s * rate))))
    out = np.convolve(x, np.full(width, 1.0 / width), mode="same")[: x.size]
    return out * np.sqrt(width) if keep_std else out


def _lag(x: np.ndarray, t: np.ndarray, tau: float, y0: float) -> np.ndarray:
    """First-order lag y' = (x - y) / tau, solved in closed form with a cumulative integral."""
    growth = np.exp(t / tau)
    return (y0 + _integrate(growth * x / tau, float(t[1] - t[0]) if t.size > 1 else 0.0)) / growth


# --------------------------------------------------------------------------------------------------------------
# Writers


def _ordered(data: Mapping[str, np.ndarray]) -> list[str]:
    return (["t"] if "t" in data else []) + [k for k in data if k != "t"]


def csv_columns(data: Mapping[str, np.ndarray], units: Mapping[str, str | None] | None = None
                ) -> list[tuple[str, str | None, np.ndarray]]:
    """Flatten data into CSV columns (name, unit, values); vector signals become name_x, name_y, name_z."""
    units = UNITS if units is None else units
    columns = []
    for key in _ordered(data):
        values = np.asarray(data[key])
        unit = units.get(key)
        if values.ndim == 2:
            k = values.shape[1]
            suffixes = ("_x", "_y", "_z") if k == 3 else tuple(f"_{j}" for j in range(k))
            columns.extend((f"{key}{s}", unit, values[:, j]) for j, s in enumerate(suffixes))
        else:
            columns.append((key, unit, values))
    return columns


def write_csv(path: str | Path, data: Mapping[str, np.ndarray], *, units: Mapping[str, str | None] | None = None,
              block_rows: int = 16384) -> Path:
    """Write a comma-separated file with "name [unit]" headers. Floats use shortest round-trip formatting."""
    path = Path(path)
    columns = csv_columns(data, units)
    header = ",".join(f"{name} [{unit}]" if unit else name for name, unit, _ in columns)
    n = columns[0][2].shape[0] if columns else 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + "\n")
        for start in range(0, n, block_rows):
            stop = min(start + block_rows, n)
            text = [values[start:stop].astype(str) for _, _, values in columns]
            lines = text[0]
            for col in text[1:]:
                lines = np.char.add(np.char.add(lines, ","), col)
            fh.write("\n".join(lines.tolist()))
            fh.write("\n")
    return path


def write_h5(path: str | Path, data: Mapping[str, np.ndarray], *, chunked: bool = False,
             compression: str | None = None, units: Mapping[str, str | None] | None = None) -> Path:
    """Write an HDF5 file: one dataset per signal path, a top-level "t", `units` and `time` attributes."""
    try:
        import h5py
    except ImportError:
        raise SystemExit("writing HDF5 needs h5py (pip install h5py)") from None
    path = Path(path)
    units = UNITS if units is None else units
    with h5py.File(path, "w") as f:
        for key in _ordered(data):
            values = np.ascontiguousarray(data[key])
            options = {}
            if (chunked or compression) and values.size:
                options["chunks"] = (min(values.shape[0], 16384), *values.shape[1:])
                if compression:
                    options["compression"] = compression
            dset = f.create_dataset(key, data=values, **options)
            unit = units.get(key)
            if unit:
                dset.attrs["units"] = unit
            if key != "t" and "t" in data:
                dset.attrs["time"] = "/t"
    return path


MAT73_HEADER_TEXT = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Thu Jan  1 00:00:00 1970 HDF5 schema 1.00 ."
MATLAB_CLASSES: dict[str, str] = {
    "f8": "double", "f4": "single", "i1": "int8", "u1": "uint8", "i2": "int16", "u2": "uint16",
    "i4": "int32", "u4": "uint32", "i8": "int64", "u8": "uint64",
}


def mat_tree(data: Mapping[str, np.ndarray]) -> dict:
    """Nest "group/name" keys into structs; the clock "t" becomes the top-level "tout" Simulink writes."""
    tree: dict = {}
    for key in _ordered(data):
        parts = ["tout"] if key == "t" else key.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = np.asarray(data[key])
    return tree


def write_mat(path: str | Path, data: Mapping[str, np.ndarray], *, version: str = "5",
              compression: bool = True) -> Path:
    """Write a MATLAB MAT-file with a top-level "tout" clock and one struct per signal group.

    version "5" uses scipy (readable by every MATLAB release); "7.3" writes MATLAB's HDF5 layout with h5py.
    MAT-files hold no units, so a policy for this file declares them under signals.decl.
    """
    path = Path(path)
    tree = mat_tree(data)
    if version == "5":
        try:
            import scipy.io
        except ImportError:
            raise SystemExit("writing MAT-files needs scipy (pip install scipy)") from None
        scipy.io.savemat(path, tree, oned_as="column", do_compression=compression, long_field_names=True)
    elif version == "7.3":
        _write_mat73(path, tree, compression)
    else:
        raise ValueError(f"unknown MAT-file version {version!r}; expected '5' or '7.3'")
    return path


def _write_mat73(path: Path, tree: Mapping, compression: bool) -> None:
    try:
        import h5py
    except ImportError:
        raise SystemExit("writing MAT-files needs h5py (pip install h5py)") from None
    with h5py.File(path, "w", userblock_size=512, track_order=True) as f:
        for name, value in tree.items():
            _write_mat73_node(f, name, value, compression, h5py)
    header = MAT73_HEADER_TEXT.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"
    with open(path, "r+b") as fh:
        fh.write(header)


def _write_mat73_node(group, name: str, value, compression: bool, h5py) -> None:
    if isinstance(value, Mapping):
        sub = group.create_group(name, track_order=True)
        sub.attrs["MATLAB_class"] = np.bytes_("struct")
        for key, child in value.items():
            _write_mat73_node(sub, key, child, compression, h5py)
        fields = np.empty(len(value), dtype=object)
        for i, key in enumerate(value):
            fields[i] = np.frombuffer(key.encode("ascii"), dtype="S1")
        sub.attrs.create("MATLAB_fields", fields, dtype=h5py.vlen_dtype(np.dtype("S1")))
        return
    arr = np.asarray(value)
    logical = arr.dtype == np.bool_
    if logical:
        arr = arr.astype(np.uint8)
    matlab = arr.reshape(-1, 1) if arr.ndim == 1 else arr  # MATLAB column vectors are n-by-1
    stored = np.ascontiguousarray(matlab.T)  # HDF5 dimensions are MATLAB's, reversed
    options = {"compression": "gzip", "compression_opts": 3} if compression and stored.size >= 64 else {}
    dset = group.create_dataset(name, data=stored, **options)
    dset.attrs["MATLAB_class"] = np.bytes_("logical" if logical else MATLAB_CLASSES[arr.dtype.str[1:]])
    if logical:
        dset.attrs["MATLAB_int_decode"] = np.int32(1)


# --------------------------------------------------------------------------------------------------------------
# Command line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a synthetic rocket ascent run to CSV, HDF5 or a MATLAB MAT-file.")
    parser.add_argument("--format", choices=("csv", "h5", "hdf5", "mat", "mat73"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="output file path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=120.0, help="seconds")
    parser.add_argument("--rate", type=float, default=1000.0, help="samples per second")
    parser.add_argument("--anomaly", choices=ANOMALIES, default=None)
    parser.add_argument("--chunked", action="store_true", help="HDF5: chunked datasets")
    parser.add_argument("--compression", choices=("gzip",), default=None, help="HDF5: dataset compression")
    args = parser.parse_args(argv)

    try:
        data = simulate(seed=args.seed, duration=args.duration, rate=args.rate, anomaly=args.anomaly)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        write_csv(args.out, data)
    elif args.format in ("mat", "mat73"):
        write_mat(args.out, data, version="7.3" if args.format == "mat73" else "5")
    else:
        write_h5(args.out, data, chunked=args.chunked, compression=args.compression)
    print(f"wrote {args.out} ({data['t'].size} samples, {args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
