"""Verifier-owned SI unit table, transcribed from the Units section of docs/policy.md.

Kept separate from the compiler's unit module on purpose: the verifier must not share code with the
compiler. SI value = number * factor + offset (absolute), or number * factor (delta).
"""

from __future__ import annotations

import math

# symbol -> (dimension, factor to SI base, offset added for absolute values)
SI_UNITS: dict[str, tuple[str, float, float]] = {
    # time (s)
    "s": ("time", 1.0, 0.0),
    "ms": ("time", 1e-3, 0.0),
    "us": ("time", 1e-6, 0.0),
    "µs": ("time", 1e-6, 0.0),  # MICRO SIGN + s
    "ns": ("time", 1e-9, 0.0),
    "min": ("time", 60.0, 0.0),
    "h": ("time", 3600.0, 0.0),
    # angle (rad)
    "rad": ("angle", 1.0, 0.0),
    "mrad": ("angle", 1e-3, 0.0),
    "deg": ("angle", math.pi / 180.0, 0.0),
    "°": ("angle", math.pi / 180.0, 0.0),  # DEGREE SIGN
    "degree": ("angle", math.pi / 180.0, 0.0),
    "degrees": ("angle", math.pi / 180.0, 0.0),
    # angular rate (rad/s)
    "rad/s": ("angular_rate", 1.0, 0.0),
    "deg/s": ("angular_rate", math.pi / 180.0, 0.0),
    # pressure (Pa)
    "Pa": ("pressure", 1.0, 0.0),
    "hPa": ("pressure", 100.0, 0.0),
    "kPa": ("pressure", 1e3, 0.0),
    "MPa": ("pressure", 1e6, 0.0),
    "bar": ("pressure", 1e5, 0.0),
    "mbar": ("pressure", 100.0, 0.0),
    "psi": ("pressure", 6894.757293168361, 0.0),
    # length (m)
    "m": ("length", 1.0, 0.0),
    "mm": ("length", 1e-3, 0.0),
    "cm": ("length", 1e-2, 0.0),
    "km": ("length", 1e3, 0.0),
    "ft": ("length", 0.3048, 0.0),
    "in": ("length", 0.0254, 0.0),
    "nmi": ("length", 1852.0, 0.0),
    # velocity (m/s)
    "m/s": ("velocity", 1.0, 0.0),
    "km/s": ("velocity", 1e3, 0.0),
    "km/h": ("velocity", 1.0 / 3.6, 0.0),
    "ft/s": ("velocity", 0.3048, 0.0),
    "kn": ("velocity", 1852.0 / 3600.0, 0.0),
    # acceleration (m/s^2)
    "m/s^2": ("acceleration", 1.0, 0.0),
    "m/s2": ("acceleration", 1.0, 0.0),
    "ft/s^2": ("acceleration", 0.3048, 0.0),
    # force (N)
    "N": ("force", 1.0, 0.0),
    "kN": ("force", 1e3, 0.0),
    "MN": ("force", 1e6, 0.0),
    "lbf": ("force", 4.4482216152605, 0.0),
    # mass (kg)
    "g": ("mass", 1e-3, 0.0),
    "kg": ("mass", 1.0, 0.0),
    "t": ("mass", 1e3, 0.0),
    # temperature (K)
    "K": ("temperature", 1.0, 0.0),
    "degC": ("temperature", 1.0, 273.15),
    "degF": ("temperature", 5.0 / 9.0, 255.37222222222223),
    # frequency (Hz)
    "Hz": ("frequency", 1.0, 0.0),
    "kHz": ("frequency", 1e3, 0.0),
    # dimensionless (1)
    "1": ("dimensionless", 1.0, 0.0),
    "%": ("dimensionless", 0.01, 0.0),
    # bytes (B)
    "B": ("bytes", 1.0, 0.0),
    "KB": ("bytes", 1e3, 0.0),
    "MB": ("bytes", 1e6, 0.0),
    "GB": ("bytes", 1e9, 0.0),
    "KiB": ("bytes", 1024.0, 0.0),
    "MiB": ("bytes", 1048576.0, 0.0),
    "GiB": ("bytes", 1073741824.0, 0.0),
}


def to_si(value: float, unit: str | None, *, delta: bool) -> float:
    """Convert `value` in `unit` to the SI base unit of its dimension.

    A bare number (`unit is None`) is returned unchanged. With `delta=True` the affine offset is ignored.
    Raises ValueError for a unit symbol that is not in the table.
    """
    if unit is None:
        return value
    entry = SI_UNITS.get(unit)
    if entry is None:
        raise ValueError(f"unknown unit {unit!r}")
    _, factor, offset = entry
    if delta:
        return value * factor
    return value * factor + offset


def dimension(unit: str | None) -> str | None:
    """Return the dimension name of `unit`, or None for a bare number or a unit not in the table (opaque)."""
    if unit is None:
        return None
    entry = SI_UNITS.get(unit)
    return None if entry is None else entry[0]
