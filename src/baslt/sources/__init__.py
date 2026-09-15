"""Source adapters: open simulation output (in-memory arrays, CSV, HDF5) as runs of signals.

Importing this package does not import numpy; adapters load when a format is first used.
"""

from __future__ import annotations

from importlib import import_module

from .base import SourceAdapter, adapter_class, available_formats, open_source, register_adapter

_LAZY: dict[str, str] = {
    "NumpySource": "baslt.sources.numpy_src",
    "CsvSource": "baslt.sources.csv_src",
    "Hdf5Source": "baslt.sources.hdf5_src",
}

__all__ = ["SourceAdapter", "adapter_class", "available_formats", "open_source", "register_adapter", *_LAZY]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'baslt.sources' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
