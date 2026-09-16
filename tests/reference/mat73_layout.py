"""Write small MATLAB v7.3 MAT-files for tests, following the layout MATLAB itself writes.

This is independent of the example writer in examples/rocket_sim.py and covers what a reader must handle:
numeric classes, char, logical, empty values, cell arrays, struct arrays, sparse matrices, complex numbers and
MATLAB objects. Arrays are given in MATLAB shape (rows, columns); a 1-D array is a MATLAB column vector.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

HEADER = (
    b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Tue Sep 15 12:00:00 2026 HDF5 schema 1.00 ."
).ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"

CLASSES = {
    "f8": "double", "f4": "single", "i1": "int8", "u1": "uint8", "i2": "int16", "u2": "uint16",
    "i4": "int32", "u4": "uint32", "i8": "int64", "u8": "uint64",
}


def _class_attr(obj, name: str) -> None:
    obj.attrs["MATLAB_class"] = np.bytes_(name)


def _matlab_2d(values) -> np.ndarray:
    arr = np.asarray(values)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


class Mat73Writer:
    """Context manager building one file. Every method takes the parent group first."""

    def __init__(self, path: str | Path, *, header: bytes = HEADER) -> None:
        self.path = Path(path)
        self.header = header
        self.f = h5py.File(self.path, "w", userblock_size=512, track_order=True)
        self.refs = self.f.create_group("#refs#")
        self._count = 0

    def __enter__(self) -> "Mat73Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.f.close()
        with open(self.path, "r+b") as fh:
            fh.write(self.header)

    @property
    def root(self):
        return self.f

    def _ref_name(self) -> str:
        self._count += 1
        return f"r{self._count}"

    # ---------------------------------------------------------------------------------------------- values

    def numeric(self, parent, name: str, values, *, cls: str | None = None, **options):
        arr = _matlab_2d(values)
        dset = parent.create_dataset(name, data=np.ascontiguousarray(arr.T), **options)
        _class_attr(dset, cls or CLASSES[arr.dtype.str[1:]])
        return dset

    def logical(self, parent, name: str, values):
        arr = _matlab_2d(np.asarray(values, dtype=bool)).astype(np.uint8)
        dset = parent.create_dataset(name, data=np.ascontiguousarray(arr.T))
        _class_attr(dset, "logical")
        dset.attrs["MATLAB_int_decode"] = np.int32(1)
        return dset

    def char(self, parent, name: str, text: str):
        if not text:
            return self.empty(parent, name, cls="char", dims=(1, 0))
        codes = np.frombuffer(text.encode("utf-16-le"), dtype="<u2").reshape(1, -1)  # a 1-by-L row
        dset = parent.create_dataset(name, data=np.ascontiguousarray(codes.T))
        _class_attr(dset, "char")
        dset.attrs["MATLAB_int_decode"] = np.int32(2)
        return dset

    def empty(self, parent, name: str, *, cls: str = "double", dims=(0, 0)):
        dset = parent.create_dataset(name, data=np.asarray(dims, dtype=np.uint64))
        _class_attr(dset, cls)
        dset.attrs["MATLAB_empty"] = np.uint8(1)
        return dset

    def complex(self, parent, name: str, values):
        arr = np.asarray(values, dtype=np.complex128).reshape(-1, 1)
        compound = np.dtype([("real", "<f8"), ("imag", "<f8")])
        data = np.empty(arr.T.shape, dtype=compound)
        data["real"] = arr.T.real
        data["imag"] = arr.T.imag
        dset = parent.create_dataset(name, data=data)
        _class_attr(dset, "double")
        return dset

    def sparse(self, parent, name: str):
        group = parent.create_group(name)
        _class_attr(group, "double")
        group.attrs["MATLAB_sparse"] = np.uint64(3)
        group.create_dataset("data", data=np.array([1.0, 2.0]))
        group.create_dataset("ir", data=np.array([0, 2], dtype=np.uint64))
        group.create_dataset("jc", data=np.array([0, 1, 2], dtype=np.uint64))
        return group

    def object(self, parent, name: str, cls: str = "timeseries"):
        dset = parent.create_dataset(name, data=np.array([[3707764736], [2], [1], [1], [1], [1]], dtype=np.uint32))
        _class_attr(dset, cls)
        dset.attrs["MATLAB_object_decode"] = np.int32(3)
        return dset

    # ------------------------------------------------------------------------------------------ containers

    def struct(self, parent, name: str, fields: list[str] | None = None):
        group = parent.create_group(name, track_order=True)
        _class_attr(group, "struct")
        if fields is not None:
            self.set_fields(group, fields)
        return group

    @staticmethod
    def set_fields(group, names: list[str]) -> None:
        packed = np.empty(len(names), dtype=object)
        for i, field in enumerate(names):
            packed[i] = np.frombuffer(field.encode("ascii"), dtype="S1")
        group.attrs.create("MATLAB_fields", packed, dtype=h5py.vlen_dtype(np.dtype("S1")))

    def _target(self, value):
        """Write one referenced value into #refs# and return its reference."""
        name = self._ref_name()
        if isinstance(value, str):
            dset = self.char(self.refs, name, value)
        elif value is None:
            dset = self.empty(self.refs, name)
        else:
            dset = self.numeric(self.refs, name, value)
        return dset.ref

    def cell(self, parent, name: str, values: list, *, shape: tuple[int, int] | None = None):
        rows, cols = shape or (1, len(values))
        refs = np.empty((cols, rows), dtype=h5py.ref_dtype)  # HDF5 order is MATLAB order reversed
        for i, value in enumerate(values):
            refs[i // rows, i % rows] = self._target(value)
        dset = parent.create_dataset(name, data=refs, dtype=h5py.ref_dtype)
        _class_attr(dset, "cell")
        return dset

    def struct_array(self, parent, name: str, elements: list[dict]):
        """A 1-by-n struct array: each field is a dataset of references, one per element."""
        group = parent.create_group(name, track_order=True)
        _class_attr(group, "struct")
        names = list(elements[0])
        for field in names:
            refs = np.empty((len(elements), 1), dtype=h5py.ref_dtype)
            for i, element in enumerate(elements):
                refs[i, 0] = self._target(element[field])
            group.create_dataset(field, data=refs, dtype=h5py.ref_dtype)
        self.set_fields(group, names)
        return group
