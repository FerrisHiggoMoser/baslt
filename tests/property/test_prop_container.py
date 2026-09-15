"""Property tests: encodings, ZIP layout and full artifacts over random arrays and member orderings."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402
from reference.matlab_mirror import load, read_members  # noqa: E402

from baslt.container.encode import decode_array, encode_array  # noqa: E402
from baslt.container.reader import read_artifact  # noqa: E402
from baslt.container.spec import (  # noqa: E402
    DTYPE_WIDTHS,
    ArrayDesc,
    canonical_json,
    header_json,
    zstd_available,
)
from baslt.container.zipwriter import Member, compress_member, write_zip, zip_size  # noqa: E402

DTYPES = list(DTYPE_WIDTHS)
METHODS = [0, 8, 93] if zstd_available() else [0, 8]


@st.composite
def value_arrays(draw):
    dtype = draw(st.sampled_from(DTYPES))
    n = draw(st.integers(0, 40))
    k = draw(st.sampled_from([None, 1, 2, 3, 5]))
    shape = (n,) if k is None else (n, k)
    arr = draw(hnp.arrays(np.dtype(dtype), shape))
    return arr


@st.composite
def idx_arrays(draw):
    big = draw(st.booleans())
    deltas = draw(st.lists(st.integers(1, 2**20), max_size=40))
    start = draw(st.integers(0, 2**45 if big else 2**20))
    values = np.cumsum(np.array([start, *deltas], dtype=np.int64))
    if big:
        return (values + 2**33).astype("<f8")
    return values.astype("<u4")


def desc_of(arr: np.ndarray, enc: str, member: str, offset: int) -> ArrayDesc:
    n = arr.shape[0]
    k = 1 if arr.ndim == 1 else arr.shape[1]
    return ArrayDesc("v", member, offset, n * k * arr.dtype.itemsize, n, k, arr.dtype.str, enc)


@given(value_arrays(), st.sampled_from(["raw", "shuffle"]), st.integers(0, 16))
def test_value_round_trip(arr, enc, pad):
    buf = b"\x55" * pad + encode_array(arr, enc)
    out = decode_array(buf, desc_of(arr, enc, "s/0", pad))
    n = arr.shape[0]
    expected = arr.reshape(n) if arr.ndim == 2 and arr.shape[1] == 1 else arr
    assert out.shape == expected.shape and out.dtype == arr.dtype
    assert out.tobytes() == np.ascontiguousarray(expected).tobytes()


@given(idx_arrays())
def test_idx_round_trip(idx):
    out = decode_array(encode_array(idx, "delta+shuffle"), desc_of(idx, "delta+shuffle", "s/0", 0))
    assert out.dtype == np.float64
    assert [int(x) for x in out] == [int(x) for x in idx]


names = st.from_regex(r"[a-z0-9_][a-z0-9._]{0,6}(/[a-z0-9_][a-z0-9._]{0,6})?", fullmatch=True)
payloads = st.one_of(
    st.binary(max_size=600),
    st.builds(lambda chunk, reps: chunk * reps, st.binary(min_size=1, max_size=16), st.integers(1, 200)),
)


@given(st.lists(st.tuples(names, payloads, st.sampled_from(METHODS)), min_size=1, max_size=12,
                unique_by=lambda m: m[0]))
def test_zip_layout_properties(specs):
    members = [Member(name, data, method) for name, data, method in specs]
    out = write_zip(members)
    compressed = [compress_member(m) for m in members]
    assert len(out) == zip_size((m.name, len(c[1])) for m, c in zip(members, compressed))
    assert out == write_zip(members)
    with zipfile.ZipFile(io.BytesIO(out)) as zf:
        assert zf.namelist() == [m.name for m in members]
        for m in members:
            assert zf.read(m.name) == m.data
    for m, (method, packed, _) in zip(members, compressed):
        if m.name in ("baslt.json", "manifest.json") or m.method == 0:
            assert method == 0
        if method != 0:
            assert len(packed) < len(m.data)
    if 93 not in {c[0] for c in compressed}:
        assert list(read_members(out).values()) == [m.data for m in members]


@st.composite
def artifacts(draw):
    n_signals = draw(st.integers(1, 4))
    signals, data_members, expected = [], [], {}
    for i in range(n_signals):
        v = draw(value_arrays())
        n = v.shape[0]
        t = draw(hnp.arrays(np.dtype("<f8"), (n,)))
        idx = np.arange(n, dtype="<u4") * draw(st.integers(1, 1000))
        enc = draw(st.sampled_from(["raw", "shuffle"]))
        buf, arrays = bytearray(), []
        for aname, arr, aenc, member in (("t", t, "shuffle", f"t/{i}"), ("v", v, enc, f"s/{i}"),
                                         ("idx", idx, "delta+shuffle", f"s/{i}")):
            b = encode_array(arr, aenc)
            if member.startswith("t/"):
                data_members.append((member, b))
                offset = 0
            else:
                offset = len(buf)
                buf += b
            d = desc_of(arr, aenc, member, offset)
            d.name = aname
            arrays.append(d.to_json())
        data_members.append((f"s/{i}", bytes(buf)))
        signals.append({"name": f"sig{i}", "labels": [], "arrays": arrays})
        expected[f"sig{i}"] = {"t": t, "v": v, "idx": idx}
    order = draw(st.permutations(data_members))
    method = draw(st.sampled_from([0, 8]))
    members = [
        Member("baslt.json", header_json("deflate" if method else "store"), 0),
        Member("index.json", canonical_json({"container": 1, "signals": signals}), method),
        Member("manifest.json", canonical_json({"status": "pass", "x": float("nan")}), method),
        Member("policy.json", canonical_json({"canonical": {"version": 1}}), method),
        *(Member(name, data, method) for name, data in order),
    ]
    return members, expected


@given(artifacts())
def test_artifact_reader_and_mirror_agree(case):
    members, expected = case
    data = write_zip(members)
    art = read_artifact(data)
    mirror = load(data)
    assert art.size == len(data)
    assert mirror["index"] == art.index and mirror["manifest"] == art.manifest
    for name, arrays in expected.items():
        for aname, arr in arrays.items():
            ours = art.array(name, aname)
            theirs = mirror["signals"][name][aname]
            if aname == "idx":
                assert ours.dtype == theirs.dtype == np.float64
                np.testing.assert_array_equal(ours, arr)
                np.testing.assert_array_equal(theirs, arr)
            else:
                shape = (arr.shape[0],) if arr.ndim == 2 and arr.shape[1] == 1 else arr.shape
                assert ours.shape == theirs.shape == shape
                assert ours.tobytes() == theirs.tobytes() == np.ascontiguousarray(arr).tobytes()
