"""Array encodings: round trips, byte layouts and delta rules."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.container.encode import container_dtype, decode_array, encode_array
from baslt.container.spec import DTYPE_WIDTHS, ArrayDesc, idx_dtype, roles_dtype

DTYPES = list(DTYPE_WIDTHS)
SHAPES = [(0,), (1,), (17,), (0, 3), (4, 1), (9, 3)]


def sample(dtype: str, shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dt.kind == "f":
        a = (rng.standard_normal(shape) * 1e3).astype(dt)
        flat = a.reshape(-1)
        specials = np.array([np.nan, np.inf, -np.inf, -0.0, np.finfo(dt).tiny / 2, np.finfo(dt).max], dtype=dt)
        m = min(flat.size, specials.size)
        flat[:m] = specials[:m]
        return a
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, size=shape, dtype=dt, endpoint=True)
    flat = a.reshape(-1)
    m = min(flat.size, 2)
    flat[:m] = np.array([info.min, info.max], dtype=dt)[:m]
    return a


def desc_for(arr: np.ndarray, enc: str, offset: int = 0, dtype: str | None = None) -> ArrayDesc:
    code = dtype or container_dtype(arr.dtype)
    n = arr.shape[0]
    k = 1 if arr.ndim == 1 else arr.shape[1]
    return ArrayDesc("v", "s/0", offset, n * k * DTYPE_WIDTHS[code], n, k, code, enc)


def expected_shape(arr: np.ndarray) -> tuple[int, ...]:
    return (arr.shape[0],) if arr.ndim == 1 or arr.shape[1] == 1 else arr.shape


@pytest.mark.parametrize("enc", ["raw", "shuffle"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_round_trip_every_dtype(dtype, shape, enc):
    arr = sample(dtype, shape, seed=len(shape) * 31 + shape[0])
    buf = encode_array(arr, enc)
    assert len(buf) == arr.size * DTYPE_WIDTHS[dtype]
    out = decode_array(buf, desc_for(arr, enc))
    assert out.dtype == np.dtype(dtype)
    assert out.shape == expected_shape(arr)
    assert out.tobytes() == np.ascontiguousarray(arr.reshape(out.shape)).tobytes()


@pytest.mark.parametrize("enc", ["raw", "shuffle"])
def test_decode_at_offset_inside_member(enc):
    arr = sample("<f8", (11, 2), seed=3)
    payload = encode_array(arr, enc)
    member = b"\xaa" * 13 + payload + b"\xbb" * 7
    desc = desc_for(arr, enc, offset=13)
    for buf in (member, bytearray(member), memoryview(member)):
        out = decode_array(buf, desc)
        assert out.tobytes() == arr.tobytes()


@pytest.mark.parametrize("enc", ["raw", "shuffle"])
@pytest.mark.parametrize("shape", [(8,), (1,), (1, 3), (4, 2), (3, 1), (1, 1)])
def test_decoded_array_is_writable_and_independent(enc, shape):
    arr = np.arange(1, 1 + int(np.prod(shape)), dtype="<i4").reshape(shape)
    encoded = encode_array(arr, enc)
    for buf in (bytes(encoded), bytearray(encoded), memoryview(bytearray(encoded))):
        out = decode_array(buf, desc_for(arr, enc))
        assert out.flags.writeable and out.flags.c_contiguous
        if not isinstance(buf, bytes):
            buf[:] = b"\x00" * len(buf)
        np.testing.assert_array_equal(out, arr.reshape(out.shape))
        out[...] = 99
        np.testing.assert_array_equal(out, 99)


def test_raw_vector_layout_is_component_sequential():
    arr = np.array([[1, 10], [2, 20], [3, 30]], dtype="<i2")
    buf = encode_array(arr, "raw")
    assert buf == arr[:, 0].tobytes() + arr[:, 1].tobytes()
    # MATLAB reshape(a, n, k) and NumPy a.reshape(k, n).T
    flat = np.frombuffer(buf, dtype="<i2")
    np.testing.assert_array_equal(flat.reshape(2, 3).T, arr)
    np.testing.assert_array_equal(flat.reshape((3, 2), order="F"), arr)


def test_shuffle_byte_planes_formula():
    arr = np.array([[1.5, -2.25], [3.0e9, 7.0], [np.nan, -0.0]], dtype="<f4")
    raw = encode_array(arr, "raw")
    out = encode_array(arr, "shuffle")
    w = 4
    count = arr.size
    assert len(out) == len(raw) == count * w
    for p in range(w):
        for i in range(count):
            assert out[p * count + i] == raw[i * w + p]


def test_shuffle_of_single_byte_dtype_equals_raw():
    arr = np.arange(-50, 50, dtype="|i1")
    assert encode_array(arr, "shuffle") == encode_array(arr, "raw") == arr.tobytes()


def test_non_contiguous_and_fortran_inputs():
    base = sample("<f8", (20, 3), seed=9)
    for arr in (base[::2], np.asfortranarray(base), base[:, ::-1]):
        buf = encode_array(arr, "shuffle")
        out = decode_array(buf, desc_for(arr, "shuffle"))
        np.testing.assert_array_equal(out.view("<u8"), np.ascontiguousarray(arr).view("<u8"))


def test_bool_is_stored_as_u1():
    arr = np.array([True, False, True, True])
    assert container_dtype(arr.dtype) == "|u1"
    buf = encode_array(arr, "shuffle")
    assert buf == b"\x01\x00\x01\x01"
    np.testing.assert_array_equal(decode_array(buf, desc_for(arr, "shuffle")), [1, 0, 1, 1])


def test_big_endian_input_is_written_little_endian():
    arr = np.array([1, 256, 65536], dtype=">u4")
    assert container_dtype(arr.dtype) == "<u4"
    assert encode_array(arr, "raw") == arr.astype("<u4").tobytes()


def test_container_dtype_mapping():
    assert [container_dtype(np.dtype(c)) for c in DTYPES] == DTYPES
    assert container_dtype(np.int8) == "|i1"
    assert container_dtype(np.float32) == "<f4"
    for bad in (np.float16, np.complex128, object, "U4", "M8[s]"):
        with pytest.raises(ValueError):
            container_dtype(bad)


def test_encode_rejects_bad_shapes_encodings_and_dtypes():
    with pytest.raises(ValueError, match="shape"):
        encode_array(np.float64(1.0), "raw")
    with pytest.raises(ValueError, match="shape"):
        encode_array(np.zeros((2, 2, 2)), "shuffle")
    with pytest.raises(ValueError, match="encoding"):
        encode_array(np.zeros(3), "lz4")
    with pytest.raises(ValueError):
        encode_array(np.zeros(3, dtype=np.float16), "raw")
    with pytest.raises(ValueError):
        encode_array(np.zeros(3, dtype=np.complex128), "raw")


# --- delta+shuffle --------------------------------------------------------------------------


def test_delta_u4_bytes_and_round_trip():
    idx = np.array([0, 3, 4, 100, 70000, 2**32 - 1], dtype="<u4")
    buf = encode_array(idx, "delta+shuffle")
    deltas = np.array([0, 3, 1, 96, 69900, 2**32 - 1 - 70000], dtype="<u4")
    assert buf == encode_array(deltas, "shuffle")
    out = decode_array(buf, desc_for(idx, "delta+shuffle"))
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, idx.astype(np.float64))


def test_delta_f8_large_exact_integers():
    idx = np.array([2**32 - 2, 2**32, 2**40 + 1, 2**52, 2**53 - 1], dtype="<f8")
    buf = encode_array(idx, "delta+shuffle")
    out = decode_array(buf, desc_for(idx, "delta+shuffle"))
    assert out.dtype == np.float64
    assert out.tolist() == idx.tolist()
    assert [int(x) for x in out] == [2**32 - 2, 2**32, 2**40 + 1, 2**52, 2**53 - 1]


@pytest.mark.parametrize("dtype", ["<u4", "<f8"])
@pytest.mark.parametrize("values", [[], [0], [5], [0, 1, 2, 3]])
def test_delta_small_and_empty(dtype, values):
    idx = np.array(values, dtype=dtype)
    buf = encode_array(idx, "delta+shuffle")
    out = decode_array(buf, desc_for(idx, "delta+shuffle"))
    assert out.dtype == np.float64 and out.shape == (len(values),)
    assert out.tolist() == [float(v) for v in values]


def test_delta_f8_negative_zero_start_is_normalised():
    idx = np.array([-0.0, 1.0], dtype="<f8")
    out = decode_array(encode_array(idx, "delta+shuffle"), desc_for(idx, "delta+shuffle"))
    assert np.signbit(out).tolist() == [False, False]


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2, 3], dtype="<i8"),
        np.array([1, 2, 3], dtype="<i4"),
        np.array([1, 2, 3], dtype="<u8"),
        np.array([1, 2, 3], dtype="<u2"),
        np.array([1, 2, 3], dtype="<f4"),
        np.array([[1, 2], [3, 4]], dtype="<u4"),
    ],
    ids=["i8", "i4", "u8", "u2", "f4", "2d"],
)
def test_delta_rejects_non_idx_dtypes_and_shapes(arr):
    with pytest.raises(ValueError):
        encode_array(arr, "delta+shuffle")


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([3, 2], "<u4"),
        ([1, 1], "<u4"),
        ([0, 5, 5, 6], "<u4"),
        ([-1.0, 2.0], "<f8"),
        ([0.5, 2.0], "<f8"),
        ([0.0, np.nan], "<f8"),
        ([0.0, np.inf], "<f8"),
        ([0.0, 2.0**53], "<f8"),
        ([4.0, 3.0], "<f8"),
    ],
)
def test_delta_rejects_non_increasing_or_inexact_values(values, dtype):
    with pytest.raises(ValueError):
        encode_array(np.array(values, dtype=dtype), "delta+shuffle")


# --- decode validation ----------------------------------------------------------------------


def test_decode_rejects_bad_descriptors():
    arr = np.arange(6, dtype="<f8")
    buf = encode_array(arr, "shuffle")
    good = desc_for(arr, "shuffle")
    bad = [
        ArrayDesc("v", "s/0", 0, 47, 6, 1, "<f8", "shuffle"),  # nbytes mismatch
        ArrayDesc("v", "s/0", 8, 48, 6, 1, "<f8", "shuffle"),  # outside the member
        ArrayDesc("v", "s/0", 0, 48, 6, 1, "<f2", "shuffle"),  # unknown dtype
        ArrayDesc("v", "s/0", 0, 48, 6, 1, "<f8", "gzip"),  # unknown encoding
        ArrayDesc("v", "s/0", 0, 48, 3, 2, "<f8", "delta+shuffle"),  # delta with 2 components
        ArrayDesc("v", "s/0", 0, 24, 6, 1, "<i4", "delta+shuffle"),  # delta with a non-idx dtype
        ArrayDesc("v", "s/0", 0, 0, 0, 0, "<f8", "raw"),  # zero components
    ]
    decode_array(buf, good)
    for desc in bad:
        with pytest.raises(ValueError):
            decode_array(buf, desc)


def test_idx_and_roles_dtype_rules():
    assert idx_dtype(0) == "<u4"
    assert idx_dtype(2**32 - 1) == "<u4"
    assert idx_dtype(2**32) == "<f8"
    assert [roles_dtype(b) for b in (0, 1, 8, 9, 16, 17, 32, 33, 64)] == [
        "|u1", "|u1", "|u1", "<u2", "<u2", "<u4", "<u4", "<u8", "<u8"
    ]
    with pytest.raises(ValueError):
        roles_dtype(65)


def test_large_arrays_round_trip():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((200_000, 3))
    out = decode_array(encode_array(v, "shuffle"), desc_for(v, "shuffle"))
    assert out.tobytes() == v.tobytes()
    idx = np.cumsum(rng.integers(1, 50, 300_000)).astype("<u4")
    out = decode_array(encode_array(idx, "delta+shuffle"), desc_for(idx, "delta+shuffle"))
    np.testing.assert_array_equal(out, idx)
