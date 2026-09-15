"""Byte-level format contract that MATLAB load.m and the JavaScript reader rely on."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from reference.matlab_mirror import (
    MirrorError,
    check_uniform_keys,
    combine_pairs,
    decode_member,
    load,
    parse_json,
    read_members,
    uint32_pairs,
)

from baslt.container.encode import container_dtype, encode_array
from baslt.container.reader import read_artifact
from baslt.container.spec import DTYPE_WIDTHS, ArrayDesc, canonical_json, header_json
from baslt.container.zipwriter import Member, write_zip
from baslt.errors import ContainerError

DOC_HEADER = b'{"byte_order":"little","codec":"deflate","container":1,"format":"baslt","level":6,"reader_min":1}'

DOC_INDEX = {
    "container": 1,
    "signals": [
        {
            "name": "q_dyn", "path": "aero/q", "kind": "continuous", "interp": "linear", "unit": "Pa",
            "labels": [], "n_source": 120001, "n": 5231, "components": 1,
            "arrays": [
                {"name": "t", "member": "t/0", "offset": 0, "nbytes": 41848, "n": 5231, "components": 1, "dtype": "<f8", "enc": "shuffle"},
                {"name": "v", "member": "s/0", "offset": 0, "nbytes": 41848, "n": 5231, "components": 1, "dtype": "<f8", "enc": "shuffle"},
                {"name": "idx", "member": "s/0", "offset": 41848, "nbytes": 20924, "n": 5231, "components": 1, "dtype": "<u4", "enc": "delta+shuffle"},
                {"name": "roles", "member": "s/0", "offset": 62772, "nbytes": 10462, "n": 5231, "components": 1, "dtype": "<u2", "enc": "shuffle"},
            ],
            "roles": [{"bit": 0, "id": "extent"}, {"bit": 1, "id": "hard.q_dyn.global_extrema"}, {"bit": 2, "id": "soft"}],
            "requirements": [
                {"id": "hard.q_dyn.threshold_crossing[0]", "op": "threshold_crossing", "bits": [3],
                 "params": [{"name": "value", "value": 65000.0}, {"name": "edge", "value": "both"}]}
            ],
        }
    ],
}


def values_for(dtype: str, shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dt.kind == "f":
        a = (rng.standard_normal(shape) * 1e6).astype(dt)
        if a.size:
            a.reshape(-1)[0] = np.nan
        return a
    info = np.iinfo(dt)
    return rng.integers(info.min, info.max, size=shape, dtype=dt, endpoint=True)


def coverage_artifact(codec: str = "deflate") -> tuple[bytes, dict]:
    """Every dtype and encoding, vectors, empty arrays, both idx dtypes and a shared clock."""
    method = {"deflate": 8, "store": 0}[codec]
    t = np.cumsum(np.full(40, 0.01))
    t_bytes = encode_array(t, "shuffle")
    t_desc = ArrayDesc("t", "t/0", 0, len(t_bytes), 40, 1, "<f8", "shuffle").to_json()
    expected: dict[str, dict[str, np.ndarray]] = {}
    signals, data_members = [], []
    k = 0
    for dtype in DTYPE_WIDTHS:
        for enc in ("raw", "shuffle"):
            for shape in ((40,), (40, 3)):
                name = f"sig_{dtype[1:]}_{enc}_{len(shape)}"
                v = values_for(dtype, shape, seed=k)
                idx = np.arange(40, dtype="<u4") * 7 + 2 if k % 2 else np.arange(40, dtype="<f8") * 2.0**40 + 2**33
                arrays = [dict(t_desc)]
                buf = bytearray()
                for aname, arr, aenc in (("v", v, enc), ("idx", idx, "delta+shuffle"),
                                         ("roles", np.arange(40, dtype="<u8") << np.uint64(60), "shuffle")):
                    b = encode_array(arr, aenc)
                    comps = 1 if arr.ndim == 1 else arr.shape[1]
                    arrays.append(ArrayDesc(aname, f"s/{k}", len(buf), len(b), arr.shape[0], comps,
                                            container_dtype(arr.dtype), aenc).to_json())
                    buf += b
                data_members.append(Member(f"s/{k}", bytes(buf), method))
                signals.append({"name": name, "labels": [], "components": arrays[1]["components"], "arrays": arrays})
                expected[name] = {"t": t, "v": v, "idx": idx.astype(np.float64)}
                k += 1
    # an empty vector signal with its own empty clock
    empty_t = np.empty(0)
    signals.append({
        "name": "empty", "labels": [], "components": 2,
        "arrays": [
            ArrayDesc("t", "t/1", 0, 0, 0, 1, "<f8", "shuffle").to_json(),
            ArrayDesc("v", f"s/{k}", 0, 0, 0, 2, "<i2", "shuffle").to_json(),
            ArrayDesc("idx", f"s/{k}", 0, 0, 0, 1, "<u4", "delta+shuffle").to_json(),
            ArrayDesc("roles", f"s/{k}", 0, 0, 0, 1, "|u1", "raw").to_json(),
        ],
    })
    data_members.append(Member(f"s/{k}", b"", method))
    expected["empty"] = {"t": empty_t, "v": np.empty((0, 2), "<i2"), "idx": np.empty(0)}
    index = {"container": 1, "signals": signals}
    manifest = {"status": "pass", "values": [float("nan"), float("inf"), -float("inf"), 1.5],
                "requirements": [{"id": "a", "evidence": {"max": float("nan")}}, {"id": "b", "evidence": {}}]}
    policy = {"canonical": {"version": 1, "name": "µs check"}, "sha256": "ab" * 32,
              "source_format": "yaml", "source_text": "version: 1\nname: µs check\n"}
    members = [
        Member("baslt.json", header_json(codec), 0),
        Member("index.json", canonical_json(index), method),
        Member("manifest.json", canonical_json(manifest), method),
        Member("policy.json", canonical_json(policy), method),
        *data_members,
        Member("t/0", t_bytes, method),
        Member("t/1", b"", method),
    ]
    return write_zip(members), expected


@pytest.mark.parametrize("codec", ["deflate", "store"])
def test_mirror_decodes_identically_to_reader(codec):
    data, expected = coverage_artifact(codec)
    art = read_artifact(data)
    mirror = load(data)
    assert mirror["header"] == art.header
    assert mirror["index"] == art.index
    assert mirror["manifest"] == art.manifest
    assert mirror["policy"] == art.policy
    assert list(mirror["signals"]) == art.signal_names()
    for sig in art.index["signals"]:
        name = sig["name"]
        for desc in sig["arrays"]:
            ours = art.array(name, desc["name"])
            theirs = mirror["signals"][name][desc["name"]]
            assert ours.shape == theirs.shape
            if desc["enc"] == "delta+shuffle":
                assert ours.dtype == theirs.dtype == np.float64
                np.testing.assert_array_equal(ours, theirs)
            else:
                assert ours.dtype == theirs.dtype
                assert ours.tobytes() == theirs.tobytes()
        for aname, arr in expected[name].items():
            assert art.array(name, aname).tobytes() == np.ascontiguousarray(arr).tobytes()


def test_header_member_at_byte_30():
    data, _ = coverage_artifact()
    assert data[:4] == b"PK\x03\x04"
    assert data[30:40] == b"baslt.json"
    method, = struct.unpack_from("<H", data, 8)
    extra_len, = struct.unpack_from("<H", data, 28)
    size, = struct.unpack_from("<I", data, 18)
    assert method == 0 and extra_len == 0
    assert data[40 : 40 + size] == DOC_HEADER == header_json()


def test_eocd_is_last_22_bytes_and_local_extras_are_zero():
    data, _ = coverage_artifact()
    eocd = len(data) - 22
    assert data[eocd : eocd + 4] == b"PK\x05\x06"
    _, _, _, n_disk, n_total, cd_size, cd_offset, comment_len = struct.unpack_from("<4sHHHHIIH", data, eocd)
    assert comment_len == 0 and n_disk == n_total and cd_offset + cd_size == eocd
    pos = cd_offset
    for _ in range(n_total):
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", data, pos + 28)
        local, = struct.unpack_from("<I", data, pos + 42)
        assert extra_len == 0 and comment_len == 0
        assert struct.unpack_from("<HH", data, local + 26) == (name_len, 0)
        pos += 46 + name_len
    assert pos == eocd


def test_mirror_detects_corruption():
    data, _ = coverage_artifact()
    members = read_members(data)
    assert list(members)[:4] == ["baslt.json", "index.json", "manifest.json", "policy.json"]
    bad = bytearray(data)
    bad[45] ^= 0xFF  # inside the STORED header JSON
    with pytest.raises(MirrorError, match="CRC"):
        read_members(bytes(bad))
    with pytest.raises(MirrorError):
        read_members(data[:-1])


def test_json_members_have_no_non_finite_tokens():
    data, _ = coverage_artifact()
    members = read_members(data)
    for name in ("baslt.json", "index.json", "manifest.json", "policy.json"):
        obj = parse_json(members[name])
        assert members[name] == canonical_json(obj)
    manifest = parse_json(members["manifest.json"])
    assert manifest["values"] == ["NaN", "Infinity", "-Infinity", 1.5]
    policy_raw = members["policy.json"]
    assert "µ".encode() in policy_raw and b"\\u00b5" not in policy_raw


def test_canonical_json_rules():
    obj = {"b": [1.0, float("nan"), (float("inf"), -float("inf"))], "a": {"z": None, "y": True, "x": "°"},
           "c": np.float64(np.nan), "d": np.float32(2.5), "e": np.int64(7), "f": np.arange(3)}
    raw = canonical_json(obj)
    assert raw == (
        '{"a":{"x":"°","y":true,"z":null},"b":[1.0,"NaN",["Infinity","-Infinity"]],'
        '"c":"NaN","d":2.5,"e":7,"f":[0,1,2]}'
    ).encode("utf-8")
    parse_json(raw)
    with pytest.raises(MirrorError):
        parse_json(b'{"a":NaN}')
    with pytest.raises(MirrorError):
        parse_json(b'{"a":-Infinity}')
    assert json.loads(canonical_json({"x": 0.1 + 0.2}))["x"] == 0.1 + 0.2


def test_canonical_json_rejects_strings_that_are_not_unicode():
    # a POSIX file name with undecodable bytes arrives as a lone surrogate escape
    with pytest.raises(ContainerError, match=r"\$\.source\.path") as info:
        canonical_json({"ok": "µs", "source": {"path": "run\udcff.h5"}})
    assert info.value.exit_code == 3
    assert "\n" not in str(info.value)
    str(info.value).encode("utf-8")  # the message itself is printable
    with pytest.raises(ContainerError, match=r"\$\.issues\[1\]"):
        canonical_json({"issues": ["fine", ("bad\ud800",)]})
    with pytest.raises(ContainerError, match="key"):
        canonical_json({"k\udcff": 1})
    assert canonical_json({"path": "run\U0001f680.h5"}) == '{"path":"run\U0001f680.h5"}'.encode()


def test_uniform_keys_of_doc_example_and_artifact():
    check_uniform_keys(DOC_INDEX)
    data, _ = coverage_artifact()
    art = read_artifact(data)
    check_uniform_keys(art.index)
    check_uniform_keys(art.manifest)


@pytest.mark.parametrize(
    "obj",
    [
        [{"a": 1}, {"b": 1}],
        {"x": [{"a": 1}, {"a": 1, "b": 2}]},
        [{"a": 1}, 3],
        {"s": [{"arrays": [{"n": 1}, {"m": 1}]}]},
        [{"a": [{"k": 1}]}, {"a": [{"k": 1}, {"j": 2}]}],
    ],
)
def test_check_uniform_keys_rejects(obj):
    with pytest.raises(AssertionError):
        check_uniform_keys(obj)


def test_check_uniform_keys_accepts_scalars_and_empty_lists():
    check_uniform_keys({"a": [], "b": [1, 2.0, "x", None], "c": [[1, 2], [3]], "d": [{"x": {"p": 1}}, {"x": {"q": 2}}]})


@pytest.mark.parametrize("dtype", ["<i8", "<u8"])
def test_eight_byte_integers_as_uint32_pairs(dtype):
    info = np.iinfo(dtype)
    v = np.array([info.min, info.max, 0, 1, 2**53 + 1, 2**63 - 2 if dtype == "<u8" else -(2**53) - 3], dtype=dtype)
    for enc in ("raw", "shuffle"):
        member = b"\x07" * 3 + encode_array(v, enc)
        desc = ArrayDesc("v", "s/0", 3, 48, 6, 1, dtype, enc).to_json()
        low, high = uint32_pairs(member, desc)
        assert combine_pairs(low, high, signed=dtype == "<i8") == [int(x) for x in v.tolist()]
        assert decode_member(member, desc).tolist() == v.tolist()


def test_column_major_reshape_matches_numpy_layout():
    v = np.arange(24, dtype="<i4").reshape(8, 3)
    member = encode_array(v, "shuffle")
    desc = ArrayDesc("v", "s/0", 0, len(member), 8, 3, "<i4", "shuffle").to_json()
    np.testing.assert_array_equal(decode_member(member, desc), v)


def test_float64_cumsum_is_exact_below_2_53():
    idx = np.array([0.0, 1.0, 2.0**32, 2.0**52, 2.0**53 - 1])
    member = encode_array(idx, "delta+shuffle")
    desc = ArrayDesc("idx", "s/0", 0, 40, 5, 1, "<f8", "delta+shuffle").to_json()
    assert [int(x) for x in decode_member(member, desc)] == [0, 1, 2**32, 2**52, 2**53 - 1]
