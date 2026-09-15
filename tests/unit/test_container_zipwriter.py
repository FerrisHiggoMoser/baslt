"""ZIP subset writer: byte-exact headers, size formula, STORED fallback and determinism."""

from __future__ import annotations

import io
import struct
import zipfile
import zlib

import numpy as np
import pytest

from baslt.container.spec import header_json, zstd_available
from baslt.container.zipwriter import Member, compress_member, write_zip, zip_size
from baslt.errors import ContainerError


def parse_zip(data: bytes) -> tuple[list[dict], dict]:
    """Every local and central header field, parsed with struct."""
    eocd = dict(
        zip(
            ("sig", "disk", "cd_disk", "n_disk", "n_total", "cd_size", "cd_offset", "comment_len"),
            struct.unpack_from("<4sHHHHIIH", data, len(data) - 22),
        )
    )
    entries = []
    pos = eocd["cd_offset"]
    central_keys = (
        "sig", "made_by", "needed", "flags", "method", "time", "date", "crc", "csize", "usize",
        "name_len", "extra_len", "comment_len", "disk_start", "internal", "external", "offset",
    )
    local_keys = ("sig", "needed", "flags", "method", "time", "date", "crc", "csize", "usize", "name_len", "extra_len")
    for _ in range(eocd["n_total"]):
        c = dict(zip(central_keys, struct.unpack_from("<4sHHHHHHIIIHHHHHII", data, pos)))
        c["name"] = data[pos + 46 : pos + 46 + c["name_len"]]
        pos += 46 + c["name_len"] + c["extra_len"] + c["comment_len"]
        lo = dict(zip(local_keys, struct.unpack_from("<4sHHHHHIIIHH", data, c["offset"])))
        lo["name"] = data[c["offset"] + 30 : c["offset"] + 30 + lo["name_len"]]
        start = c["offset"] + 30 + lo["name_len"] + lo["extra_len"]
        entries.append({"central": c, "local": lo, "payload": data[start : start + c["csize"]]})
    eocd["cd_end"] = pos
    return entries, eocd


def random_bytes(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def sample_members() -> list[Member]:
    return [
        Member("baslt.json", header_json(), 8),
        Member("index.json", b'{"signals":[' + b'{"name":"x"},' * 200 + b'{"name":"y"}]}', 8),
        Member("manifest.json", b'{"status":"pass"}' + b" " * 500, 8),
        Member("policy.json", b"{}", 8),
        Member("s/0", random_bytes(4096), 8),
        Member("t/0", np.linspace(0, 1, 2000).tobytes(), 8),
    ]


def test_header_fields_byte_exact():
    members = sample_members()
    data = write_zip(members)
    entries, eocd = parse_zip(data)
    assert eocd == {
        "sig": b"PK\x05\x06", "disk": 0, "cd_disk": 0, "n_disk": 6, "n_total": 6,
        "cd_size": eocd["cd_size"], "cd_offset": eocd["cd_offset"], "comment_len": 0,
        "cd_end": len(data) - 22,
    }
    assert eocd["cd_offset"] + eocd["cd_size"] == len(data) - 22
    expected_offset = 0
    expected_methods = [0, 8, 0, 0, 0, 8]  # baslt.json/manifest.json forced; policy.json and s/0 not smaller
    for m, e, method in zip(members, entries, expected_methods):
        c, lo = e["central"], e["local"]
        name = m.name.encode()
        crc = zlib.crc32(m.data) & 0xFFFFFFFF
        assert lo == {
            "sig": b"PK\x03\x04", "needed": 20, "flags": 0, "method": method, "time": 0, "date": 0x0021,
            "crc": crc, "csize": len(e["payload"]), "usize": len(m.data), "name_len": len(name),
            "extra_len": 0, "name": name,
        }
        assert c == {
            "sig": b"PK\x01\x02", "made_by": 20, "needed": 20, "flags": 0, "method": method, "time": 0,
            "date": 0x0021, "crc": crc, "csize": len(e["payload"]), "usize": len(m.data),
            "name_len": len(name), "extra_len": 0, "comment_len": 0, "disk_start": 0, "internal": 0,
            "external": 0, "offset": expected_offset, "name": name,
        }
        expected_offset += 30 + len(name) + c["csize"]
        if method == 0:
            assert e["payload"] == m.data
        else:
            assert zlib.decompress(e["payload"], -15) == m.data
    assert expected_offset == eocd["cd_offset"]


def test_baslt_json_is_first_and_named_at_byte_30():
    data = write_zip(sample_members())
    assert data[:4] == b"PK\x03\x04"
    assert data[30:40] == b"baslt.json"
    assert struct.unpack_from("<H", data, 8)[0] == 0
    assert data[40 : 40 + len(header_json())] == header_json()


@pytest.mark.parametrize("name", ["baslt.json", "manifest.json"])
def test_header_and_manifest_always_stored(name):
    data = b'{"a":1}' * 1000
    method, packed, crc = compress_member(Member(name, data, 8))
    assert (method, packed, crc) == (0, data, zlib.crc32(data))
    if zstd_available():
        assert compress_member(Member(name, data, 93))[0] == 0


def test_stored_fallback_when_deflate_not_smaller():
    noise = random_bytes(1000, seed=5)
    assert compress_member(Member("s/0", noise, 8)) == (0, noise, zlib.crc32(noise))
    assert compress_member(Member("s/1", b"", 8)) == (0, b"", 0)
    assert compress_member(Member("s/2", b"a", 8))[0] == 0
    method, packed, crc = compress_member(Member("s/3", b"\x00" * 1000, 8))
    assert method == 8 and len(packed) < 1000 and crc == zlib.crc32(b"\x00" * 1000)


def test_deflate_is_raw_level_6_by_default():
    data = np.sin(np.linspace(0, 50, 5000)).tobytes()
    method, packed, _ = compress_member(Member("s/0", data, 8))
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    assert method == 8 and packed == c.compress(data) + c.flush()
    assert zlib.decompress(packed, -15) == data
    assert not packed.startswith(b"\x78")  # no zlib header
    fast = compress_member(Member("s/0", data, 8), level=1)[1]
    c1 = zlib.compressobj(1, zlib.DEFLATED, -15)
    assert fast == c1.compress(data) + c1.flush()


def test_store_method_is_kept():
    data = b"\x00" * 1000
    assert compress_member(Member("s/0", data, 0)) == (0, data, zlib.crc32(data))


def test_size_formula_matches_output_for_random_member_sets():
    rng = np.random.default_rng(42)
    for trial in range(40):
        members = []
        for i in range(int(rng.integers(1, 12))):
            kind = int(rng.integers(0, 3))
            size = int(rng.integers(0, 3000))
            if kind == 0:
                data = random_bytes(size, seed=trial * 100 + i)
            elif kind == 1:
                data = bytes(size)
            else:
                data = np.cumsum(rng.integers(0, 3, size)).astype("<u2").tobytes()
            name = f"s/{i}" if i % 3 else f"t/{trial}.{i}"
            members.append(Member(name, data, int(rng.choice([0, 8]))))
        rng.shuffle(members)
        csizes = [(m.name, len(compress_member(m)[1])) for m in members]
        out = write_zip(members)
        assert len(out) == zip_size(csizes)
        assert zip_size(csizes) == 22 + sum(76 + 2 * len(n) + c for n, c in csizes)


def test_zipfile_reads_everything():
    members = sample_members()
    data = write_zip(members)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.testzip() is None
        assert zf.namelist() == [m.name for m in members]
        for m, info in zip(members, zf.infolist()):
            assert zf.read(info) == m.data
            assert info.flag_bits == 0 and info.extra == b"" and info.comment == b""
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert zf.comment == b""


def test_deterministic_output():
    first = write_zip(sample_members())
    second = write_zip(sample_members())
    assert first == second
    third = write_zip([Member(m.name, bytearray(m.data), m.method) for m in sample_members()])
    assert third == first


def test_accepts_bytes_like_data():
    data = b"\x01\x02" * 400
    out = write_zip([Member("s/0", memoryview(data), 8), Member("s/1", bytearray(data), 0)])
    with zipfile.ZipFile(io.BytesIO(out)) as zf:
        assert zf.read("s/0") == data and zf.read("s/1") == data


@pytest.mark.parametrize("method", [0, 8])
def test_multibyte_buffers_use_their_byte_length(method):
    values = np.linspace(0.0, 1.0, 10, dtype="<f8")
    sources = {
        "ndarray": values,
        "ndarray_2d": np.arange(24, dtype="<u4").reshape(6, 4),
        "ndarray_strided": np.arange(40, dtype="<f8")[::4],
        "memoryview_d": memoryview(values.tobytes()).cast("d"),
        "memoryview_I": memoryview(np.arange(50, dtype="<u4").tobytes()).cast("I"),
    }
    for label, src in sources.items():
        expected = bytes(src)
        assert len(expected) != len(src), label
        out = write_zip([Member("s/0", src, method)])
        (e,), _ = parse_zip(out)
        assert e["local"]["usize"] == e["central"]["usize"] == len(expected), label
        assert e["local"]["crc"] == zlib.crc32(expected), label
        with zipfile.ZipFile(io.BytesIO(out)) as zf:
            assert zf.read("s/0") == expected, label
        assert out == write_zip([Member("s/0", expected, method)]), label
        assert compress_member(Member("s/0", src, method)) == compress_member(Member("s/0", expected, method))


@pytest.mark.parametrize("name", ["A", "dir/", "/x", "a//b", "../x", "s/./0", "café", "", "a b", "x\\y"])
def test_rejects_invalid_member_names(name):
    with pytest.raises(ContainerError):
        write_zip([Member(name, b"x", 0)])


def test_rejects_empty_duplicate_and_bad_method():
    with pytest.raises(ContainerError):
        write_zip([])
    with pytest.raises(ContainerError, match="duplicate"):
        write_zip([Member("s/0", b"x", 0), Member("s/0", b"y", 0)])
    with pytest.raises(ContainerError, match="method"):
        write_zip([Member("s/0", b"x", 12)])
    with pytest.raises(ContainerError, match="level"):
        compress_member(Member("s/0", b"x" * 100, 8), level=10)


def test_member_is_slotted_dataclass():
    m = Member("s/0", b"", 8)
    with pytest.raises(AttributeError):
        m.other = 1  # type: ignore[attr-defined]


@pytest.mark.skipif(not zstd_available(), reason="needs compression.zstd (Python 3.14+)")
def test_zstd_members():
    from compression import zstd

    data = np.linspace(0, 1, 4000).tobytes()
    method, packed, crc = compress_member(Member("s/0", data, 93))
    assert method == 93 and zstd.decompress(packed) == data and crc == zlib.crc32(data)
    noise = random_bytes(500)
    assert compress_member(Member("s/1", noise, 93))[0] == 0
    out = write_zip([Member("baslt.json", header_json("zstd"), 93), Member("s/0", data, 93)])
    entries, _ = parse_zip(out)
    assert entries[0]["local"]["needed"] == 20 and entries[0]["local"]["method"] == 0
    assert entries[1]["local"]["needed"] == 63 and entries[1]["central"]["needed"] == 63
    assert entries[1]["central"]["made_by"] == 20 and entries[1]["central"]["method"] == 93
    with zipfile.ZipFile(io.BytesIO(out)) as zf:
        assert zf.read("s/0") == data
    assert len(out) == zip_size([("baslt.json", len(header_json("zstd"))), ("s/0", len(packed))])
    assert out == write_zip([Member("baslt.json", header_json("zstd"), 93), Member("s/0", data, 93)])


@pytest.mark.skipif(zstd_available(), reason="only without compression.zstd")
def test_zstd_unavailable_is_a_container_error():
    with pytest.raises(ContainerError, match="--codec deflate"):
        compress_member(Member("s/0", b"\x00" * 1000, 93))
