"""Artifact reader: valid files decode exactly; every profile violation is a ContainerError."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from reference.matlab_mirror import MirrorError, load

from baslt.container import reader as reader_module
from baslt.container import spec as spec_module
from baslt.container.encode import container_dtype, encode_array
from baslt.container.reader import Artifact, read_artifact
from baslt.container.spec import CODEC_METHODS, ArrayDesc, canonical_json, header_dict, header_json, zstd_available
from baslt.container.zipwriter import Member, write_zip
from baslt.errors import ContainerError

# --- a small artifact builder ----------------------------------------------------------------


def sample_data() -> dict:
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 9.9, 100)
    q = np.sin(t) * 6.5e4
    q[7] = np.nan
    idx = np.sort(rng.choice(100_000, 100, replace=False)).astype("<u4")
    return {
        "t/0": t,
        "q_dyn": {
            "v": (q, "shuffle"),
            "idx": (idx, "delta+shuffle"),
            "roles": (rng.integers(0, 2**16, 100).astype("<u2"), "shuffle"),
        },
        "position": {
            "v": (rng.standard_normal((100, 3)), "shuffle"),
            "idx": (np.arange(100, dtype="<u4") * 3, "delta+shuffle"),
            "roles": (rng.integers(0, 4, 100).astype("|u1"), "raw"),
        },
        "mode": {
            "v": (rng.integers(0, 3, 100).astype("<i4"), "shuffle"),
            "idx": (np.arange(100, dtype="<u4"), "delta+shuffle"),
            "roles": (np.ones(100, dtype="|u1"), "shuffle"),
        },
    }


def build_members(codec: str = "deflate", *, index_edit=None, header=None) -> list[Member]:
    method = CODEC_METHODS[codec]
    d = sample_data()
    t = d["t/0"]
    t_bytes = encode_array(t, "shuffle")
    t_desc = ArrayDesc("t", "t/0", 0, len(t_bytes), t.shape[0], 1, "<f8", "shuffle").to_json()
    signals = []
    data_members = []
    for k, name in enumerate(("q_dyn", "position", "mode")):
        member = f"s/{k}"
        buf = bytearray()
        arrays = [dict(t_desc)]
        for aname, (arr, enc) in d[name].items():
            b = encode_array(arr, enc)
            comps = 1 if arr.ndim == 1 else arr.shape[1]
            arrays.append(
                ArrayDesc(aname, member, len(buf), len(b), arr.shape[0], comps, container_dtype(arr.dtype), enc).to_json()
            )
            buf += b
        data_members.append(Member(member, bytes(buf), method))
        signals.append({
            "name": name, "path": f"sim/{name}", "kind": "continuous", "interp": "linear", "unit": None,
            "labels": [], "n_source": 100_000, "n": 100, "components": arrays[1]["components"],
            "arrays": arrays, "roles": [{"bit": 0, "id": "extent"}], "requirements": [],
        })
    index = {"container": 1, "signals": signals}
    if index_edit is not None:
        index_edit(index)
    manifest = {"status": "pass", "budget": {"soft_max_abs_err": float("nan")}, "requirements": []}
    policy = {"canonical": {"version": 1}, "sha256": "0" * 64, "source_format": "dict", "source_text": None}
    head = header_json(codec) if header is None else canonical_json(header)
    return [
        Member("baslt.json", head, 0),
        Member("index.json", canonical_json(index), method),
        Member("manifest.json", canonical_json(manifest), method),
        Member("policy.json", canonical_json(policy), method),
        *data_members,
        Member("t/0", t_bytes, method),
    ]


def build(codec: str = "deflate", **kwargs) -> bytes:
    return write_zip(build_members(codec, **kwargs))


# --- raw zip surgery -------------------------------------------------------------------------


def explode(data: bytes) -> list[dict]:
    n_total, cd_size, cd_offset = struct.unpack_from("<HII", data, len(data) - 22 + 10)
    pos = cd_offset
    out = []
    for _ in range(n_total):
        f = struct.unpack_from("<4sHHHHHHIIIHHHHHII", data, pos)
        name_len, extra_len, comment_len, local = f[10], f[11], f[12], f[16]
        name = data[pos + 46 : pos + 46 + name_len]
        start = local + 30 + name_len
        out.append({
            "name": name, "needed": f[2], "flags": f[3], "method": f[4], "crc": f[7], "usize": f[9],
            "payload": data[start : start + f[8]], "local_extra": b"", "central_extra": b"",
        })
        pos += 46 + name_len + extra_len + comment_len
    return out


def raw_zip(entries: list[dict], comment: bytes = b"") -> bytes:
    chunks, central, offset = [], [], 0
    for e in entries:
        local = struct.pack(
            "<4sHHHHHIIIHH", b"PK\x03\x04", e.get("local_needed", e["needed"]), e.get("local_flags", e["flags"]),
            e.get("local_method", e["method"]), 0, 0x21, e["crc"], len(e["payload"]), e["usize"],
            len(e["name"]), len(e["local_extra"]),
        )
        chunks += [local, e["name"], e["local_extra"], e["payload"]]
        central.append(
            struct.pack(
                "<4sHHHHHHIIIHHHHHII", b"PK\x01\x02", 20, e["needed"], e["flags"], e["method"], 0, 0x21,
                e["crc"], len(e["payload"]), e["usize"], len(e["name"]), len(e["central_extra"]), 0, 0, 0, 0,
                offset,
            )
            + e["name"]
            + e["central_extra"]
        )
        offset += 30 + len(e["name"]) + len(e["local_extra"]) + len(e["payload"])
    cd = b"".join(central)
    eocd = struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, len(entries), len(entries), len(cd), offset, len(comment))
    return b"".join(chunks) + cd + eocd + comment


def entry(entries: list[dict], name: str) -> dict:
    return next(e for e in entries if e["name"] == name.encode())


def data_offset(data: bytes, name: str) -> int:
    for e_off in range(len(data) - 4):
        if data[e_off : e_off + 4] == b"PK\x03\x04":
            name_len = struct.unpack_from("<H", data, e_off + 26)[0]
            if data[e_off + 30 : e_off + 30 + name_len] == name.encode():
                return e_off + 30 + name_len
    raise AssertionError(name)


# --- valid artifacts -------------------------------------------------------------------------


def test_raw_zip_helpers_reproduce_writer_output():
    data = build()
    assert raw_zip(explode(data)) == data


@pytest.mark.parametrize("codec", ["deflate", "store", pytest.param("zstd", marks=pytest.mark.skipif(
    not zstd_available(), reason="needs compression.zstd"))])
def test_read_valid_artifact(codec, tmp_path):
    data = build(codec)
    path = tmp_path / "run.baslt"
    path.write_bytes(data)
    d = sample_data()
    for source in (data, bytearray(data), path, str(path)):
        art = read_artifact(source)
        assert isinstance(art, Artifact)
        assert art.header == header_dict(codec)
        assert art.size == len(data)
        assert art.signal_names() == ["q_dyn", "position", "mode"]
        assert art.manifest["budget"]["soft_max_abs_err"] == "NaN"
        assert art.policy["sha256"] == "0" * 64
        assert [e.name for e in art.entries][:4] == ["baslt.json", "index.json", "manifest.json", "policy.json"]
        for name in ("q_dyn", "position", "mode"):
            t = art.array(name, "t")
            assert t.tobytes() == d["t/0"].tobytes()
            for aname, (arr, enc) in d[name].items():
                out = art.array(name, aname)
                if enc == "delta+shuffle":
                    assert out.dtype == np.float64
                    np.testing.assert_array_equal(out, arr)
                else:
                    assert out.dtype == arr.dtype and out.shape == arr.shape
                    assert out.tobytes() == arr.tobytes()
        methods = {e.name: e.method for e in art.entries}
        if codec == "store":
            assert set(methods.values()) == {0}
        else:
            assert methods["baslt.json"] == 0 and methods["manifest.json"] == 0
            assert methods["t/0"] == CODEC_METHODS[codec]


def test_shared_clock_descriptors():
    art = read_artifact(build())
    descs = [art.descriptor(name, "t") for name in art.signal_names()]
    assert {(d.member, d.offset, d.nbytes) for d in descs} == {("t/0", 0, 800)}


def test_unknown_signal_or_array():
    art = read_artifact(build())
    with pytest.raises(ContainerError, match="no signal"):
        art.array("nope", "v")
    with pytest.raises(ContainerError, match="no array"):
        art.array("q_dyn", "w")


# --- tampering and profile violations --------------------------------------------------------


def flip(data: bytes, pos: int) -> bytes:
    return data[:pos] + bytes([data[pos] ^ 0xFF]) + data[pos + 1 :]


@pytest.mark.parametrize("member", ["s/0", "t/0", "manifest.json", "baslt.json", "index.json"])
def test_flipped_data_byte(member):
    data = build()
    start = data_offset(data, member)
    for delta in (0, 5):
        with pytest.raises(ContainerError):
            read_artifact(flip(data, start + delta))


needs_zstd = pytest.mark.skipif(not zstd_available(), reason="needs compression.zstd")


@needs_zstd
@pytest.mark.parametrize("member", ["t/0", "index.json"])
def test_flipped_zstd_data_byte(member):
    data = build("zstd")
    e = entry(explode(data), member)
    assert e["method"] == 93
    start = data_offset(data, member)
    for delta in (0, 1, 4, 5, len(e["payload"]) // 2, len(e["payload"]) - 1):
        with pytest.raises(ContainerError, match="not a valid .baslt file"):
            read_artifact(flip(data, start + delta))


@pytest.mark.parametrize("codec", ["deflate", pytest.param("zstd", marks=needs_zstd)])
@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p + b"JUNKJUNK", "does not end at its compressed size"),
        (lambda p: p + p, "does not end at its compressed size"),
        (lambda p: p + b"\x00", "does not end at its compressed size"),
        (lambda p: p[:-4], "truncated|cannot decompress"),
    ],
    ids=["junk", "second_stream", "zero_byte", "truncated"],
)
def test_compressed_stream_must_end_at_compressed_size(codec, change, match):
    # every header stays consistent: only the stream itself is wrong
    entries = explode(build(codec))
    e = entry(entries, "t/0")
    assert e["method"] == CODEC_METHODS[codec]
    e["payload"] = change(e["payload"])
    data = raw_zip(entries)
    with pytest.raises(ContainerError, match=match):
        read_artifact(data)
    if codec == "deflate":
        with pytest.raises(MirrorError):
            load(data)


@pytest.mark.parametrize("codec", ["deflate", pytest.param("zstd", marks=needs_zstd)])
def test_member_decompressing_beyond_its_size(codec):
    entries = explode(build(codec))
    entry(entries, "t/0")["usize"] -= 1
    with pytest.raises(ContainerError, match="more than"):
        read_artifact(raw_zip(entries))


@pytest.mark.parametrize("where", ["needed", "local_needed"])
@pytest.mark.parametrize("needed", [0, 10, 45, 63, 64, 0xFFFF])
def test_version_needed_must_match_method(where, needed):
    entries = explode(build())
    entry(entries, "t/0")[where] = needed
    with pytest.raises(ContainerError, match="version needed to extract is"):
        read_artifact(raw_zip(entries))
    entries = explode(build())
    entry(entries, "baslt.json")[where] = needed
    with pytest.raises(ContainerError, match="version needed to extract is"):
        read_artifact(raw_zip(entries))


@needs_zstd
@pytest.mark.parametrize("needed", [20, 62, 64])
def test_zstd_version_needed_is_63(needed):
    entries = explode(build("zstd"))
    e = entry(entries, "t/0")
    assert e["needed"] == 63
    e["needed"] = needed
    with pytest.raises(ContainerError, match="expected 63"):
        read_artifact(raw_zip(entries))


def test_changed_method():
    data = build()
    entries = explode(data)
    entry(entries, "s/0")["method"] = 12
    with pytest.raises(ContainerError, match="method 12"):
        read_artifact(raw_zip(entries))
    entries = explode(data)
    entry(entries, "s/0")["local_method"] = 0
    with pytest.raises(ContainerError, match="local header method"):
        read_artifact(raw_zip(entries))
    entries = explode(data)
    assert entry(entries, "s/0")["method"] == 8
    entry(entries, "s/0")["method"] = 0
    with pytest.raises(ContainerError):
        read_artifact(raw_zip(entries))


@pytest.mark.parametrize("where", ["local_extra", "central_extra"])
def test_extra_field(where):
    entries = explode(build())
    entry(entries, "index.json")[where] = b"\xfe\xca\x00\x00"
    with pytest.raises(ContainerError, match="extra"):
        read_artifact(raw_zip(entries))


@pytest.mark.parametrize("flags", [0x0001, 0x0008, 0x0800])
@pytest.mark.parametrize("where", ["flags", "local_flags"])
def test_nonzero_flags(flags, where):
    entries = explode(build())
    entry(entries, "t/0")[where] = flags
    with pytest.raises(ContainerError, match="flags"):
        read_artifact(raw_zip(entries))


@pytest.mark.parametrize("cut", [1, 22, 100, "half", "all_but_21", "empty"])
def test_truncated(cut):
    data = build()
    if cut == "half":
        bad = data[: len(data) // 2]
    elif cut == "all_but_21":
        bad = data[:21]
    elif cut == "empty":
        bad = b""
    else:
        bad = data[:-cut]
    with pytest.raises(ContainerError):
        read_artifact(bad)


def test_comment_trailing_and_leading_bytes():
    data = build()
    with pytest.raises(ContainerError, match="last 22 bytes"):
        read_artifact(raw_zip(explode(data), comment=b"hello"))
    with pytest.raises(ContainerError):
        read_artifact(data + b"\x00")
    with pytest.raises(ContainerError):
        read_artifact(b"\x00" * 4 + data)
    with pytest.raises(ContainerError):
        read_artifact(np.random.default_rng(0).integers(0, 256, 5000, dtype=np.uint8).tobytes())


def test_crc_and_size_fields():
    data = build()
    entries = explode(data)
    entry(entries, "s/1")["crc"] ^= 1
    with pytest.raises(ContainerError, match="CRC"):
        read_artifact(raw_zip(entries))
    entries = explode(data)
    entry(entries, "s/1")["usize"] += 1
    with pytest.raises(ContainerError):
        read_artifact(raw_zip(entries))
    entries = explode(data)
    stored = entry(entries, "manifest.json")
    stored["crc"] ^= 0x80000000
    with pytest.raises(ContainerError, match="CRC"):
        read_artifact(raw_zip(entries))


def test_first_member_must_be_stored_baslt_json():
    members = build_members()
    with pytest.raises(ContainerError, match="first member"):
        read_artifact(write_zip([members[1], members[0], *members[2:]]))
    entries = explode(write_zip(members))
    head = entries[0]
    import zlib

    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    head["payload"] = c.compress(header_json()) + c.flush()
    head["method"] = 8
    with pytest.raises(ContainerError, match="STORED"):
        read_artifact(raw_zip(entries))


def test_invalid_names_and_duplicates():
    entries = explode(build())
    entries[-1]["name"] = b"t/"
    with pytest.raises(ContainerError, match="name"):
        read_artifact(raw_zip(entries))
    entries = explode(build())
    entries[-1]["name"] = b"s/0"
    with pytest.raises(ContainerError, match="duplicate"):
        read_artifact(raw_zip(entries))


@pytest.mark.parametrize(
    "header,match",
    [
        ({**header_dict(), "container": 2}, "upgrade"),
        ({**header_dict(), "reader_min": 2}, "upgrade"),
        ({**header_dict(), "format": "zip"}, "format"),
        ({**header_dict(), "container": "1"}, "container"),
        ({k: v for k, v in header_dict().items() if k != "reader_min"}, "reader_min"),
        ({**header_dict(), "byte_order": "big"}, "byte_order"),
    ],
)
def test_header_checks(header, match):
    with pytest.raises(ContainerError, match=match):
        read_artifact(build(header=header))


@pytest.mark.parametrize("key", ["container", "reader_min"])
def test_newer_version_is_reported_before_profile_checks(key, monkeypatch):
    data = build(header={**header_dict(), key: 2})
    variants = [data, raw_zip(explode(data), comment=b"a newer trailer")]
    for name, field, value in (
        ("t/0", "central_extra", b"\x01\x00\x00\x00"),
        ("t/0", "local_extra", b"\x01\x00\x00\x00"),
        ("s/1", "method", 14),
        ("index.json", "needed", 64),
        ("s/0", "flags", 0x0008),
        ("t/0", "payload", b"not a stream"),
    ):
        entries = explode(data)
        entry(entries, name)[field] = value
        variants.append(raw_zip(entries))

    def no_decompression(*_args, **_kwargs):
        raise AssertionError("members were decompressed before the header version was checked")

    monkeypatch.setattr(reader_module, "_read_members", no_decompression)
    for bad in variants:
        with pytest.raises(ContainerError, match="upgrade baslt"):
            read_artifact(bad)


def test_leading_header_is_trusted_only_when_its_crc_matches():
    data = build(header={**header_dict(), "container": 2})
    entries = explode(data)
    entry(entries, "baslt.json")["crc"] ^= 1
    with pytest.raises(ContainerError, match="CRC"):
        read_artifact(raw_zip(entries))


def test_missing_json_member_and_bad_json():
    members = build_members()
    with pytest.raises(ContainerError, match="policy.json"):
        read_artifact(write_zip([m for m in members if m.name != "policy.json"]))
    bad = [Member(m.name, b'{"status":NaN}', m.method) if m.name == "manifest.json" else m for m in members]
    with pytest.raises(ContainerError, match="manifest.json"):
        read_artifact(write_zip(bad))
    bad = [Member(m.name, b"[1,2]", m.method) if m.name == "policy.json" else m for m in members]
    with pytest.raises(ContainerError, match="object"):
        read_artifact(write_zip(bad))


def _with_number(members: list[Member], name: str, number: bytes) -> list[Member]:
    out = []
    for m in members:
        if m.name == name:
            m = Member(m.name, b'{"zz_number":' + number + b"," + bytes(m.data)[1:], m.method)
        out.append(m)
    return out


@pytest.mark.parametrize("name", ["baslt.json", "index.json", "manifest.json", "policy.json"])
@pytest.mark.parametrize("number", [b"1e400", b"-1e400", b"1.8e308", b"[0.5,1E+999]"])
def test_json_numbers_overflowing_to_infinity_are_rejected(name, number):
    with pytest.raises(ContainerError, match="not finite"):
        read_artifact(write_zip(_with_number(build_members(), name, number)))


def test_large_finite_json_numbers_are_accepted():
    members = _with_number(build_members(), "manifest.json", b"[1.7976931348623157e308,-5e-324,1e-400]")
    art = read_artifact(write_zip(members))
    assert art.manifest["zz_number"] == [1.7976931348623157e308, -5e-324, 0.0]


def _edit_array(field, value, signal=0, array=1):
    def edit(index):
        index["signals"][signal]["arrays"][array][field] = value

    return edit


@pytest.mark.parametrize(
    "edit,match",
    [
        (_edit_array("offset", 10_000), "outside"),
        (_edit_array("nbytes", 799), "nbytes"),
        (_edit_array("n", 101), "nbytes"),
        (_edit_array("member", "s/9"), "not a data member"),
        (_edit_array("member", "manifest.json"), "not a data member"),
        (_edit_array("dtype", "<f2"), "dtype"),
        (_edit_array("enc", "lz4"), "encoding"),
        (_edit_array("offset", -1), "non-negative"),
        (_edit_array("n", True), "non-negative integer"),
        (lambda ix: ix["signals"][1].update(name="q_dyn"), "twice"),
        (lambda ix: ix["signals"][0]["arrays"].append(dict(ix["signals"][0]["arrays"][0])), "twice"),
        (lambda ix: ix["signals"][0]["arrays"][1].pop("enc"), "missing"),
        (lambda ix: ix.pop("signals"), "signals"),
        (lambda ix: ix["signals"][0].pop("arrays"), "arrays"),
    ],
)
def test_descriptor_checks(edit, match):
    with pytest.raises(ContainerError, match=match):
        read_artifact(build(index_edit=edit))


def test_zstd_member_without_zstd_support(monkeypatch):
    monkeypatch.setattr(spec_module, "zstd_available", lambda: False)
    entries = explode(build())
    e = entry(entries, "t/0")
    e["method"] = 93
    e["needed"] = 63
    with pytest.raises(ContainerError, match="recompile with --codec deflate"):
        read_artifact(raw_zip(entries))


def test_missing_file(tmp_path):
    with pytest.raises(ContainerError, match="cannot read"):
        read_artifact(tmp_path / "nope.baslt")


def test_error_messages_are_single_clear_lines():
    try:
        read_artifact(build()[:-3])
    except ContainerError as exc:
        assert "\n" not in str(exc) and str(exc).startswith("not a valid .baslt file")
        assert exc.exit_code == 3
    else:
        raise AssertionError("expected ContainerError")


def test_index_json_is_canonical():
    data = build()
    art = read_artifact(data)
    raw = art.members["index.json"]
    assert raw == canonical_json(art.index)
    assert json.loads(raw) == art.index
