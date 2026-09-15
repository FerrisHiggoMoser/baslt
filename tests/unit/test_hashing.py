from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from baslt import hashing


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_full_matches_plain_sha256(tmp_path):
    data = bytes(range(256)) * 1000
    p = _write(tmp_path, "a.bin", data)
    info = hashing.hash_file(p, "full")
    assert info.algorithm == "sha256"
    assert info.mode == "full"
    assert info.covered_bytes == len(data)
    assert info.value == hashlib.sha256(data).hexdigest()
    assert info.is_proof


def test_full_streams_large_files(tmp_path):
    data = b"x" * (hashing.CHUNK * 2 + 12345)
    p = _write(tmp_path, "big.bin", data)
    assert hashing.hash_file(p, "full").value == hashlib.sha256(data).hexdigest()


def test_sampled_is_reproducible_and_is_not_a_plain_sha256(tmp_path):
    data = bytes((i * 7) % 251 for i in range(3_000_000))
    p = _write(tmp_path, "s.bin", data)
    first = hashing.hash_file(p, "sampled")
    second = hashing.hash_file(p, "sampled")
    assert first.value == second.value
    assert first.algorithm == "sha256-sampled-v1"
    assert first.value != hashlib.sha256(data).hexdigest()
    assert not first.is_proof


def test_sampled_covers_small_files_completely(tmp_path):
    data = b"abcdefgh" * 1000
    p = _write(tmp_path, "small.bin", data)
    info = hashing.hash_file(p, "sampled")
    assert info.covered_bytes == len(data)
    expected = hashlib.sha256(
        hashing.SAMPLED_PREFIX + struct.pack("<Q", len(data)) + data
    ).hexdigest()
    assert info.value == expected


def test_sampled_ranges_are_disjoint_sorted_and_inside_the_file():
    for size in (0, 1, 1000, hashing.SAMPLED_BLOCK, 5_000_000, 40_000_000, 9_000_000_000):
        ranges = hashing.sampled_ranges(size)
        assert all(0 <= a < b <= size for a, b in ranges)
        for (_, prev_stop), (start, _) in zip(ranges, ranges[1:]):
            assert prev_stop < start
        if size:
            assert ranges[0][0] == 0
            assert ranges[-1][1] == size


def test_sampled_covered_bytes_bounded_for_huge_files(tmp_path):
    ranges = hashing.sampled_ranges(9_000_000_000)
    covered = sum(b - a for a, b in ranges)
    assert covered <= 2 * hashing.SAMPLED_EDGE + hashing.SAMPLED_BLOCKS * hashing.SAMPLED_BLOCK


def test_sampled_detects_a_change_inside_a_sampled_block(tmp_path):
    data = bytearray(b"q" * 3_000_000)
    p = _write(tmp_path, "t.bin", bytes(data))
    before = hashing.hash_file(p, "sampled")
    data[0] = ord("Z")  # first edge range is always covered
    p.write_bytes(bytes(data))
    assert hashing.hash_file(p, "sampled").value != before.value


def test_sampled_size_change_alone_changes_the_digest(tmp_path):
    p = _write(tmp_path, "u.bin", b"k" * 1000)
    first = hashing.hash_file(p, "sampled")
    p.write_bytes(b"k" * 1001)
    assert hashing.hash_file(p, "sampled").value != first.value


def test_arrays_digest_is_order_independent_but_content_sensitive():
    a = {"t": np.arange(10.0), "x": np.ones(5, dtype=np.int32)}
    b = {"x": np.ones(5, dtype=np.int32), "t": np.arange(10.0)}
    assert hashing.hash_arrays(a).value == hashing.hash_arrays(b).value
    c = {"t": np.arange(10.0), "x": np.ones(5, dtype=np.int64)}
    assert hashing.hash_arrays(c).value != hashing.hash_arrays(a).value
    d = {"t": np.arange(10.0), "y": np.ones(5, dtype=np.int32)}
    assert hashing.hash_arrays(d).value != hashing.hash_arrays(a).value


def test_arrays_digest_handles_shape_and_strides():
    base = np.arange(12.0).reshape(3, 4)
    assert hashing.hash_arrays({"a": base}).value != hashing.hash_arrays({"a": base.reshape(4, 3)}).value
    assert hashing.hash_arrays({"a": base.T}).value == hashing.hash_arrays({"a": np.ascontiguousarray(base.T)}).value
    info = hashing.hash_arrays({"a": base})
    assert info.covered_bytes == base.nbytes and info.mode == "arrays" and info.is_proof


def test_arrays_digest_of_empty_input():
    info = hashing.hash_arrays({})
    assert info.covered_bytes == 0 and info.value


def test_none_mode():
    info = hashing.hash_none()
    assert info.mode == "none" and info.value == "" and not info.is_proof
    assert hashing.matches("/nonexistent/path", info)


def test_matches_and_recompute(tmp_path):
    p = _write(tmp_path, "v.bin", b"hello world" * 100)
    for mode in ("full", "sampled"):
        info = hashing.hash_file(p, mode)
        assert hashing.matches(p, info)
        p.write_bytes(b"hello worlD" * 100)
        assert not hashing.matches(p, info)
        p.write_bytes(b"hello world" * 100)

    arrays = {"t": np.arange(4.0)}
    info = hashing.hash_arrays(arrays)
    assert hashing.matches(arrays, info)
    assert not hashing.matches({"t": np.arange(5.0)}, info)


def test_recompute_rejects_mismatched_sources(tmp_path):
    p = _write(tmp_path, "w.bin", b"data")
    with pytest.raises(ValueError, match="cannot be recomputed from a file path"):
        hashing.recompute(p, hashing.hash_arrays({"t": np.arange(2.0)}))
    with pytest.raises(ValueError, match="cannot recompute"):
        hashing.recompute({"t": np.arange(2.0)}, hashing.hash_file(p, "full"))


def test_default_mode_and_unknown_mode(tmp_path):
    p = _write(tmp_path, "x.bin", b"0")
    assert hashing.default_mode(p) == "full"
    assert hashing.default_mode({"t": np.arange(2.0)}) == "arrays"
    with pytest.raises(ValueError, match="unknown file hash mode"):
        hashing.hash_file(p, "bogus")


def test_json_round_trip(tmp_path):
    p = _write(tmp_path, "y.bin", b"abc")
    info = hashing.hash_file(p, "full")
    again = hashing.HashInfo.from_json(info.to_json())
    assert again == info
    assert set(info.to_json()) == {"algorithm", "mode", "covered_bytes", "value"}
