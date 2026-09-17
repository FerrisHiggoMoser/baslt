"""The report page shell: packed arrays, embedded JSON, escaping and the content security policy."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from html.parser import HTMLParser

import numpy as np
import pytest

from baslt.report.html import Packer, asset, esc, json_script, json_text, page

pytestmark = pytest.mark.minimal


def unpack(blob: str) -> bytes:
    return zlib.decompress(base64.b64decode(blob), -15)


def read(ref: dict, data: bytes) -> np.ndarray:
    if ref["k"] == "f4":
        return np.frombuffer(data, dtype="<f4", count=ref["n"], offset=ref["o"]).astype(np.float64)
    codes = np.frombuffer(data, dtype="<u2", count=ref["n"], offset=ref["o"])
    values = ref["lo"] + codes * ((ref["hi"] - ref["lo"]) / 65534)
    return np.where(codes == 65535, np.nan, values)


def test_arrays_round_trip_through_the_blob():
    packer = Packer()
    times = packer.float32([0.0, 0.5, 1.25])
    values = packer.uint16([1.0, np.nan, 3.0, 2.0, -np.inf])
    flat = packer.uint16([7.0, 7.0])
    empty = packer.uint16([])
    odd = packer.uint16([1.0], lo=0.0, hi=2.0)
    data = unpack(packer.blob())
    assert [ref["o"] % 4 for ref in (times, values, flat, empty, odd)] == [0] * 5
    assert len(data) == packer.size
    assert read(times, data).tolist() == [0.0, 0.5, 1.25]
    got = read(values, data)
    assert got[0] == 1.0 and np.isnan(got[1]) and got[2] == 3.0 and got[3] == pytest.approx(2.0, abs=4e-5)
    assert np.isnan(got[4])
    assert (values["lo"], values["hi"]) == (1.0, 3.0)
    assert read(flat, data).tolist() == [7.0, 7.0]
    assert empty["n"] == 0 and read(odd, data).tolist() == [pytest.approx(1.0, abs=2e-5)]
    size = {"f4": 4, "u2": 2}
    for ref, digest in zip((times, values, flat, empty, odd), packer.digests):
        chunk = data[ref["o"]:ref["o"] + ref["n"] * size[ref["k"]]]
        assert hashlib.sha256(chunk).hexdigest() == digest


def test_coarse_codes():
    packer = Packer()
    ref = packer.uint8q([0.0, 5.0, np.nan, 20.0, 10.0], 0.0, 10.0)
    flat = packer.uint8q([3.0], 3.0, 3.0)
    data = unpack(packer.blob())
    codes = np.frombuffer(data, dtype=np.uint8, count=5, offset=ref["o"])
    assert codes.tolist() == [0, 127, 255, 254, 254]  # values beyond hi are clipped
    assert (ref["k"], ref["lo"], ref["hi"]) == ("q1", 0.0, 10.0)
    assert np.frombuffer(data, dtype=np.uint8, count=1, offset=flat["o"]).tolist() == [0]
    small = packer.uint8([1, 2, 255])
    assert small["k"] == "u1" and unpack(packer.blob())[small["o"]:small["o"] + 3] == bytes([1, 2, 255])


def test_quantization_error_is_bounded():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1e5, 2000)
    packer = Packer()
    ref = packer.uint16(x)
    got = read(ref, unpack(packer.blob()))
    assert np.max(np.abs(got - x)) <= (x.max() - x.min()) / 65534
    assert got.min() == pytest.approx(x.min(), rel=1e-12) and got.max() == pytest.approx(x.max(), rel=1e-12)


def test_json_in_script_elements_cannot_close_them():
    data = {"text": "</script><script>alert(1)</script>", "line": "a\u2028b\u2029c", "amp": "<!-- & -->"}
    element = json_script(data, "manifest")
    inner = element[len('<script type="application/json" id="manifest">'):-len("</script>")]
    assert "<" not in inner and ">" not in inner and "\u2028" not in inner
    assert json.loads(inner) == data
    with pytest.raises(ValueError):
        json_text({"bad": float("nan")})


def test_escaping():
    assert esc('<a href="x">&\'') == "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;"
    assert esc(None) == ""


class Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, dict]] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if "id" in attrs:
            self.ids.append(attrs["id"])


def test_the_page_allows_only_its_own_script_and_style():
    text = page("T <1>", "<main><p>hi</p></main>", styles=["p{color:red}"], scripts=["console.log(1)"],
                data={"manifest": {"a": 1}}, blob="AAAA", description="d")
    parser = Collector()
    parser.feed(text)
    meta = {a.get("http-equiv"): a.get("content") for tag, a in parser.tags if tag == "meta"}
    policy = meta["Content-Security-Policy"]
    script_hash = base64.b64encode(hashlib.sha256(b"console.log(1)").digest()).decode()
    style_hash = base64.b64encode(hashlib.sha256(b"p{color:red}").digest()).decode()
    assert f"script-src 'sha256-{script_hash}'" in policy and f"style-src 'sha256-{style_hash}'" in policy
    assert "default-src 'none'" in policy and "unsafe" not in policy
    assert "<title>T &lt;1&gt;</title>" in text
    assert parser.ids == ["manifest", "blob"]


def test_the_assets_load_nothing_from_outside():
    for name in ("report.js", "report.css"):
        text = asset(name)
        assert all(ord(ch) < 128 for ch in text), name
        assert not re.search(r"https?:|//[a-z0-9.-]+\.[a-z]{2,}/|@import|url\(|fetch\(|XMLHttpRequest|eval\(",
                             text), name
    assert len(asset("report.js").encode()) + len(asset("report.css").encode()) <= 70_000
