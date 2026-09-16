"""local_extrema through compile and verify, including tampered artifacts."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.api import compile, verify
from reference.artifact_edit import position, rebuild

T = np.linspace(0.0, 20.0, 4001)
X = np.sin(T * 3) + 0.3 * np.sin(T * 17) + np.where(T > 12, 2.0, 0.0)
X[900:905] = np.nan
POS = np.column_stack([np.sin(T), np.cos(T * 2), T])
POLICY = {"version": 1, "hard": {
    "x": {"local_extrema": {"prominence": 0.5, "separation": "0.4 s"}},
    "pos": {"local_extrema": {"prominence": 0.2, "kind": "max"}},
}}


@pytest.fixture(scope="module")
def artifact():
    return compile({"t": T, "x": X, "pos": POS}, POLICY).bytes


def _checks(data, **kwargs):
    result = verify(data, **kwargs)
    return result, {c.id: c for c in result.checks}


def test_compiles_and_verifies_with_the_source(artifact):
    result, checks = _checks(artifact, source={"x": (T, X), "pos": (T, POS)})
    assert result.status == "pass", result.render()
    assert checks["hard.x.local_extrema.prominence"].basis == "artifact"
    assert checks["hard.x.local_extrema.peaks"].basis == "source"
    assert "hard.x.local_extrema.separation" in checks
    assert "hard.pos.local_extrema.separation" not in checks


def _peak_and_bits(data, signal):
    from baslt.container.reader import read_artifact

    art = read_artifact(data)
    entry = next(e for e in art.index["signals"] if e["name"] == signal)
    legend = {r["id"]: r["bit"] for r in entry["roles"]}
    claim = next(r for r in art.manifest["requirements"] if r["signal"] == signal)
    return claim, legend[f"{claim['id']}#peak"], legend[f"{claim['id']}#base"]


def test_dropping_a_base_sample_is_caught(artifact):
    claim, _, base_bit = _peak_and_bits(artifact, "x")
    peak = claim["evidence"]["peaks"][0]

    def drop_bases(decoded):
        arrays = decoded["x"]
        flagged = (arrays["roles"].astype(np.uint64) & np.uint64(1 << base_bit)) != 0
        keep = ~flagged
        keep[position(arrays, peak["index"])] = True
        for name in ("t", "v", "idx", "roles"):
            arrays[name] = arrays[name][keep]

    tampered = rebuild(artifact, arrays=drop_bases)
    result, checks = _checks(tampered)
    assert checks["hard.x.local_extrema.prominence"].status == "fail", checks["hard.x.local_extrema.prominence"]
    assert result.exit_code == 1


def test_flagging_a_non_peak_is_caught(artifact):
    _, peak_bit, _ = _peak_and_bits(artifact, "x")

    def flag_extra(decoded):
        roles = decoded["x"]["roles"]
        unflagged = np.flatnonzero((roles.astype(np.uint64) & np.uint64(1 << peak_bit)) == 0)
        roles[unflagged[len(unflagged) // 2]] |= roles.dtype.type(1 << peak_bit)

    result, checks = _checks(rebuild(artifact, arrays=flag_extra))
    assert checks["hard.x.local_extrema.prominence"].status == "fail"


def test_overstated_prominence_is_caught(artifact):
    def inflate(manifest):
        claim = next(r for r in manifest["requirements"] if r["id"] == "hard.x.local_extrema")
        claim["evidence"]["peaks"][0]["prominence"] = 99.0

    _, checks = _checks(rebuild(artifact, manifest=inflate))
    assert checks["hard.x.local_extrema.prominence"].status == "fail"
    assert "below the claimed" in checks["hard.x.local_extrema.prominence"].message


def test_a_hidden_source_peak_is_caught_with_the_source(artifact):
    def undercount(manifest):
        claim = next(r for r in manifest["requirements"] if r["id"] == "hard.x.local_extrema")
        claim["evidence"]["maxima"] -= 1

    tampered = rebuild(artifact, manifest=undercount)
    _, checks = _checks(tampered, source={"x": (T, X), "pos": (T, POS)})
    assert checks["hard.x.local_extrema.peaks"].status == "fail"


def test_crowded_claims_break_the_separation_check(artifact):
    def crowd(manifest):
        claim = next(r for r in manifest["requirements"] if r["id"] == "hard.x.local_extrema")
        items = claim["evidence"]["peaks"]
        same = [p for p in items if p["kind"] == "max"]
        same[1]["t"] = same[0]["t"] + 0.1

    _, checks = _checks(rebuild(artifact, manifest=crowd))
    assert checks["hard.x.local_extrema.separation"].status == "fail"
