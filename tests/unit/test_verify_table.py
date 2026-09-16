"""Rendering: the table is the one docs/cli.md prints, and `to_json` carries a top-level status.

The doc's worked example is reproduced line for line, so a change to the column geometry has to be a deliberate
edit to this expectation rather than a quiet drift.
"""

from __future__ import annotations

import json

import pytest

from baslt.verify.checks import VerifyResult
from baslt.verify.structure import Check
from baslt.verify.table import check_label, render, table_lines, to_json

pytestmark = pytest.mark.minimal

# Copied from the `baslt verify` example in docs/cli.md.
DOC_TABLE = [
    "STATUS  CHECK                                         BASIS     SOURCE         ARTIFACT       TOLERANCE",
    "PASS    structure                                     artifact  -              -              -",
    "PASS    hard.q_dyn.global_extrema max                 attested  68142.3 Pa     68142.3 Pa     0",
    "PASS    hard.q_dyn.threshold_crossing[0] crossings    artifact  4              4              5 ms",
    "WARN    sync.flight_dynamics unaligned                artifact  -              3              0",
    "N/A     hard.skin_temp.global_extrema                 -         -              -              -",
]

DOC_CHECKS = [
    Check("structure.container", "pass", "artifact"),
    Check("hard.q_dyn.global_extrema.max", "pass", "attested", "68142.3 Pa", "68142.3 Pa", "0"),
    Check("hard.q_dyn.threshold_crossing[0].crossings", "pass", "artifact", "4", "4", "5 ms"),
    Check("sync.flight_dynamics.unaligned", "warn", "artifact", None, "3", "0"),
    Check("hard.skin_temp.global_extrema", "not_applicable", "-"),
]


def doc_result(**kwargs) -> VerifyResult:
    return VerifyResult(
        checks=list(DOC_CHECKS),
        policy={"name": "flight_review", "sha256": "3f9a1c0b" + "0" * 56},
        source={"path": "run.h5", "digest": {"algorithm": "sha256", "mode": "full", "value": "5e021234" + "0" * 56}},
        artifact={"size_bytes": 1950000, "max_bytes": 2097152, "ratio": 4638.0},
        **kwargs,
    )


# --- the table ---------------------------------------------------------------------------------------------


def test_table_matches_the_documented_example_line_for_line():
    assert table_lines(DOC_CHECKS) == DOC_TABLE


def test_header_block_matches_the_documented_example():
    lines = render(doc_result()).split("\n")
    assert lines[0] == "Policy   flight_review   sha256 3f9a1c0b…"
    # docs/cli.md abbreviates the source digest in prose; the renderer always shows eight hex digits.
    assert lines[1].startswith("Source   run.h5          sha256 (full) ")
    assert lines[2] == "Artifact 1.86 MiB of 2.00 MiB   ratio 4.64e+03"
    assert lines[3] == ""


def test_render_puts_the_table_and_result_around_the_header():
    text = render(doc_result())
    lines = text.split("\n")
    assert lines[4:10] == DOC_TABLE
    assert lines[10] == ""
    assert lines[11] == "Result: PASS WITH WARNINGS (3 pass, 1 warn, 1 n/a, 0 fail)"
    assert text.endswith("\n")


def test_result_line_counts_every_status():
    checks = [Check(f"a.{i}.max", "pass") for i in range(11)]
    checks.append(Check("b.unaligned", "warn"))
    checks.append(Check("c.global_extrema", "not_applicable"))
    result = VerifyResult(checks=checks)
    assert render(result).rstrip().endswith("Result: PASS WITH WARNINGS (11 pass, 1 warn, 1 n/a, 0 fail)")


def test_a_failure_is_reported_as_fail():
    result = VerifyResult(checks=[Check("x.max", "pass"), Check("y.min", "fail")])
    assert render(result).rstrip().endswith("Result: FAIL (1 pass, 0 warn, 0 n/a, 1 fail)")


def test_a_clean_run_is_reported_as_pass():
    result = VerifyResult(checks=[Check("structure.container", "pass")])
    assert render(result).rstrip().endswith("Result: PASS (1 pass, 0 warn, 0 n/a, 0 fail)")


# --- the CHECK column --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("check_id", "expected"),
    [
        ("structure.container", "structure container"),
        ("structure.bind", "structure bind"),
        ("hard.q_dyn.global_extrema.max", "hard.q_dyn.global_extrema max"),
        ("hard.q_dyn.threshold_crossing[0].fidelity", "hard.q_dyn.threshold_crossing[0] fidelity"),
        ("hard.q_dyn.violation[0].runs", "hard.q_dyn.violation[0] runs"),
        ("hard.mode.state_transitions.source", "hard.mode.state_transitions source"),
        ("source.digest", "source digest"),
        # A requirement that could not be evaluated has no aspect and stays whole.
        ("hard.skin_temp.global_extrema", "hard.skin_temp.global_extrema"),
        ("hard.mode.state_transitions", "hard.mode.state_transitions"),
        ("hard.q_dyn.threshold_crossing[0]", "hard.q_dyn.threshold_crossing[0]"),
    ],
)
def test_check_label_splits_only_a_known_aspect(check_id, expected):
    assert check_label(check_id) == expected


# --- the structure row -------------------------------------------------------------------------------------


def test_structure_collapses_to_one_row_while_it_passes():
    checks = [Check(f"structure.{name}", "pass") for name in
              ("container", "descriptors", "invariants", "size", "policy", "bind")]
    rows = table_lines(checks)
    assert len(rows) == 2
    assert rows[1].startswith("PASS    structure ")


def test_structure_expands_when_a_check_does_not_pass():
    checks = [Check("structure.container", "pass"), Check("structure.bind", "fail", message="mismatch")]
    rows = table_lines(checks)
    assert len(rows) == 3
    assert "structure container" in rows[1]
    assert "structure bind" in rows[2]


# --- geometry ----------------------------------------------------------------------------------------------


def test_columns_stay_aligned_when_a_check_id_is_long():
    long_id = "hard." + "a" * 60 + ".global_extrema.max"
    rows = table_lines([Check(long_id, "pass", "artifact", "1", "1", "0")])
    basis_at = rows[0].index("BASIS")
    assert rows[1].index("artifact") == basis_at
    assert rows[1].index(check_label(long_id)) == rows[0].index("CHECK")


def test_missing_values_render_as_dashes():
    rows = table_lines([Check("x.max", "pass", "artifact", None, "", None)])
    assert rows[1][rows[0].index("BASIS"):].split() == ["artifact", "-", "-", "-"]


def test_rows_have_no_trailing_whitespace():
    for line in render(doc_result()).split("\n"):
        assert line == line.rstrip()


def test_an_artifact_without_a_budget_omits_the_ceiling():
    result = VerifyResult(checks=[], artifact={"size_bytes": 2048, "max_bytes": None, "ratio": None})
    assert render(result).split("\n")[0] == "Artifact 2.00 KiB"


def test_a_source_without_a_digest_says_so():
    result = VerifyResult(checks=[], source={"path": "run.csv", "digest": {"mode": "none"}}, artifact={})
    assert "no digest" in render(result).split("\n")[1]


# --- json --------------------------------------------------------------------------------------------------


def test_to_json_has_a_top_level_status():
    payload = to_json(doc_result())
    assert payload["status"] == "pass_with_warnings"
    assert payload["exit_code"] == 0
    assert payload["counts"] == {"pass": 3, "warn": 1, "not_applicable": 1, "fail": 0}
    assert json.loads(json.dumps(payload)) == payload


def test_to_json_lists_every_check_with_its_fields():
    payload = to_json(doc_result())
    assert [check["id"] for check in payload["checks"]] == [check.id for check in DOC_CHECKS]
    first = payload["checks"][1]
    assert first == {
        "id": "hard.q_dyn.global_extrema.max",
        "status": "pass",
        "basis": "attested",
        "claimed": "68142.3 Pa",
        "measured": "68142.3 Pa",
        "allowed": "0",
        "message": "",
    }


def test_to_json_reports_strict_and_its_exit_code():
    payload = to_json(doc_result(strict=True))
    assert payload["strict"] is True
    assert payload["exit_code"] == 1  # --strict turns a warning into a failing exit code


def test_result_and_to_json_agree_through_the_dataclass():
    result = doc_result()
    assert result.to_json() == to_json(result)
    assert result.render() == render(result)
