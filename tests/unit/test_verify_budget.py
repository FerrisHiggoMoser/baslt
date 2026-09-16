"""`budget.accounting` and `budget.errors` on compiled artifacts, and a tamper for every rule they enforce."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.api import compile, verify
from baslt.container.reader import read_artifact
from reference.artifact_edit import rebuild

pytestmark = pytest.mark.minimal

N = 4001
T = np.linspace(0.0, 40.0, N)
X = np.sin(T) + 0.1 * np.sin(T * 17.0)
Y = np.cos(T * 0.5)
MODE = (np.sin(T * 3.0) > 0).astype(np.int64)
SOURCE = {"x": (T, X), "y": (T, Y), "mode": (T, MODE)}
POLICY = {
    "version": 1,
    "name": "budget_probe",
    "artifact": {"max_size": "24 KiB"},
    "signals": {"decl": {"mode": {"kind": "discrete"}}},
    "hard": {"x": {"global_extrema": {}}, "mode": {"state_transitions": {}}},
    "soft": [{"match": "x", "priority": "high"}, {"match": "*", "priority": "low"}],
    "sync_groups": {"g": {"members": ["x", "y"]}},
}


@pytest.fixture(scope="module")
def artifact() -> bytes:
    result = compile(SOURCE, POLICY)
    assert result.status == "pass"
    assert result.manifest["budget"]["discretionary_bytes"] > 0
    return result.bytes


@pytest.fixture(scope="module")
def hard_only() -> bytes:
    return compile(SOURCE, {**POLICY, "soft": [{"match": "*", "priority": "none"}]}).bytes


def checks(data, **kwargs):
    result = verify(data, **kwargs)
    return result, {check.id: check for check in result.checks}


def expect_fail(data, check_id, fragment, **kwargs):
    result, found = checks(data, **kwargs)
    check = found[check_id]
    assert check.status == "fail", f"{check_id} is {check.status}: {check.message}"
    assert fragment in check.message, check.message
    assert result.status == "fail"
    return check


def budget_edit(change):
    def edit(manifest):
        change(manifest["budget"])
    return edit


def row_edit(name, change):
    def edit(manifest):
        change(next(r for r in manifest["budget"]["signals"] if r["name"] == name))
    return edit


# --- well-formed artifacts ------------------------------------------------------------------------------------


def test_a_compiled_artifact_passes_both_checks(artifact):
    result, found = checks(artifact)
    assert found["budget.accounting"].status == "pass"
    assert found["budget.accounting"].basis == "artifact"
    assert "budget.errors" not in found
    assert result.status == "pass"

    result, found = checks(artifact, source=SOURCE)
    assert found["budget.errors"].status == "pass"
    assert found["budget.errors"].basis == "source"
    assert result.status == "pass"
    assert "budget accounting" in result.render()
    assert "budget errors" in result.render()


def test_a_shared_clock_is_paid_for_once(artifact):
    parsed = read_artifact(artifact)
    clocks = {name: next(a["member"] for a in parsed.signal(name)["arrays"] if a["name"] == "t")
              for name in ("x", "y")}
    assert clocks["x"] == clocks["y"]  # the sync group gives both the same retained timestamps
    sizes = {entry.name: entry.compressed_size for entry in parsed.entries}
    rows = {r["name"]: r for r in parsed.manifest["budget"]["signals"]}
    x_member = parsed.signal("x")["arrays"][1]["member"]
    y_member = parsed.signal("y")["arrays"][1]["member"]
    assert rows["x"]["bytes"] == sizes[x_member] + sizes[clocks["x"]]
    assert rows["y"]["bytes"] == sizes[y_member]


def test_rebuilding_without_edits_reproduces_the_artifact(artifact, hard_only):
    assert rebuild(artifact) == artifact
    assert rebuild(hard_only, budget=False) == hard_only


# --- byte totals ----------------------------------------------------------------------------------------------


def test_missing_budget_section(artifact):
    data = rebuild(artifact, manifest=lambda m: m.pop("budget"), budget=False)
    expect_fail(data, "budget.accounting", "no 'budget' section")


def test_totals_must_add_up_to_the_file(artifact):
    data = rebuild(artifact, manifest=budget_edit(lambda b: b.update(overhead_bytes=b["overhead_bytes"] + 1)),
                   budget=False)
    expect_fail(data, "budget.accounting", "required + discretionary + overhead bytes are")


def test_overhead_cannot_hide_data_bytes(artifact):
    def shift(budget):
        budget["required_bytes"] += 10
        budget["overhead_bytes"] -= 10
    data = rebuild(artifact, manifest=budget_edit(shift), budget=False)
    expect_fail(data, "budget.accounting", "the data members take")


@pytest.mark.parametrize("value", [-1, 1.5, "12", None, True])
def test_byte_fields_are_non_negative_integers(artifact, value):
    data = rebuild(artifact, manifest=budget_edit(lambda b: b.update(discretionary_bytes=value)), budget=False)
    expect_fail(data, "budget.accounting", "expected a non-negative integer")


def test_discretionary_bytes_need_soft_samples(hard_only):
    def invent(budget):
        budget["required_bytes"] -= 5
        budget["discretionary_bytes"] += 5
    data = rebuild(hard_only, manifest=budget_edit(invent), budget=False)
    expect_fail(data, "budget.accounting", "no signal has soft samples")


def test_budget_source_matches_the_budget(artifact):
    data = rebuild(artifact, manifest=budget_edit(lambda b: b.update(source="none")), budget=False)
    expect_fail(data, "budget.accounting", "budget.source is 'none' but artifact.max_bytes is")
    data = rebuild(artifact, manifest=budget_edit(lambda b: b.update(source="guess")), budget=False)
    expect_fail(data, "budget.accounting", "expected one of")


# --- rows -----------------------------------------------------------------------------------------------------


def test_every_signal_needs_one_row_in_index_order(artifact):
    data = rebuild(artifact, manifest=budget_edit(lambda b: b["signals"].pop()))
    expect_fail(data, "budget.accounting", "missing ['mode']")
    data = rebuild(artifact, manifest=budget_edit(lambda b: b["signals"].reverse()))
    expect_fail(data, "budget.accounting", "in a different order")
    data = rebuild(artifact, manifest=budget_edit(lambda b: b["signals"].append(dict(b["signals"][0]))))
    expect_fail(data, "budget.accounting", "lists 'x' twice")
    data = rebuild(artifact, manifest=budget_edit(lambda b: b["signals"].append({**b["signals"][0], "name": "z"})))
    expect_fail(data, "budget.accounting", "extra ['z']")


def test_retained_must_match_the_index(artifact):
    def grow(row):
        row["retained"] += 1
        row["soft"] += 1
    expect_fail(rebuild(artifact, manifest=row_edit("y", grow)), "budget.accounting", "index.json n is")


def test_hard_and_soft_must_add_up(artifact):
    data = rebuild(artifact, manifest=row_edit("y", lambda row: row.update(soft=row["soft"] + 1)))
    expect_fail(data, "budget.accounting", "is not retained")


def test_row_bytes_must_match_the_members(artifact):
    data = rebuild(artifact, manifest=row_edit("x", lambda row: row.update(bytes=row["bytes"] + 1)), budget=False)
    expect_fail(data, "budget.accounting", "but its members")


def test_hard_covers_every_contract_sample(artifact):
    def shrink(row):
        row["soft"] += row["hard"]
        row["hard"] = 0
    expect_fail(rebuild(artifact, manifest=row_edit("x", shrink)), "budget.accounting", "carry a contract role")


def test_soft_covers_every_soft_only_sample(artifact):
    def hide(row):
        row["hard"] += row["soft"]
        row["soft"] = 0
    # mode is in no sync group, so its preview samples carry the soft role alone
    expect_fail(rebuild(artifact, manifest=row_edit("mode", hide)), "budget.accounting", "carry only the soft role")


@pytest.mark.parametrize("value", [0.5, "0.5", "5.000000e-01 ", "-5.00000e-01", " 5.00000e-01", "abcdefghijkl"])
def test_errors_are_twelve_character_strings(artifact, value):
    data = rebuild(artifact, manifest=row_edit("x", lambda row: row.update(soft_max_abs_err=value)))
    expect_fail(data, "budget.accounting", "soft_max_abs_err")


@pytest.mark.parametrize("value", ["         nan", "         inf", "0.000000e+00", "1.250000e+03"])
def test_error_strings_that_parse(artifact, value):
    data = rebuild(artifact, manifest=row_edit("y", lambda row: row.update(soft_max_abs_err=value)))
    assert checks(data)[1]["budget.accounting"].status == "pass"


# --- errors against the source --------------------------------------------------------------------------------


def claimed_error(data, name) -> float:
    return float(next(r for r in read_artifact(data).manifest["budget"]["signals"] if r["name"] == name)
                 ["soft_max_abs_err"])


def test_a_wrong_error_fails_with_the_source(artifact):
    data = rebuild(artifact, manifest=row_edit("x", lambda row: row.update(soft_max_abs_err="9.000000e+00")))
    assert checks(data)[1]["budget.accounting"].status == "pass"
    check = expect_fail(data, "budget.errors", "the source gives", source=SOURCE)
    assert check.basis == "source"


def test_an_error_within_the_formatting_rounding_passes(artifact):
    value = claimed_error(artifact, "x") * (1 + 4e-7)
    data = rebuild(artifact, manifest=row_edit("x", lambda row: row.update(soft_max_abs_err="%.6e" % value)))
    assert checks(data, source=SOURCE)[1]["budget.errors"].status == "pass"


def test_a_claimed_zero_error_is_checked(artifact):
    # state_transitions keeps every change of mode, so its hold reconstruction is exact; x is only previewed.
    assert claimed_error(artifact, "mode") == 0
    assert claimed_error(artifact, "x") > 0
    data = rebuild(artifact, manifest=row_edit("x", lambda row: row.update(soft_max_abs_err="0.000000e+00")))
    expect_fail(data, "budget.errors", "'x'", source=SOURCE)


def test_an_edited_value_changes_the_error(artifact):
    def spike(decoded):
        v = decoded["y"]["v"].copy()
        v[len(v) // 2] += 50.0  # y has no contract of its own, so only the source can tell
        decoded["y"]["v"] = v
    data = rebuild(artifact, arrays=spike)
    result, found = checks(data)
    assert result.status == "pass"
    _, found = checks(data, source=SOURCE)
    assert found["source.samples"].status == "fail"
    assert found["budget.errors"].status == "fail"
