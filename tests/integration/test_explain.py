"""`baslt explain`: suggestions from an infeasible-budget report, measured ones against the source, and the CLI."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from baslt import explain
from baslt.advice import POLICY_EDIT_MARGIN, SHOWN_RELAXATIONS, render, suggested_budget
from baslt.api import compile
from baslt.cli import main
from baslt.errors import InfeasibleBudget, UsageError

pytestmark = pytest.mark.minimal

T = np.linspace(0.0, 20.0, 20001)
RNG = np.random.default_rng(11)
ALPHA = 7.0 + 0.05 * RNG.standard_normal(T.size)  # hovers around the 7 deg level and crosses it constantly
ALPHA[:2000] = 0.0
Q = np.sin(T) * 1000.0
SOURCE = {"alpha": (T, ALPHA), "q": (T, Q)}
POLICY = {
    "version": 1,
    "name": "explain_probe",
    "artifact": {"max_size": "8 KiB"},
    "signals": {"decl": {"alpha": {"unit": "deg"}, "q": {"unit": "Pa"}}},
    "hard": {
        "alpha": {"threshold_crossing": [{"value": "7 deg"}], "window_extrema": {"interval": "0.1 s"}},
        "q": {"global_extrema": {}},
    },
    "soft": [{"match": "*", "priority": "low"}],
}


@pytest.fixture(scope="module")
def report_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("explain") / "run.error.json"
    result = compile(SOURCE, POLICY, on_error="record", error_report=path)
    assert result.status == "error" and result.exit_code == 2
    return path


@pytest.fixture(scope="module")
def measured(report_path):
    return explain(report_path, source=SOURCE, policy=POLICY)


def report(path):
    return json.loads(path.read_text())


# --- report only ----------------------------------------------------------------------------------------------


def test_the_report_alone_gives_a_budget_and_the_largest_requirements(report_path):
    result = explain(report_path)
    data = report(report_path)
    assert result["status"] == "pass"
    assert result["minimum_bytes"] == data["minimum_bytes"] > data["max_bytes"] == 8192
    assert result["excess_bytes"] == data["minimum_bytes"] - 8192
    assert result["measured"] is None

    budget, *requirements = result["suggestions"]
    assert budget["max_bytes"] == suggested_budget(data["minimum_bytes"])
    assert budget["max_bytes"] % (64 * 1024) == 0 and budget["max_bytes"] >= data["minimum_bytes"]
    assert [r["target"] for r in requirements][:2] == ["hard.alpha.threshold_crossing[0]", "hard.alpha.window_extrema"]
    sizes = [r["standalone_bytes"] for r in requirements]
    assert sizes == sorted(sizes, reverse=True)
    assert "hysteresis" in requirements[0]["change"]


def test_a_mapping_works_as_well_as_a_path(report_path):
    assert explain(report(report_path)) == explain(report_path)


def test_requirements_without_samples_are_left_out(report_path):
    data = report(report_path)
    data["requirements"].append({"id": "hard.q.violation[0]", "signal": "q", "op": "violation",
                                 "samples": 0, "standalone_bytes": 0})
    assert "hard.q.violation[0]" not in [s["target"] for s in explain(data)["suggestions"]]


@pytest.mark.parametrize("minimum, expected", [(1, 65536), (65536, 65536), (65537, 131072), (1048577, 1114112)])
def test_suggested_budgets_round_up_to_64_kib(minimum, expected):
    assert suggested_budget(minimum) == expected


def test_budget_text_uses_whole_mebibytes_when_it_can(report_path):
    data = report(report_path)
    data["minimum_bytes"] = 2 * 1024 * 1024 - 5
    assert explain(data)["suggestions"][0]["max_size"] == "2 MiB"
    data["minimum_bytes"] = 100_000
    assert explain(data)["suggestions"][0]["max_size"] == "128 KiB"


def test_only_infeasible_reports_are_explained(tmp_path, report_path):
    other = tmp_path / "other.error.json"
    other.write_text(json.dumps({"status": "error", "error": "SourceError", "message": "no such file"}))
    with pytest.raises(UsageError, match="infeasible-budget report"):
        explain(other)
    with pytest.raises(UsageError, match="no error report"):
        explain(tmp_path / "missing.error.json")
    broken = tmp_path / "broken.error.json"
    broken.write_text("{not json")
    with pytest.raises(UsageError, match="not a readable JSON"):
        explain(broken)
    with pytest.raises(UsageError, match="both --source and --policy"):
        explain(report_path, source=SOURCE)
    with pytest.raises(UsageError, match="both --source and --policy"):
        explain(report_path, policy=POLICY)


# --- measured -------------------------------------------------------------------------------------------------


def test_measured_relaxations_start_from_the_reported_minimum(measured, report_path):
    assert measured["measured"]["minimum_bytes"] == report(report_path)["minimum_bytes"]
    assert measured["measured"]["max_bytes"] == 8192
    rows = measured["measured"]["relaxations"]
    assert {row["target"] for row in rows} >= {"hard.alpha.threshold_crossing[0]", "hard.alpha.window_extrema",
                                               "hard.q.global_extrema"}
    order = [(not row["fits"], row["minimum_bytes"]) for row in rows]
    assert order == sorted(order)
    for row in rows:
        assert row["saved_bytes"] == measured["measured"]["minimum_bytes"] - row["minimum_bytes"]
        assert row["fits"] == (row["minimum_bytes"] + POLICY_EDIT_MARGIN <= 8192)


def relaxed_policy(row):
    policy = copy.deepcopy(POLICY)
    signal, op = row["target"].split(".")[1:3]
    op = op.split("[")[0]
    if row["change"] == "demote to soft":
        del policy["hard"][signal][op]
        if not policy["hard"][signal]:
            del policy["hard"][signal]
    elif row["change"].startswith("interval "):
        policy["hard"][signal][op]["interval"] = row["change"].split("-> ")[1]
    else:
        return None
    return policy


def compiled_minimum(policy) -> int | None:
    """The reported minimum under the policy's own 8 KiB budget, or None when the policy fits it."""
    try:
        compile(SOURCE, policy)
    except InfeasibleBudget as exc:
        return exc.report["minimum_bytes"]
    return None


def test_measured_sizes_hold_within_the_policy_edit_margin(measured):
    tried = 0
    for row in measured["measured"]["relaxations"]:
        policy = relaxed_policy(row)
        if policy is None:
            continue
        tried += 1
        edge = row["minimum_bytes"]
        actual = compiled_minimum(policy)
        if row["fits"]:
            assert actual is None, row
        if actual is None:  # it fits: the measurement may not have claimed more than the margin above the budget
            # A drop removes its whole stanza from the embedded policy text, which is worth far more than the
            # margin the measurement leaves for an edit, so its claim is pessimistic by more than the margin.
            if row["change"] != "demote to soft":
                assert edge - POLICY_EDIT_MARGIN <= 8192, row
            continue
        # A change never costs more than measured plus the margin; removing policy text only makes it smaller.
        assert actual <= edge + POLICY_EDIT_MARGIN, row
        if row["change"] != "demote to soft":
            assert actual >= edge - POLICY_EDIT_MARGIN, row
    assert tried >= 4


def test_the_combined_plan_loosens_before_it_drops(measured):
    combined = measured["measured"]["combined"]
    assert combined["steps"]
    changes = [step["change"] for step in combined["steps"]]
    drops = [i for i, change in enumerate(changes) if change == "demote to soft"]
    assert all(i >= len(changes) - len(drops) for i in drops)  # drops only come after every loosening
    sizes = [step["minimum_bytes"] for step in combined["steps"]]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)
    assert combined["minimum_bytes"] == sizes[-1]
    assert combined["fits"] == (sizes[-1] + POLICY_EDIT_MARGIN <= 8192)


def test_a_larger_budget_can_be_measured_instead(report_path):
    result = explain(report_path, source=SOURCE, policy=POLICY, max_size="1 MiB")
    assert result["measured"]["max_bytes"] == 1024 * 1024
    assert all(row["fits"] for row in result["measured"]["relaxations"])
    assert result["measured"]["combined"]["steps"] == []


def test_a_cli_budget_is_measured_again(tmp_path):
    policy = {**POLICY, "artifact": {}}
    path = tmp_path / "cli.error.json"
    result = compile(SOURCE, policy, max_size=5000, on_error="record", error_report=path)
    assert result.status == "error"
    assert report(path)["budget_source"] == "cli"
    assert explain(path, source=SOURCE, policy=policy)["measured"]["max_bytes"] == 5000


def test_the_text_form(measured):
    text = render(measured)
    assert text.startswith("The hard requirements need ")
    assert "Smallest budget that works: max_size: " in text
    assert "Largest requirements" in text
    assert "Measured on the source" in text
    assert "Together (" in text
    rows = measured["measured"]["relaxations"]
    if len(rows) > SHOWN_RELAXATIONS:
        assert f"... {len(rows) - SHOWN_RELAXATIONS} more with --json" in text


# --- command line ---------------------------------------------------------------------------------------------


def test_cli_explain(report_path, capsys):
    assert main(["explain", str(report_path)]) == 0
    out = capsys.readouterr().out
    assert "Smallest budget that works" in out and "Measured" not in out

    assert main(["explain", str(report_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass" and payload["kind"] == "explanation"


def test_cli_explain_measures_with_source_and_policy(tmp_path, capsys):
    source = tmp_path / "run.csv"
    rows = np.column_stack([T, ALPHA, Q])
    np.savetxt(source, rows, delimiter=",", header="t,alpha,q", comments="", fmt="%.17g")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({**POLICY, "signals": {**POLICY["signals"], "time": "t"}}))
    report_file = tmp_path / "run.error.json"
    code = main(["compile", str(source), "--policy", str(policy), "-o", str(tmp_path / "run.baslt"),
                 "--error-report", str(report_file)])
    assert code == 2 and report_file.exists()
    capsys.readouterr()
    assert main(["explain", str(report_file), "--source", str(source), "--policy", str(policy)]) == 0
    assert "Measured on the source" in capsys.readouterr().out


def test_cli_explain_usage_errors(tmp_path, report_path, capsys):
    assert main(["explain", str(tmp_path / "missing.json")]) == 3
    assert main(["explain", str(report_path), "--source", "x.csv"]) == 3
    assert "both --source and --policy" in capsys.readouterr().err
