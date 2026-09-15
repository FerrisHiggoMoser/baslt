"""Reading policies from YAML, JSON, text and mappings, with file locations."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from types import MappingProxyType

import pytest

from baslt.errors import PolicyError, UsageError
from baslt.policy import Policy, load_policy
from baslt.policy.parse import parse_json, parse_yaml

DOCS = Path(__file__).resolve().parents[2] / "docs" / "policy.md"


def doc_example() -> str:
    text = DOCS.read_text(encoding="utf-8")
    return re.search(r"## Complete example\s*```yaml\n(.*?)```", text, re.S).group(1)


LOCATED = """\
version: 1
name: demo
hard:
  q_dyn:
    threshold_crossing:
      - {value: 65 kPa, edge: rising}
      - value: 7 kPa
        edge: falling
events:
  E:
    when: {signal: q_dyn, rises_above: 1 kPa}
"""


def test_docs_example_loads():
    policy = load_policy(doc_example(), format="yaml")
    assert isinstance(policy, Policy)
    assert policy.name == "flight_review"
    assert policy.source_format == "yaml"
    assert policy.source_text == doc_example()
    assert len(policy.hard) == 8
    assert policy.locations["hard.q_dyn"] == "<string>:25:3"
    assert policy.locations["hard.q_dyn.threshold_crossing[0]"] == "<string>:28:9"
    assert policy.locations["hard.q_dyn.threshold_crossing[0].edge"] == "<string>:28:25"
    assert policy.locations["signals.decl.q_dyn.unit"] == "<string>:18:31"


def test_yaml_file_locations(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(LOCATED, encoding="utf-8")
    policy = load_policy(path)
    loc = policy.locations
    f = str(path)
    assert loc["version"] == f"{f}:1:1"
    assert loc["name"] == f"{f}:2:1"
    assert loc["hard"] == f"{f}:3:1"
    assert loc["hard.q_dyn"] == f"{f}:4:3"
    assert loc["hard.q_dyn.threshold_crossing"] == f"{f}:5:5"
    assert loc["hard.q_dyn.threshold_crossing[0]"] == f"{f}:6:9"
    assert loc["hard.q_dyn.threshold_crossing[0].value"] == f"{f}:6:10"
    assert loc["hard.q_dyn.threshold_crossing[0].edge"] == f"{f}:6:25"
    assert loc["hard.q_dyn.threshold_crossing[1]"] == f"{f}:7:9"
    assert loc["hard.q_dyn.threshold_crossing[1].value"] == f"{f}:7:9"
    assert loc["hard.q_dyn.threshold_crossing[1].edge"] == f"{f}:8:9"
    assert loc["events.E.when.rises_above"] == f"{f}:11:27"
    assert policy.source_text == LOCATED
    assert policy.source_format == "yaml"


def test_validation_error_carries_yaml_location(tmp_path):
    path = tmp_path / "flight_review.yaml"
    path.write_text(LOCATED.replace("edge: rising", "edge: up"), encoding="utf-8")
    with pytest.raises(PolicyError) as info:
        load_policy(str(path))
    assert str(info.value) == (
        f"hard.q_dyn.threshold_crossing[0].edge: expected one of rising, falling, both, got 'up' ({path}:6:25)"
    )


def test_missing_parameter_uses_parent_location():
    text = "version: 1\nhard:\n  aoa:\n    local_extrema: {}\n"
    with pytest.raises(PolicyError) as info:
        load_policy(text, format="yaml")
    issue = info.value.issues[0]
    assert issue.path == "hard.aoa.local_extrema"
    assert issue.location == "<string>:4:5"


def test_yml_and_json_files_and_str_paths(tmp_path):
    data = {"version": 1, "name": "x", "hard": {"q": {"global_extrema": {}}}}
    yml = tmp_path / "p.yml"
    yml.write_text("version: 1\nname: x\nhard:\n  q:\n    global_extrema: {}\n", encoding="utf-8")
    js = tmp_path / "p.JSON"
    js.write_text(json.dumps(data), encoding="utf-8")
    a = load_policy(yml)
    b = load_policy(str(js))
    c = load_policy(data)
    assert a.source_format == "yaml"
    assert b.source_format == "json"
    assert b.locations == {}
    assert b.source_text == json.dumps(data)
    assert c.source_format == "dict"
    assert c.source_text is None
    assert c.locations == {}
    assert a.sha256 == b.sha256 == c.sha256


def test_raw_text_with_format():
    y = load_policy("version: 1\nname: t\n", format="yaml")
    j = load_policy('{"version": 1, "name": "t"}', format="json")
    yml = load_policy("version: 1\nname: t\n", format="yml")
    assert y.sha256 == j.sha256 == yml.sha256
    assert j.source_text == '{"version": 1, "name": "t"}'


def test_format_overrides_extension(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text('{"version": 1}', encoding="utf-8")
    policy = load_policy(path, format="json")
    assert policy.source_format == "json"


def test_mapping_sources():
    policy = load_policy(MappingProxyType({"version": 1}))
    assert policy.source_format == "dict"
    assert load_policy({"version": 1}, format="dict").sha256 == policy.sha256


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"source": {"version": 1}, "format": "json"}, "does not apply to a mapping"),
        ({"source": "version: 1", "format": "toml"}, "unknown policy format 'toml'"),
        ({"source": "version: 1", "format": "dict"}, "needs a mapping"),
        ({"source": 42}, "got int"),
        ({"source": "policy.toml"}, "cannot tell the policy format"),
        ({"source": "version: 1\nname: x\n"}, "needs format='yaml' or format='json'"),
        ({"source": '{"version": 1}'}, "needs format='yaml' or format='json'"),
    ],
)
def test_usage_errors(kwargs, fragment):
    with pytest.raises(UsageError, match=re.escape(fragment)):
        load_policy(**kwargs)


def test_format_dict_with_path_is_usage_error(tmp_path):
    with pytest.raises(UsageError, match="needs a mapping, not a file"):
        load_policy(tmp_path / "p.yaml", format="dict")


def test_missing_file_and_directory(tmp_path):
    with pytest.raises(UsageError, match="policy file not found"):
        load_policy(tmp_path / "nope.yaml")
    folder = tmp_path / "dir.yaml"
    folder.mkdir()
    with pytest.raises(UsageError, match="is a directory"):
        load_policy(folder)


def test_non_utf8_file(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"version: 1\nname: \xff\n")
    with pytest.raises(PolicyError, match="not valid UTF-8"):
        load_policy(path)


def test_invalid_yaml_syntax_has_location():
    with pytest.raises(PolicyError) as info:
        load_policy("version: 1\nhard: {q: [1, 2\n", format="yaml")
    issue = info.value.issues[0]
    assert issue.message.startswith("invalid YAML: ")
    assert re.fullmatch(r"<string>:\d+:\d+", issue.location)


def test_invalid_yaml_in_file_names_the_file(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("version: 1\n  name: x\n", encoding="utf-8")
    with pytest.raises(PolicyError) as info:
        load_policy(path)
    assert info.value.issues[0].location.startswith(f"{path}:2:")


def test_multiple_yaml_documents_rejected():
    with pytest.raises(PolicyError, match="invalid YAML"):
        load_policy("version: 1\n---\nversion: 1\n", format="yaml")


def test_duplicate_yaml_keys():
    text = "version: 1\nname: a\nname: b\nartifact:\n  codec: store\n  codec: store\n"
    with pytest.raises(PolicyError) as info:
        load_policy(text, format="yaml")
    rendered = [issue.render() for issue in info.value.issues]
    assert rendered == [
        "name: duplicate key 'name' (<string>:3:1)",
        "artifact.codec: duplicate key 'codec' (<string>:6:3)",
    ]


def test_yaml_merge_keys_and_anchors():
    text = (
        "version: 1\n"
        "hard:\n"
        "  a:\n"
        "    threshold_crossing: &tc {value: 5, edge: rising}\n"
        "  b:\n"
        "    threshold_crossing:\n"
        "      <<: *tc\n"
        "      edge: falling\n"
    )
    policy = load_policy(text, format="yaml")
    edges = {req.signal: req.params["edge"] for req in policy.hard}
    assert edges == {"a": "rising", "b": "falling"}
    assert policy.locations["hard.b.threshold_crossing.edge"] == "<string>:8:7"


def test_yaml_recursive_alias_does_not_hang():
    text = "version: 1\nname: &n [*n]\n"
    with pytest.raises(PolicyError, match="name: expected a non-empty string"):
        load_policy(text, format="yaml")


def test_nested_yaml_aliases_do_not_explode():
    # One node object is shared by every use of an anchor. Walking it once per expanded path cost
    # time and memory exponential in the nesting depth for a policy of a few hundred bytes.
    lines = ["version: 1", "name: bomb", "a0: &a0 [1, 2]"]
    for i in range(1, 9):
        ref = f"*a{i - 1}"
        lines.append(f"a{i}: &a{i} [{ref}, {ref}, {ref}, {ref}]")
    text = "\n".join(lines) + "\n"
    start = time.perf_counter()
    data, locations = parse_yaml(text, "bomb.yaml")
    assert time.perf_counter() - start < 2.0
    assert len(locations) < 200  # one entry per node, not one per path through the node tree
    node = data["a8"]
    for _ in range(8):
        node = node[0]
    assert node == [1, 2]


def test_duplicate_yaml_keys_are_capped():
    text = "version: 1\n" + "".join("name: x\n" for _ in range(80))
    with pytest.raises(PolicyError) as info:
        load_policy(text, format="yaml")
    assert len(info.value.issues) == 50  # the limit docs/policy.md states for every stage
    assert all(issue.message == "duplicate key 'name'" for issue in info.value.issues)


def test_deeply_nested_yaml_is_a_policy_error():
    text = "version: 1\nname: " + "[" * 20000 + "]" * 20000 + "\n"
    with pytest.raises(PolicyError, match="nested too deeply"):
        load_policy(text, format="yaml")


def test_yaml_non_string_keys():
    with pytest.raises(PolicyError) as info:
        load_policy("version: 1\nsignals:\n  decl:\n    on: {unit: Pa}\n", format="yaml")
    assert info.value.issues[0].render() == (
        "signals.decl: names must be non-empty strings, got a boolean (<string>:3:3)"
    )


def test_empty_yaml_and_wrong_top_level():
    with pytest.raises(PolicyError, match="the policy is empty"):
        load_policy("", format="yaml")
    with pytest.raises(PolicyError, match="the policy is empty"):
        load_policy("# only a comment\n", format="yaml")
    with pytest.raises(PolicyError, match="a policy must be a mapping, got a list"):
        load_policy("- version: 1\n", format="yaml")
    with pytest.raises(PolicyError, match="a policy must be a mapping, got a list"):
        load_policy("[1]", format="json")


def test_invalid_json_has_location():
    with pytest.raises(PolicyError) as info:
        load_policy('{\n  "version": 1\n  "name": "x"\n}', format="json")
    issue = info.value.issues[0]
    assert issue.message == "invalid JSON: Expecting ',' delimiter"
    assert issue.location == "<string>:3:3"


def test_invalid_json_file_location(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"version": 1 "x": 2}', encoding="utf-8")
    with pytest.raises(PolicyError) as info:
        load_policy(path)
    assert info.value.issues[0].location == f"{path}:1:15"


def test_json_duplicate_keys_and_nan():
    with pytest.raises(PolicyError, match="duplicate key 'version'"):
        load_policy('{"version": 1, "version": 1}', format="json")
    with pytest.raises(PolicyError, match="NaN is not allowed"):
        load_policy('{"version": 1, "hard": {"q": {"violation": {"above": NaN}}}}', format="json")
    with pytest.raises(PolicyError, match="Infinity is not allowed"):
        parse_json('{"a": -Infinity}')


def test_parse_helpers_directly():
    data, locations = parse_yaml("a:\n  - 1\n", "f.yaml")
    assert data == {"a": [1]}
    assert locations == {"a": "f.yaml:1:1", "a[0]": "f.yaml:2:5"}
    assert parse_yaml("", "f.yaml") == (None, {})
    assert parse_json('{"a": [1]}') == {"a": [1]}


def test_missing_pyyaml_gives_install_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(UsageError, match=re.escape("install baslt[yaml]")):
        load_policy("version: 1\n", format="yaml")
    path = tmp_path / "p.yaml"
    path.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(UsageError, match=re.escape("baslt[yaml]")):
        load_policy(path)
    # JSON and mappings do not need PyYAML
    assert load_policy('{"version": 1}', format="json").version == 1
    assert load_policy({"version": 1}).version == 1


def test_yaml_is_imported_lazily():
    import subprocess

    code = "import sys, baslt.policy.parse; print('yaml' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_yaml_python_tags_are_rejected():
    with pytest.raises(PolicyError) as info:
        load_policy("version: 1\nname: !!python/name:os.system\n", format="yaml")
    issue = info.value.issues[0]
    assert issue.message.startswith("invalid YAML: ")
    assert re.fullmatch(r"<string>:2:\d+", issue.location)


def test_policy_error_exit_code():
    with pytest.raises(PolicyError) as info:
        load_policy({"version": 2})
    assert info.value.exit_code == 3
