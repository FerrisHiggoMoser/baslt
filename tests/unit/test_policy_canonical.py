"""Canonical policy form and hash: identical across YAML, JSON and dict, stable under defaults."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from baslt.policy import canonical_json, canonical_sha256, load_policy

DOCS = Path(__file__).resolve().parents[2] / "docs" / "policy.md"

MINIMAL_SHA256 = "942049d3afbc38ead46e0b4b5e304c5ecd4f565f899bd67cf3c9d0dbe97656c2"


def doc_example() -> str:
    text = DOCS.read_text(encoding="utf-8")
    return re.search(r"## Complete example\s*```yaml\n(.*?)```", text, re.S).group(1)


def doc_example_dict() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(doc_example())


def test_yaml_json_dict_hash_identically(tmp_path):
    data = doc_example_dict()
    from_yaml_text = load_policy(doc_example(), format="yaml")
    yaml_file = tmp_path / "flight_review.yaml"
    yaml_file.write_text(doc_example(), encoding="utf-8")
    json_file = tmp_path / "flight_review.json"
    json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    hashes = {
        from_yaml_text.sha256,
        load_policy(yaml_file).sha256,
        load_policy(json_file).sha256,
        load_policy(json.dumps(data), format="json").sha256,
        load_policy(data).sha256,
    }
    assert len(hashes) == 1
    assert load_policy(json_file).canonical == from_yaml_text.canonical


def test_hash_is_sha256_of_canonical_json():
    policy = load_policy(doc_example(), format="yaml")
    text = canonical_json(policy.canonical)
    assert policy.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert policy.sha256 == canonical_sha256(policy.canonical)
    assert ": " not in text and ", " not in text.replace("65 kPa, ", "")
    assert json.loads(text) == policy.canonical


def test_canonical_json_format():
    assert canonical_json({"b": 1, "a": [1, {"d": None, "c": "µs"}]}) == '{"a":[1,{"c":"µs","d":null}],"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_minimal_policy_hash_is_pinned():
    policy = load_policy({"version": 1})
    assert policy.canonical == {
        "version": 1,
        "name": "policy",
        "artifact": {"max_size": None, "codec": "deflate", "hash": None, "soft_value_dtype": "source", "on_not_applicable": "warn"},
        "signals": {"time": None, "time_unit": "s", "on_non_monotonic": "error", "include": ["*"], "exclude": [], "decl": {}},
        "hard": {},
        "events": {},
        "trajectories": {},
        "sync_groups": {},
        "soft": [],
        "review": {"thumbnails": [], "kpis": []},
    }
    assert policy.sha256 == MINIMAL_SHA256


def test_explicit_defaults_do_not_change_hash():
    sparse = {
        "version": 1,
        "signals": {"decl": {"q_dyn": {"path": "aero/q"}}},
        "hard": {
            "q_dyn": {
                "local_extrema": {"prominence": "1 kPa"},
                "window_extrema": {"interval": "1 s"},
                "threshold_crossing": [{"value": "65 kPa"}],
                "violation": {"above": "70 kPa"},
                "global_extrema": None,
            },
        },
        "events": {"E": {"when": {"signal": "q_dyn", "rises_above": "60 kPa"}}, "F": {"when": {"signal": "q_dyn", "equals": 1}}},
        "trajectories": {"T": {"position": "pos", "max_position_error": "1 m"}},
        "soft": [{"match": "*"}],
    }
    explicit = {
        "version": 1,
        "name": "policy",
        "artifact": {"max_size": None, "codec": "deflate", "hash": None, "soft_value_dtype": "source", "on_not_applicable": "warn"},
        "signals": {
            "time": None,
            "time_unit": "s",
            "on_non_monotonic": "error",
            "include": ["*"],
            "exclude": [],
            "decl": {"q_dyn": {"path": "aero/q", "unit": None, "kind": None, "time": None}},
        },
        "hard": {
            "q_dyn": {
                "local_extrema": {"prominence": "1 kPa", "separation": "0 s", "kind": "both", "severity": "info"},
                "window_extrema": {"interval": "1 s", "origin": "0 s", "severity": "info"},
                "threshold_crossing": [
                    {
                        "value": "65 kPa",
                        "edge": "both",
                        "hysteresis": 0,
                        "debounce": "0 s",
                        "tolerance": "0 s",
                        "interpolate": "linear",
                        "severity": "info",
                    }
                ],
                "violation": {"above": "70 kPa", "below": None, "min_duration": "0 s", "severity": "limit"},
                "global_extrema": {"severity": "info"},
            },
        },
        "events": {
            "E": {
                "when": {"signal": "q_dyn", "rises_above": "60 kPa", "hysteresis": "0", "debounce": "0 s"},
                "occurrence": "first",
                "expect": None,
                "keep": {"before": "0 s", "after": "0 s", "signals": None},
                "severity": "info",
            },
            "F": {"when": {"signal": "q_dyn", "equals": 1}, "keep": {}},
        },
        "trajectories": {"T": {"position": "pos", "max_position_error": "1 m", "max_time_error": None, "linked": []}},
        "sync_groups": {},
        "soft": [{"match": "*", "priority": "medium", "max_points": None}],
        "review": {"thumbnails": ["q_dyn"], "kpis": []},
    }
    assert load_policy(sparse).sha256 == load_policy(explicit).sha256


def test_reloading_canonical_is_idempotent():
    policy = load_policy(doc_example(), format="yaml")
    again = load_policy(policy.canonical)
    assert again.canonical == policy.canonical
    assert again.sha256 == policy.sha256


def test_key_order_does_not_matter():
    data = doc_example_dict()

    def reverse(obj):
        if isinstance(obj, dict):
            return {k: reverse(obj[k]) for k in reversed(list(obj))}
        if isinstance(obj, list):
            return [reverse(v) for v in obj]
        return obj

    assert load_policy(reverse(data)).sha256 == load_policy(data).sha256
    # The docs example pins review.thumbnails; without it the default comes from the hard section.
    no_review = {key: value for key, value in data.items() if key != "review"}
    assert load_policy(reverse(no_review)).sha256 == load_policy(no_review).sha256


def test_default_thumbnails_do_not_depend_on_key_order():
    a = load_policy({"version": 1, "hard": {"a": {"global_extrema": {}}, "b": {"global_extrema": {}}}})
    b = load_policy({"version": 1, "hard": {"b": {"global_extrema": {}}, "a": {"global_extrema": {}}}})
    assert a.canonical["review"]["thumbnails"] == ["a", "b"]
    assert a.canonical == b.canonical and a.sha256 == b.sha256


def test_decl_path_default_is_filled_in_before_hashing():
    omitted = load_policy({"version": 1, "signals": {"decl": {"thrust": {"unit": "kN"}}}})
    explicit = load_policy({"version": 1, "signals": {"decl": {"thrust": {"path": "thrust", "unit": "kN"}}}})
    assert omitted.canonical["signals"]["decl"]["thrust"]["path"] == "thrust"
    assert omitted.sha256 == explicit.sha256


def test_bare_numbers_hash_the_same_in_every_format():
    # YAML 1.1 reads 1e3 as text, JSON as the number 1000.0; it is the same policy either way.
    from_yaml = load_policy("version: 1\nhard:\n  q:\n    violation: {above: 1e3}\n", format="yaml")
    from_json = load_policy('{"version": 1, "hard": {"q": {"violation": {"above": 1e3}}}}', format="json")
    assert from_yaml.canonical["hard"]["q"]["violation"]["above"] == "1000"
    assert from_yaml.sha256 == from_json.sha256
    as_dict = load_policy({"version": 1, "hard": {"q": {"violation": {"above": 1000}}}})
    assert as_dict.sha256 == from_yaml.sha256
    sizes = {load_policy({"version": 1, "artifact": {"max_size": v}}).sha256 for v in (2097152, "2097152", "2097152.0")}
    assert len(sizes) == 1


def test_quantities_keep_their_original_strings():
    policy = load_policy(
        {
            "version": 1,
            "artifact": {"max_size": "2MiB"},
            "hard": {"q": {"threshold_crossing": {"value": 65000, "debounce": "100ms", "hysteresis": 0.5}}},
        }
    )
    spec = policy.canonical["hard"]["q"]["threshold_crossing"]
    assert spec["value"] == "65000"
    assert spec["debounce"] == "100ms"
    assert spec["hysteresis"] == "0.5"
    assert policy.canonical["artifact"]["max_size"] == "2MiB"
    assert load_policy({"version": 1, "artifact": {"max_size": 2097152}}).canonical["artifact"]["max_size"] == 2097152


def test_number_and_numeric_string_quantities_hash_the_same():
    a = load_policy({"version": 1, "hard": {"q": {"violation": {"above": 5}}}})
    b = load_policy({"version": 1, "hard": {"q": {"violation": {"above": "5"}}}})
    assert a.sha256 == b.sha256


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.__setitem__("name", "other"),
        lambda d: d["artifact"].__setitem__("max_size", "3 MiB"),
        lambda d: d["hard"]["q_dyn"]["threshold_crossing"][0].__setitem__("value", "66 kPa"),
        lambda d: d["hard"]["q_dyn"].__setitem__("global_extrema", [{}]),
        lambda d: d["soft"].reverse(),
        lambda d: d["events"]["MECO"].__setitem__("severity", "limit"),
        lambda d: d["review"].__setitem__("kpis", []),
    ],
)
def test_changes_change_the_hash(change):
    data = doc_example_dict()
    base = load_policy(data).sha256
    change(data)
    assert load_policy(data).sha256 != base


def test_list_and_single_forms_have_different_ids_and_hashes():
    single = load_policy({"version": 1, "hard": {"q": {"global_extrema": {}}}})
    listed = load_policy({"version": 1, "hard": {"q": {"global_extrema": [{}]}}})
    assert single.hard[0].id == "hard.q.global_extrema"
    assert listed.hard[0].id == "hard.q.global_extrema[0]"
    assert single.canonical["hard"]["q"]["global_extrema"] == {"severity": "info"}
    assert listed.canonical["hard"]["q"]["global_extrema"] == [{"severity": "info"}]
    assert single.sha256 != listed.sha256


def test_unicode_is_not_escaped_in_canonical_json():
    policy = load_policy({"version": 1, "name": "résumé", "hard": {"q": {"window_extrema": {"interval": "500 µs"}}}})
    text = canonical_json(policy.canonical)
    assert "µs" in text and "résumé" in text
    assert policy.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_canonical_is_json_serializable_without_nan():
    policy = load_policy(doc_example(), format="yaml")
    json.dumps(policy.canonical, allow_nan=False)
