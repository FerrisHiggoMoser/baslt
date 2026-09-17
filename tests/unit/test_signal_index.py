"""SignalIndex: the name lookup the policy binder and the requirements checker share."""

from __future__ import annotations

import pytest

from baslt.errors import PolicyError
from baslt.policy import SignalIndex, UnresolvedSignal, bind_policy, load_policy
from baslt.signals import SignalInfo

pytestmark = pytest.mark.minimal


def info(name, path=None):
    return SignalInfo(name=name, path=path or "/" + name, shape=(10,), dtype="<f8", unit=None,
                      kind="continuous", n=10, time_ref="t")


INFOS = [info("aero/q"), info("aero/alpha"), info("nav/alpha"), info("prop/thrust", path="/Logs/Thrust")]


@pytest.mark.parametrize(("ref", "expected"), [
    ("aero/q", "aero/q"),
    ("/aero/q", "aero/q"),
    ("q", "aero/q"),
    ("thrust", "prop/thrust"),
    ("Logs/Thrust", "prop/thrust"),
    ("/Logs/Thrust", "prop/thrust"),
])
def test_lookup(ref, expected):
    assert SignalIndex(INFOS).lookup(ref) == expected


def test_ambiguous_and_unknown_names():
    index = SignalIndex(INFOS)
    with pytest.raises(UnresolvedSignal, match="ambiguous signal 'alpha'; it matches aero/alpha, nav/alpha"):
        index.lookup("alpha")
    with pytest.raises(UnresolvedSignal, match=r"unknown signal 'thrus'; did you mean 'thrust'\?"):
        index.lookup("thrus")
    with pytest.raises(LookupError, match=r"^unknown signal 'zzz'$"):
        index.lookup("zzz")


def test_aliases_are_suggested():
    index = SignalIndex(INFOS, aliases={"dyn_pressure": "aero/q"})
    assert "dyn_pressure" in index.candidates()
    assert "alpha" not in index.candidates()  # not unique
    with pytest.raises(UnresolvedSignal, match="did you mean 'dyn_pressure'"):
        index.lookup("dyn_pressur")


@pytest.mark.parametrize("ref", ["alpha", "thrus", "zzz"])
def test_the_binder_reports_the_same_messages(ref):
    try:
        SignalIndex(INFOS).lookup(ref)
    except UnresolvedSignal as exc:
        expected = str(exc)
    with pytest.raises(PolicyError) as caught:
        bind_policy(load_policy({"version": 1, "hard": {ref: {"global_extrema": {}}}}), INFOS)
    assert expected in str(caught.value)
