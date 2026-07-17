"""Replay the rebuilt live_sofascore against real specimens + its output oracle.

Two independent ends are pinned here:

* **Input** — real Sofascore responses in ``tests/fixtures/sofascore/`` (Serie A
  events 13981681 and 13981716). Committed, so these tests keep working during a
  ban.
* **Output** — the blocks the original (lost) module persisted to
  ``data/live/*.json``. Gitignored, so oracle-dependent tests skip on a fresh
  clone rather than fail.

The specimens are *finished* matches; the oracle is a late in-play snapshot of
the same matches. Values that legitimately moved between those two moments
(``xg``, ``touches``, …) are therefore expected to differ — the tests assert the
**mapping**, and pin the drift explicitly rather than hiding it behind a
tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data.live_sofascore import (
    _PLAYER_STAT_KEYS,
    fetch_live_data_for_matches,
    parse_incidents,
    parse_lineups,
    parse_statistics,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sofascore"
LIVE_DIR = Path(__file__).resolve().parents[1] / "data" / "live"

SPECIMEN_EVENT_ID = 13981681


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def _oracle(sofascore_id: int, field: str):
    """The stored block the original module wrote for this match, if present."""
    for path in sorted(LIVE_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for match in (payload.get("matches") or {}).values():
            if match.get("sofascore_id") == sofascore_id and match.get(field):
                return match[field]
    return None


needs_oracle = pytest.mark.skipif(
    _oracle(SPECIMEN_EVENT_ID, "live_events") is None,
    reason="oracle unavailable: data/live/*.json is gitignored",
)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


def test_events_are_newest_first():
    """Sofascore's native order, which the oracle preserves. Do not sort."""
    events = parse_incidents(_load("event_incidents.json"))
    minutes = [e["minute"] for e in events]
    assert minutes == sorted(minutes, reverse=True)


def test_period_and_ingamepenalty_are_dropped():
    """Both appear in the real response; neither has a single oracle instance."""
    raw = _load("event_incidents.json")["incidents"]
    assert {"period", "inGamePenalty"} & {i.get("incidentType") for i in raw}, (
        "specimen should still contain the types we drop"
    )
    assert {e["type"] for e in parse_incidents({"incidents": raw})} <= {
        "goal",
        "card",
        "substitution",
        "injury_time",
        "var",
    }


def test_added_time_sentinel_never_leaks():
    """`period` incidents carry addedTime=999 as 'not applicable'. The oracle's
    779 events top out at 6."""
    events = parse_incidents(_load("event_incidents.json"))
    assert all(e["added_time"] <= 6 for e in events)


def test_injury_time_has_no_side():
    """All 68 injury_time events in the oracle carry is_home=None."""
    events = parse_incidents(_load("event_incidents.json"))
    injury = [e for e in events if e["type"] == "injury_time"]
    assert injury, "specimen should contain an injuryTime incident"
    assert all(e["is_home"] is None for e in injury)
    assert all("length" in e for e in injury)


def test_var_decision_maps_from_incident_class():
    """varDecision -> var, incidentClass -> decision. Verified against the
    oracle for event 13981716 (51', penaltyAwarded, is_home=True)."""
    events = parse_incidents(_load("event_incidents_var.json"))
    var = [e for e in events if e["type"] == "var"]
    assert var, "specimen should contain a varDecision incident"
    assert var[0]["decision"] == "penaltyAwarded"
    assert var[0]["minute"] == 51
    assert var[0]["is_home"] is True


def test_own_goal_type_is_preserved():
    """live_reconciliation flips the credited side on goal_type == 'ownGoal',
    so the value must survive the mapping verbatim."""
    events = parse_incidents(
        {"incidents": [{"incidentType": "goal", "incidentClass": "ownGoal", "time": 48, "isHome": True}]}
    )
    assert events[0]["goal_type"] == "ownGoal"
    assert events[0]["is_home"] is True  # scorer's side; the reconciler flips it


def _strip(events):
    """Drop injury_time and player names before comparing to the oracle.

    Both are known, explained differences between the specimen (final) and the
    oracle (in-play snapshot):
      * Sofascore has since revised spellings — "Mutassim"/"Moatasem" Al-Musrati,
        "M'Bala"/"M'bala" Nzola.
      * The live feed had not published these injuryTime incidents at snapshot
        time (41 of 48 oracle blocks DO carry injury_time, so emitting it is
        correct — this match's snapshot simply predates them).
    """
    return [
        {k: v for k, v in e.items() if k not in ("player", "player_in", "player_out", "assist")}
        for e in events
        if e["type"] != "injury_time"
    ]


@needs_oracle
def test_events_replay_the_oracle_exactly():
    """The real assertion: modulo the two documented differences, every event
    the original wrote is reproduced — same order, same fields, same values."""
    got = parse_incidents(_load("event_incidents.json"))
    assert _strip(got) == _strip(_oracle(SPECIMEN_EVENT_ID, "live_events"))


# --------------------------------------------------------------------------
# Team statistics
# --------------------------------------------------------------------------


@needs_oracle
def test_statistics_key_set_replays_the_oracle():
    got = parse_statistics(_load("event_statistics.json"))
    assert set(got) == set(_oracle(SPECIMEN_EVENT_ID, "live_stats"))


@needs_oracle
def test_statistics_values_match_except_the_two_that_grew():
    """10 of 12 are byte-identical to the oracle. xg and accurate_passes grew
    between the snapshot and full time — the snapshot landed one pass short."""
    got = parse_statistics(_load("event_statistics.json"))
    exp = _oracle(SPECIMEN_EVENT_ID, "live_stats")
    differing = {k for k in exp if got[k] != exp[k]}
    assert differing == {"xg", "accurate_passes"}
    for key in differing:
        for side in ("home", "away"):
            assert got[key][side] >= exp[key][side], "final must not be below the snapshot"


def test_statistics_reads_the_all_period_only():
    payload = {
        "statistics": [
            {"period": "1ST", "groups": [{"statisticsItems": [{"key": "cornerKicks", "homeValue": 99, "awayValue": 99}]}]},
            {"period": "ALL", "groups": [{"statisticsItems": [{"key": "cornerKicks", "homeValue": 6, "awayValue": 5}]}]},
        ]
    }
    assert parse_statistics(payload) == {"corners": {"home": 6, "away": 5}}


# --------------------------------------------------------------------------
# Player statistics
# --------------------------------------------------------------------------


def test_assists_map_from_goalassist_not_assists():
    """There is no `assists` key in the API. Guessing it would silently write 0
    assists for every player — the oracle says Laurienté had 2."""
    assert _PLAYER_STAT_KEYS["goalAssist"] == "assists"
    assert "assists" not in _PLAYER_STAT_KEYS

    players = parse_lineups(_load("event_lineups.json"))
    laurientes = [p for p in players["home"] if p["name"].startswith("Armand")]
    assert laurientes and laurientes[0]["assists"] == 2


def test_absent_stats_are_omitted_not_zero_filled():
    """The oracle is sparse: only 6 keys appear in all 2,339 player records
    (`saves` in just 72). A 0 default would assert an unrecorded action was 0."""
    parsed = parse_lineups({"home": {"players": [{"player": {"name": "X"}, "statistics": {"touches": 3}}]}, "away": {}})
    rec = parsed["home"][0]
    assert rec["touches"] == 3
    assert "saves" not in rec
    assert "rating" not in rec


@needs_oracle
def test_no_player_key_in_the_oracle_is_unmapped():
    """Sparsity must hold per player, not just in aggregate.

    The specimen is the finished match and the oracle an in-play snapshot, so
    the final record is a **superset**: a player who does something new after
    the snapshot gains a key (Ulisses Garcia created a big chance late, so
    `big_chances_created` exists in the final and not in the oracle). Equality
    would be wrong; the real claim is that nothing the original emitted is
    missing from the rebuild.
    """
    got = parse_lineups(_load("event_lineups.json"))
    exp = _oracle(SPECIMEN_EVENT_ID, "live_player_stats")
    for g, e in zip(got["home"], exp["home"]):
        assert set(e) <= set(g), f"rebuild dropped {set(e) - set(g)} for {e['name']}"


@needs_oracle
def test_most_players_replay_the_oracle_byte_exact():
    """13 of 24 reproduce exactly. The rest differ only where the match moved on
    (touches, big_chances_created) or Sofascore revised the record (Berardi's
    listed position went M -> F after the match) — so this pins the mapping
    without pretending a finished match equals an in-play snapshot.
    """
    got = parse_lineups(_load("event_lineups.json"))
    exp = _oracle(SPECIMEN_EVENT_ID, "live_player_stats")
    exact = sum(1 for g, e in zip(got["home"], exp["home"]) if g == e)
    assert exact >= 13, f"only {exact}/24 players reproduced the oracle"


# --------------------------------------------------------------------------
# Contract at live_monitor.py:1141
# --------------------------------------------------------------------------


def test_no_match_keys_makes_no_network_call(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("resolved ids with no match keys")

    monkeypatch.setattr("scripts.data.live_sofascore.get_sofascore_match_ids", boom)
    assert fetch_live_data_for_matches([]) == {}


def test_unresolvable_matches_are_omitted_not_blanked(monkeypatch):
    """live_monitor overwrites live_events only for keys present in the result,
    so omitting a match preserves its last good data instead of wiping it."""
    monkeypatch.setattr("scripts.data.live_sofascore.get_sofascore_match_ids", lambda t: {})
    assert fetch_live_data_for_matches(["Milan vs Como"]) == {}


def test_one_bad_match_does_not_sink_the_cycle(monkeypatch):
    monkeypatch.setattr(
        "scripts.data.live_sofascore.get_sofascore_match_ids",
        lambda t: {"Milan vs Como": 1, "Roma vs Lazio": 2},
    )

    def flaky(sofascore_id):
        if sofascore_id == 1:
            raise RuntimeError("boom")
        return {"sofascore_id": 2, "events": [], "statistics": {}, "player_stats": {}, "fetched_at": "x"}

    monkeypatch.setattr("scripts.data.live_sofascore.fetch_live_data_for_match", flaky)
    out = fetch_live_data_for_matches(["Milan vs Como", "Roma vs Lazio"])
    assert list(out) == ["Roma vs Lazio"]
