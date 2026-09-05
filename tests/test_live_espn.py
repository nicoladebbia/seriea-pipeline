"""ESPN live fallback — parsed against real specimens (tests/fixtures/espn/, 2026-09-05).

* scoreboard_ita1_2026-09-05.json — three Serie A fixtures, one in play.
* summary_401874939.json — Fiorentina 1-2 Torino (goals with assists, subs, HT/FT).
* summary_401874937.json — Inter vs Napoli (adds a yellow card).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data import live_espn
from scripts.data.live_espn import find_event, parse_boxscore, parse_key_events

FIXTURES = Path(__file__).parent / "fixtures" / "espn"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def board():
    return _load("scoreboard_ita1_2026-09-05.json")


@pytest.fixture
def fio_tor():
    return _load("summary_401874939.json")


@pytest.fixture
def inter_napoli():
    return _load("summary_401874937.json")


# ---------------------------------------------------------------- lookup

def test_odds_api_names_find_the_espn_event(board):
    ev = find_event("AS Roma", "Atalanta BC", board)
    assert ev and ev["id"] == "401874781"


def test_pipeline_names_find_the_same_event(board):
    assert find_event("Roma", "Atalanta", board)["id"] == "401874781"


def test_reversed_fixture_is_not_a_match(board):
    assert find_event("Atalanta BC", "AS Roma", board) is None


def test_unknown_fixture_is_none(board):
    assert find_event("Milan", "Como", board) is None


# ---------------------------------------------------------------- events

def _home_id(summary):
    comps = summary["header"]["competitions"][0]["competitors"]
    return next(c["team"]["id"] for c in comps if c["homeAway"] == "home")


def test_events_are_newest_first_and_noise_is_dropped(fio_tor):
    ev = parse_key_events(fio_tor["keyEvents"], _home_id(fio_tor))
    types = [e["type"] for e in ev]
    assert "kickoff" not in types and "start-delay" not in types
    assert ev[0] == {"type": "period", "period": "ft", "minute": 90, "added_time": 5, "is_home": None}
    assert ev[-1]["type"] == "substitution" and ev[-1]["minute"] == 24


def test_goal_carries_scorer_assist_side_and_type(fio_tor):
    goals = [e for e in parse_key_events(fio_tor["keyEvents"], _home_id(fio_tor)) if e["type"] == "goal"]
    assert [g["minute"] for g in goals] == [82, 74, 47]  # newest first, from the specimen
    first = goals[-1]
    assert first["player"] == "Mateo Pellegrino" and first["assist"] == "Radu Dragusin"
    assert first["is_home"] is True and first["goal_type"] == "regular"
    volley = goals[0]
    assert volley["player"] == "Ch\u00e9 Adams" and volley["is_home"] is False


def test_substitution_order_is_in_then_out(fio_tor):
    sub = [e for e in parse_key_events(fio_tor["keyEvents"], _home_id(fio_tor)) if e["type"] == "substitution"][-1]
    assert sub["player_in"] == "Cher Ndour"
    assert sub["player_out"] == "Christ Inao Oula\u00ef"


def test_yellow_card(inter_napoli):
    cards = [e for e in parse_key_events(inter_napoli["keyEvents"], _home_id(inter_napoli)) if e["type"] == "card"]
    assert cards and all(c["card_type"] == "yellow" for c in cards)
    assert any(c["player"] == "Antonio Vergara" and c["is_home"] is False for c in cards)


def test_half_time_marker_and_stoppage_minute(fio_tor):
    ht = [e for e in parse_key_events(fio_tor["keyEvents"], _home_id(fio_tor)) if e.get("period") == "ht"][0]
    assert (ht["minute"], ht["added_time"]) == (45, 5)


def test_minute_parser_forms():
    assert live_espn._minute({"displayValue": "45'+5'"}) == (45, 5)
    assert live_espn._minute({"displayValue": "47'"}) == (47, 0)
    assert live_espn._minute({"displayValue": "", "value": 2819.0}) == (46, 0)
    assert live_espn._minute(None) == (0, 0)


# ---------------------------------------------------------------- stats

def test_boxscore_maps_every_card_stat_with_numbers(fio_tor):
    st = parse_boxscore(fio_tor["boxscore"])
    for key in ("possession", "shots", "shots_on_target", "blocked_shots", "corners",
                "fouls", "saves", "tackles", "clearances", "yellow_cards", "red_cards"):
        assert key in st, key
        assert isinstance(st[key]["home"], (int, float)) and isinstance(st[key]["away"], (int, float)), key
    assert st["possession"]["home"] + st["possession"]["away"] in (99, 100, 101)


def test_boxscore_absent_stats_are_omitted():
    assert parse_boxscore({"teams": [{"homeAway": "home", "statistics": [{"name": "saves", "displayValue": "2"}]},
                                     {"homeAway": "away", "statistics": []}]}) == {"saves": {"home": 2, "away": None}}


# ---------------------------------------------------------------- fetch shape

def test_fetch_returns_monitor_shape_without_network(monkeypatch, board, fio_tor):
    monkeypatch.setattr(live_espn, "_scoreboard", lambda slug: board if slug == "ita.1" else None)
    monkeypatch.setattr(live_espn, "_get_json", lambda url: fio_tor if "summary" in url else None)
    out = live_espn.fetch_live_data_for_match("Fiorentina", "Torino")
    assert out["source"] == "espn" and out["espn_id"] == "401874939"
    assert out["fetched"] == {"events": True, "statistics": True, "player_stats": False}
    assert out["events"][0]["type"] == "period" and out["statistics"]["shots"]["home"] > 0
    assert live_espn.fetch_live_data_for_match("Milan", "Como") is None


# ---------------------------------------------------------------- score / clock

def test_summary_carries_score_and_clock(monkeypatch, board, fio_tor):
    monkeypatch.setattr(live_espn, "_scoreboard", lambda slug: board if slug == "ita.1" else None)
    monkeypatch.setattr(live_espn, "_get_json", lambda url: fio_tor if "summary" in url else None)
    out = live_espn.fetch_live_data_for_match("Fiorentina", "Torino")
    assert out["score"] == [1, 2] and out["clock"] == "FT" and out["state"] == "post"


def test_refusal_pauses_every_espn_call(monkeypatch):
    class Resp:
        status_code = 429
        def json(self): return {}
    monkeypatch.setattr(live_espn, "_backoff_until", 0.0)
    monkeypatch.setattr(live_espn.cffi_requests, "get", lambda *a, **k: Resp())
    assert live_espn._get_json("https://x/1") is None
    calls = []
    monkeypatch.setattr(live_espn.cffi_requests, "get", lambda *a, **k: calls.append(1))
    assert live_espn._get_json("https://x/2") is None
    assert calls == []  # paused: no request was made


# ---------------------------------------------------------------- own goals, penalties, score

def _ke(slug, team_id, players=(), clock="60'"):
    return {"type": {"type": slug}, "clock": {"displayValue": clock}, "team": {"id": team_id},
            "participants": [{"athlete": {"displayName": p}} for p in players]}


def test_own_goal_is_stored_on_the_scorers_side_so_the_opponent_is_credited():
    # Specimen: "Own Goal by Redouane Halhal, Venezia." carries team=AC Milan (home, id 1)
    ev = parse_key_events([_ke("own-goal", "1", ["Redouane Halhal"])], home_id="1")[0]
    assert ev["goal_type"] == "ownGoal" and ev["is_home"] is False  # scorer plays for the away side
    assert live_espn.score_from_events([ev]) == [1, 0]                # ...and the home side is credited


def test_penalty_scored_is_a_goal_and_missed_is_not():
    evs = parse_key_events([_ke("penalty---scored", "1", ["A"]), _ke("penalty---missed", "1", ["B"])], home_id="1")
    assert [e["type"] for e in evs] == ["goal"] and evs[0]["goal_type"] == "penalty"


def test_red_card():
    ev = parse_key_events([_ke("red-card", "2", ["João Gomes"])], home_id="1")[0]
    assert ev == {"type": "card", "minute": 60, "added_time": 0, "is_home": False, "player": "João Gomes", "card_type": "red"}


def test_score_never_trails_the_goal_events(monkeypatch, board):
    """Header still says 1-1 while keyEvents already carry the 89' goal → board shows 2-1."""
    summary = {"header": {"competitions": [{"competitors": [
                   {"homeAway": "home", "team": {"id": "1"}, "score": "1"},
                   {"homeAway": "away", "team": {"id": "2"}, "score": "1"}],
                   "status": {"type": {"detail": "89'", "state": "in"}}}]},
               "keyEvents": [_ke("goal", "2", ["X"], "47'"), _ke("goal", "1", ["Y"], "60'"), _ke("goal---header", "1", ["Z"], "89'")],
               "boxscore": {"teams": []}}
    monkeypatch.setattr(live_espn, "_scoreboard", lambda slug: board if slug == "ita.1" else None)
    monkeypatch.setattr(live_espn, "_get_json", lambda url: summary if "summary" in url else None)
    out = live_espn.fetch_live_data_for_match("AS Roma", "Atalanta BC")
    assert out["score"] == [2, 1] and out["state"] == "in"
