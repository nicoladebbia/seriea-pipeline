"""FotMob adapter (scraper/fotmob.py) — parsers on trimmed live specimens
(Fiorentina–Torino 1-2 and Man City–Coventry 1-0, both 2026-09-05)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scraper import fotmob as fm

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


@pytest.fixture
def fio_tor() -> dict:
    return _load("fotmob_match_5749663.json")


@pytest.fixture
def ctx() -> fm.MatchContext:
    return fm.MatchContext(season="2026-2027", match_id=16285000, date="2026-09-05", round=3,
                           home_team="Fiorentina", away_team="Torino", home_score=1, away_score=2)


def test_matches_for_day_keeps_only_our_leagues_and_maps_them():
    ms = fm.matches_for_day(_load("fotmob_matches_20260905.json"))
    assert {m["league"] for m in ms} == {"serie_a", "premier_league"}
    fio = next(m for m in ms if m["home"] == "Fiorentina")
    assert fio["fotmob_id"] == 5749663 and fio["finished"] and fio["score"] == "1 - 2"
    assert fio["utc_time"].startswith("2026-09-05")


def test_find_match_uses_pipeline_names_through_the_alias_map():
    ms = fm.matches_for_day(_load("fotmob_matches_20260905.json"))
    assert fm.find_match("premier_league", "Nottingham Forest", "Tottenham", ms)["fotmob_id"] == 5795444
    assert fm.find_match("serie_a", "Fiorentina", "Torino", ms)["fotmob_id"] == 5749663
    # wrong league or unknown pairing: nothing, never a guess
    assert fm.find_match("serie_a", "Nottingham Forest", "Tottenham", ms) is None
    assert fm.find_match("serie_a", "Torino", "Fiorentina", ms) is None


def test_player_rows_skip_unused_subs_and_never_zero_fill_an_absent_stat(fio_tor, ctx):
    rows = fm.player_rows(fio_tor, ctx)
    names = {r["player_name"] for r in rows}
    assert "Duván Zapata" not in names          # no stat line = never entered
    atta = next(r for r in rows if r["player_name"] == "Arthur Atta")
    assert atta["team"] == "Fiorentina" and atta["is_home"] is True and atta["opponent"] == "Torino"
    assert atta["minutes"] == 45 and atta["total_shots"] == 4 and atta["shots_on_target"] == 2
    assert atta["accurate_passes"] == 17 and atta["total_passes"] is not None
    assert atta["source"] == "fotmob" and atta["player_id"] < 0 and atta["player_id_fotmob"] == -atta["player_id"]
    assert "carries" not in atta and "key_passes" in atta    # absent family -> absent key, never 0
    gk = next(r for r in rows if r["player_name"] == "David de Gea")
    assert gk["position"] == "G" and gk["is_starter"] is True and gk["saves"] == 2.0
    sim = next(r for r in rows if r["player_name"] == "Giovanni Simeone")
    assert sim["is_starter"] is False and sim["minutes"] == 20 and sim["position"] == "F"
    # FotMob omits a zero shot count; the player's own (empty) shot map makes it a REAL 0
    assert sim["total_shots"] == 0 and sim["shots_on_target"] == 0 and sim["goals"] == 0
    assert all(r["match_id"] == 16285000 and r["date"] == "2026-09-05" and r["season"] == "2026-2027" for r in rows)


def test_player_rows_use_the_resolver_when_it_answers(fio_tor, ctx):
    seen = []

    def resolve(team, name, fid):
        seen.append((team, name, fid))
        return 777 if name == "Arthur Atta" else None

    rows = fm.player_rows(fio_tor, ctx, resolve)
    assert next(r["player_id"] for r in rows if r["player_name"] == "Arthur Atta") == 777
    assert next(r["player_id"] for r in rows if r["player_name"] == "David de Gea") < 0
    assert ("Fiorentina", "Arthur Atta", 1246467) in seen or any(n == "Arthur Atta" for _, n, _ in seen)


def test_team_stats_rows_cover_three_periods_and_parse_fraction_strings(fio_tor, ctx):
    rows = fm.team_stats_rows(fio_tor, ctx)
    assert sorted({r["period"] for r in rows}) == ["1ST", "2ND", "ALL"] and len(rows) == 6
    fio = next(r for r in rows if r["period"] == "ALL" and r["team"] == "Fiorentina")
    tor = next(r for r in rows if r["period"] == "ALL" and r["team"] == "Torino")
    assert (fio["total_shots"], fio["shots_on_target"], tor["total_shots"], tor["shots_on_target"]) == (14, 6, 7, 4)
    assert fio["possession"] + tor["possession"] == 100
    assert fio["accurate_passes"] < fio["total_passes"]          # "289 (83%)" + "348"
    assert fio["xg"] == pytest.approx(1.67) and fio["source"] == "fotmob"
    assert fio["duel_won_pct"] + tor["duel_won_pct"] == pytest.approx(100, abs=0.2)
    assert fio["aerial_duels_pct"] is not None and fio["is_home"] is True and tor["is_home"] is False


def test_shot_rows_translate_coordinates_and_vocabulary(fio_tor, ctx):
    rows = fm.shot_rows(fio_tor, ctx)
    assert {r["shot_type"] for r in rows} >= {"goal", "block", "miss"}
    goal = next(r for r in rows if r["shot_type"] == "goal")
    assert goal["is_goal"] == 1 and goal["body_part"] == "left-foot" and goal["is_left"] == 1
    blocked = next(r for r in rows if r["shot_type"] == "block")
    assert blocked["is_goal"] == 0
    header = next(r for r in rows if r["body_part"] == "head")
    assert header["is_header"] == 1 and header["is_right"] == 0
    for r in rows:
        assert 0 <= r["shot_x"] <= 100 and 0 <= r["shot_y"] <= 100
        assert r["match_id"] == "16285000" and r["source"] == "fotmob" and 1 <= r["time"] <= 90
        assert r["distance"] == pytest.approx((r["shot_x"] ** 2 + (r["shot_y"] - 50) ** 2) ** 0.5)
    # a shot 9 m from the goal line lands inside the Sofascore box (x <= 17)
    close = min(rows, key=lambda r: r["shot_x"])
    assert close["shot_x"] <= 17


def test_shot_rows_own_goal_is_not_a_goal_for_the_shooter(fio_tor, ctx):
    md = json.loads(json.dumps(fio_tor))
    goal = next(s for s in md["content"]["shotmap"]["shots"] if s["eventType"] == "Goal")
    goal["isOwnGoal"] = True
    rows = fm.shot_rows(md, ctx)
    assert all(r["is_goal"] == 0 for r in rows if r["shot_type"] == "goal")


def test_shotmap_stats_aggregate_per_side(fio_tor, ctx):
    shots = fm.shot_rows(fio_tor, ctx)
    agg = fm.shotmap_stats_rows(shots, ctx)
    assert [r["team"] for r in agg] == ["Fiorentina", "Torino"]
    assert sum(r["shots_total"] for r in agg) == len(shots)
    assert sum(r["goals_from_shots"] for r in agg) == sum(s["is_goal"] for s in shots)
    assert all(r["source"] == "fotmob" and "home_score" not in r for r in agg)


def test_parse_match_and_is_finished(fio_tor, ctx):
    out = fm.parse_match(fio_tor, ctx)
    assert set(out) == {"player_match_stats", "match_team_stats", "all_shots_with_xg", "shotmap_stats"}
    assert all(out.values()) and fm.is_finished(fio_tor)
    md = json.loads(json.dumps(fio_tor))
    md["header"]["status"]["finished"] = False
    md["general"]["finished"] = False
    assert not fm.is_finished(md)


def test_breaker_parks_the_source_after_a_transport_failure(monkeypatch):
    calls = []

    class _Resp:
        status_code = 403
        headers = {"content-type": "application/json"}
        text = "{}"

        def json(self):
            return {}

    import types
    fake = types.SimpleNamespace(get=lambda *a, **k: (calls.append(a), _Resp())[1])
    monkeypatch.setitem(__import__("sys").modules, "curl_cffi", types.SimpleNamespace(requests=fake))
    monkeypatch.setitem(__import__("sys").modules, "curl_cffi.requests", fake)
    monkeypatch.setattr(fm, "_blocked_until", 0.0)
    assert fm.fetch_matches("2026-09-05") is None
    assert fm.blocked() and "403" in fm.blocked()
    assert fm.fetch_match_details(1) is None
    assert len(calls) == 1                     # the second call never left the process
    monkeypatch.setattr(fm, "_blocked_until", 0.0)
    assert fm.blocked() is None


def test_canonical_team_alias_and_passthrough():
    assert fm.canonical_team("Nottm Forest") == "Nottingham Forest"
    assert fm.canonical_team("Man City") == "Man City"
    assert fm.canonical_team("Coventry City") == "Coventry"
