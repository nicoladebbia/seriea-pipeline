"""Pre-match odds on the live card and the completed-match player backfill.

Until 2026-09-05 the live monitor looked pre-match odds up by the raw Odds API
name ("AS Roma vs Atalanta BC") in a file keyed by normalised names
("Roma vs Atalanta"): pre_match_odds was {} on every match ever tracked.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.data import live_monitor as lm


def test_raw_odds_api_names_resolve_to_the_normalised_odds_full_key():
    assert lm._pre_match_key("AS Roma", "Atalanta BC") == "Roma vs Atalanta"
    pre_map = {"Roma vs Atalanta": {"home": 1.67, "draw": 3.88, "away": 5.17}}
    got = lm._pre_match_odds_for(pre_map, "AS Roma", "Atalanta BC", "2026-09-05T18:45:00Z")
    assert got["home"] == 1.67


def _write_snapshot(d, stamp_local: datetime, rows: dict):
    name = f"odds_{stamp_local.strftime('%Y%m%d_%H%M%S')}.json"
    (d / name).write_text(json.dumps({"timestamp": stamp_local.isoformat(), "matches": rows}))
    return name


@pytest.fixture
def snap_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(lm, "DATA_DIR", tmp_path)
    d = tmp_path / "odds_snapshots"
    d.mkdir()
    return d


def test_closing_line_is_the_last_snapshot_before_kickoff_never_one_after(snap_dir):
    kick = datetime(2026, 9, 5, 18, 45, tzinfo=timezone.utc)
    local = lambda dt: dt.astimezone().replace(tzinfo=None)  # noqa: E731 - snapshots stamp naive local time
    row = lambda h: {"Roma vs Atalanta": {"home": h, "draw": 3.9, "away": 5.2}}  # noqa: E731
    _write_snapshot(snap_dir, local(kick - timedelta(hours=3)), row(1.72))
    before = _write_snapshot(snap_dir, local(kick - timedelta(minutes=1)), row(1.67))
    _write_snapshot(snap_dir, local(kick + timedelta(minutes=12)), row(1.20))  # in-play, must lose
    got = lm._closing_line_from_snapshots("AS Roma", "Atalanta BC", "2026-09-05T18:45:00Z")
    assert got["home"] == 1.67 and got["snapshot"] == before and got["source"] == "closing_snapshot"


def test_no_snapshot_before_kickoff_means_no_pre_match_odds(snap_dir):
    kick = datetime(2026, 9, 5, 18, 45, tzinfo=timezone.utc)
    _write_snapshot(snap_dir, (kick + timedelta(minutes=5)).astimezone().replace(tzinfo=None),
                    {"Roma vs Atalanta": {"home": 1.2, "draw": 5.0, "away": 9.0}})
    assert lm._closing_line_from_snapshots("AS Roma", "Atalanta BC", "2026-09-05T18:45:00Z") == {}
    assert lm._pre_match_odds_for({}, "AS Roma", "Atalanta BC", "2026-09-05T18:45:00Z") == {}


def test_odds_full_row_wins_and_the_snapshot_fills_a_rolled_over_match(snap_dir):
    kick = datetime(2026, 9, 5, 18, 45, tzinfo=timezone.utc)
    _write_snapshot(snap_dir, (kick - timedelta(minutes=1)).astimezone().replace(tzinfo=None),
                    {"Roma vs Atalanta": {"home": 1.67, "draw": 3.88, "away": 5.17}})
    full = {"Roma vs Atalanta": {"home": 1.65, "draw": 3.8, "away": 5.0, "totals": {"line": 2.5}}}
    assert lm._pre_match_odds_for(full, "AS Roma", "Atalanta BC", "2026-09-05T18:45:00Z")["home"] == 1.65
    assert lm._pre_match_odds_for({}, "AS Roma", "Atalanta BC", "2026-09-05T18:45:00Z")["home"] == 1.67


# ------------------------------------------------------------- player backfill

def _espn(players=True):
    return {"source": "espn", "fetched_at": "2026-09-05T21:00:00+00:00",
            "events": [{"type": "goal", "minute": 89, "player": "Hermoso"}],
            "statistics": {"shots": {"home": 39, "away": 6}},
            "player_stats": {"home": [{"name": "Mile Svilar", "saves": 1}], "away": []} if players else {},
            "fetched": {"events": True, "statistics": True, "player_stats": players}}


def test_backfill_reads_once_per_completed_match_without_players_and_writes_players_only(monkeypatch):
    from scripts.data import live_espn
    calls = []
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a, date=None: calls.append((h, a)) or _espn())
    matchday = {"matches": {
        "AS Roma vs Atalanta BC": {"status": "completed", "home_team": "AS Roma", "away_team": "Atalanta BC",
                                    "live_events": [], "live_stats": {"possession": {"home": 66, "away": 34}}},
        "Inter Milan vs Napoli": {"status": "completed", "live_player_stats": {"home": [{"name": "X"}]}},
        "Genoa vs Como": {"status": "second_half"},
    }}
    assert lm.backfill_completed_players(matchday) == 1
    assert calls == [("AS Roma", "Atalanta BC")]
    roma = matchday["matches"]["AS Roma vs Atalanta BC"]
    assert roma["live_player_stats"]["home"][0]["saves"] == 1 and roma["live_player_source"] == "espn"
    assert roma["live_events"] == [] and roma["live_stats"]["possession"]["home"] == 66  # untouched: no re-ping
    assert lm.backfill_completed_players(matchday) == 0 and len(calls) == 1


def test_backfill_gives_up_after_three_empty_reads(monkeypatch):
    from scripts.data import live_espn
    calls = []
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a, date=None: calls.append(1) or _espn(players=False))
    matchday = {"matches": {"AS Roma vs Atalanta BC": {"status": "completed"}}}
    for _ in range(5):
        lm.backfill_completed_players(matchday)
    assert len(calls) == lm.PLAYER_BACKFILL_TRIES
    assert "live_player_stats" not in matchday["matches"]["AS Roma vs Atalanta BC"]


def test_fast_tick_with_no_live_match_backfills_and_saves(monkeypatch):
    from scripts.data import live_espn
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a, date=None: _espn())
    matchday = {"matches": {"AS Roma vs Atalanta BC": {"status": "completed"}}}
    saved = []
    monkeypatch.setattr(lm, "load_matchday", lambda d: matchday)
    monkeypatch.setattr(lm, "save_matchday", lambda m: saved.append(m))
    out = lm.refresh_live_fast()
    assert out["players_backfilled"] == 1 and saved and "live_player_stats" in matchday["matches"]["AS Roma vs Atalanta BC"]


def test_backfill_asks_espn_for_the_kickoff_day(monkeypatch):
    from scripts.data import live_espn
    seen = []
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a, date=None: seen.append(date) or _espn())
    matchday = {"matches": {"Genoa vs Como": {"status": "completed", "commence_time": "2026-09-04T18:45:00Z"}}}
    lm.backfill_completed_players(matchday)
    assert seen == ["20260904"]


def test_pre_match_loader_reads_the_epl_file_too(tmp_path, monkeypatch):
    monkeypatch.setattr(lm, "DATA_DIR", tmp_path)
    up = tmp_path / "upcoming"; up.mkdir()
    (up / "odds_full.json").write_text(json.dumps({"matches": {"Roma vs Atalanta": {"h2h": {"home": 1.67, "draw": 3.9, "away": 5.2}}}}))
    (up / "odds_full_premier_league.json").write_text(json.dumps({"matches": {"Newcastle vs Bournemouth": {"h2h": {"home": 1.8, "draw": 3.6, "away": 4.4}}}}))
    pre = lm._load_pre_match_odds()
    assert pre["Roma vs Atalanta"]["home"] == 1.67
    assert lm._pre_match_odds_for(pre, "Newcastle United", "Bournemouth", "2026-09-05T11:30:00Z")["home"] == 1.8
