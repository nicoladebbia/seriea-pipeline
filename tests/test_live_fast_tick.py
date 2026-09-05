"""The fast live tick: ESPN every few seconds, Odds API untouched, sources never flicker."""

from datetime import datetime, timedelta, timezone

from scripts.data import live_monitor as lm


def _espn(score=(1, 0), clock="41'", at=None):
    return {"events": [{"type": "goal", "minute": 40, "is_home": True, "player": "X"}],
            "statistics": {"shots": {"home": 5, "away": 1}}, "player_stats": {},
            "fetched": {"events": True, "statistics": True, "player_stats": False},
            "score": list(score), "clock": clock, "state": "in", "source": "espn",
            "fetched_at": (at or datetime.now(timezone.utc)).isoformat()}


def _sofa():
    return {"sofascore_id": 7, "events": [{"type": "goal", "minute": 40, "is_home": True, "player": "X (sofa)"}],
            "statistics": {"shots": {"home": 6, "away": 1}}, "player_stats": {"home": [{"name": "X"}]},
            "fetched": {"events": True, "statistics": True, "player_stats": True},
            "source": "sofascore", "fetched_at": "s"}


def test_fast_tick_writes_score_clock_stats_and_stamps(monkeypatch):
    monkeypatch.setattr(lm, "_send_live_event_notifications", lambda *a, **k: None)
    entry = {"home_team": "AS Roma", "away_team": "Atalanta BC", "status": "first_half", "live_events": []}
    lm._apply_live_data("AS Roma vs Atalanta BC", entry, _espn(), fast=True)
    assert entry["live_score"] == [1, 0] and entry["live_clock"] == "41'"
    assert entry["live_stats"]["shots"]["home"] == 5 and entry["live_source"] == "espn"
    assert entry["live_fast_at"] and "live_player_stats" not in entry  # ESPN never blanks player stats


def test_slow_sofascore_cycle_does_not_overwrite_fresh_fast_data_but_adds_player_stats(monkeypatch):
    monkeypatch.setattr(lm, "_send_live_event_notifications", lambda *a, **k: None)
    entry = {"status": "first_half"}
    lm._apply_live_data("m", entry, _espn(), fast=True)
    lm._apply_live_data("m", entry, _sofa())
    assert entry["live_events"][0]["player"] == "X"           # ESPN's, kept
    assert entry["live_stats"]["shots"]["home"] == 5          # ESPN's, kept
    assert entry["live_player_stats"] == {"home": [{"name": "X"}]}  # Sofascore's, added
    assert entry["sofascore_id"] == 7
    assert entry["live_source"] == "espn"


def test_stale_fast_data_yields_to_sofascore(monkeypatch):
    monkeypatch.setattr(lm, "_send_live_event_notifications", lambda *a, **k: None)
    entry = {"status": "first_half"}
    lm._apply_live_data("m", entry, _espn(at=datetime.now(timezone.utc) - timedelta(seconds=lm.FAST_FRESH_S + 5)), fast=True)
    lm._apply_live_data("m", entry, _sofa())
    assert entry["live_events"][0]["player"] == "X (sofa)"
    assert entry["live_source"] == "sofascore"


def test_refresh_live_fast_touches_only_matches_on_the_pitch(monkeypatch):
    monkeypatch.setattr(lm, "_send_live_event_notifications", lambda *a, **k: None)
    day = {"date": "2026-09-05", "matches": {
        "AS Roma vs Atalanta BC": {"home_team": "AS Roma", "away_team": "Atalanta BC", "status": "second_half"},
        "Brentford vs Sunderland": {"home_team": "Brentford", "away_team": "Sunderland", "status": "completed"},
    }}
    saved = {}
    monkeypatch.setattr(lm, "load_matchday", lambda d=None: day)
    monkeypatch.setattr(lm, "save_matchday", lambda m: saved.update(m))
    asked = []
    from scripts.data import live_espn
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a: asked.append((h, a)) or _espn())
    out = lm.refresh_live_fast()
    assert out == {"has_live_matches": True, "refreshed": 1, "live": 1}
    assert asked == [("AS Roma", "Atalanta BC")]
    assert saved["matches"]["AS Roma vs Atalanta BC"]["live_score"] == [1, 0]
    assert "live_score" not in saved["matches"]["Brentford vs Sunderland"]


def test_refresh_live_fast_with_nothing_live_makes_no_call_and_no_write(monkeypatch):
    day = {"date": "2026-09-05", "matches": {"A vs B": {"status": "completed"}}}
    monkeypatch.setattr(lm, "load_matchday", lambda d=None: day)
    monkeypatch.setattr(lm, "save_matchday", lambda m: (_ for _ in ()).throw(AssertionError("wrote")))
    from scripts.data import live_espn
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a: (_ for _ in ()).throw(AssertionError("called")))
    assert lm.refresh_live_fast() == {"has_live_matches": False, "refreshed": 0}


def test_espn_failure_keeps_last_good_data(monkeypatch):
    day = {"date": "2026-09-05", "matches": {"A vs B": {"home_team": "A", "away_team": "B", "status": "first_half",
                                                        "live_stats": {"shots": {"home": 9, "away": 9}}}}}
    monkeypatch.setattr(lm, "load_matchday", lambda d=None: day)
    monkeypatch.setattr(lm, "save_matchday", lambda m: (_ for _ in ()).throw(AssertionError("wrote")))
    from scripts.data import live_espn
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a: None)
    assert lm.refresh_live_fast()["refreshed"] == 0
    assert day["matches"]["A vs B"]["live_stats"]["shots"]["home"] == 9
