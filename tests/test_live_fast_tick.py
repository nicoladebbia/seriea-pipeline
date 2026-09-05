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
    # A completed match that already has its players is settled; a pre-match one is not live.
    day = {"date": "2026-09-05", "matches": {"A vs B": {"status": "completed", "live_player_stats": {"home": [{"name": "x"}]}},
                                             "C vs D": {"status": "pre_match"}}}
    monkeypatch.setattr(lm, "load_matchday", lambda d=None: day)
    monkeypatch.setattr(lm, "save_matchday", lambda m: (_ for _ in ()).throw(AssertionError("wrote")))
    from scripts.data import live_espn
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a: (_ for _ in ()).throw(AssertionError("called")))
    assert lm.refresh_live_fast() == {"has_live_matches": False, "refreshed": 0, "players_backfilled": 0}


def test_espn_failure_keeps_last_good_data(monkeypatch):
    day = {"date": "2026-09-05", "matches": {"A vs B": {"home_team": "A", "away_team": "B", "status": "first_half",
                                                        "live_stats": {"shots": {"home": 9, "away": 9}}}}}
    monkeypatch.setattr(lm, "load_matchday", lambda d=None: day)
    monkeypatch.setattr(lm, "save_matchday", lambda m: (_ for _ in ()).throw(AssertionError("wrote")))
    from scripts.data import live_espn
    monkeypatch.setattr(live_espn, "fetch_live_data_for_match", lambda h, a: None)
    assert lm.refresh_live_fast()["refreshed"] == 0
    assert day["matches"]["A vs B"]["live_stats"]["shots"]["home"] == 9


def test_full_time_from_the_fast_feed_completes_the_match_now(monkeypatch):
    monkeypatch.setattr(lm, "_send_live_event_notifications", lambda *a, **k: None)
    entry = {"status": "second_half"}
    data = _espn(score=(2, 1), clock="FT"); data["state"] = "post"
    lm._apply_live_data("m", entry, data, fast=True)
    assert entry["status"] == "completed" and entry["final_score"] == [2, 1] and entry["completed_by"] == "espn"


def test_half_time_from_the_fast_feed(monkeypatch):
    monkeypatch.setattr(lm, "_send_live_event_notifications", lambda *a, **k: None)
    entry = {"status": "first_half"}
    lm._apply_live_data("m", entry, _espn(clock="HT"), fast=True)
    assert entry["status"] == "half_time"


def test_odds_poll_cannot_unfinish_a_match():
    assert lm._status_after_poll("completed", "second_half") == "completed"
    assert lm._status_after_poll("second_half", "completed") == "completed"
    assert lm._status_after_poll("first_half", "half_time") == "half_time"


# ---------------------------------------------------------------- goal pings

def _goal(minute=40, player="X"):
    return {"type": "goal", "minute": minute, "is_home": True, "player": player, "goal_type": "regular"}


def _ping_harness(monkeypatch, mode, has_bets):
    import scripts.pipeline.notify as notify
    sent = []
    monkeypatch.setattr(lm, "_goal_ping_mode", lambda: mode)
    monkeypatch.setattr(lm, "_get_bet_context", lambda *a, **k: {"has_bets": True, "bets": []} if has_bets else None)
    monkeypatch.setattr(notify, "notify_goal", lambda **k: sent.append(k))
    monkeypatch.setattr(notify, "notify", lambda *a, **k: sent.append(("notify", a, k)))
    return sent


def test_all_mode_pings_a_goal_on_a_match_with_no_bet(monkeypatch):
    sent = _ping_harness(monkeypatch, "all", has_bets=False)
    md = {"home_team": "AS Roma", "away_team": "Atalanta BC"}
    lm._send_live_event_notifications("AS Roma vs Atalanta BC", md, [], [_goal()])
    assert len(sent) == 1 and sent[0]["scorer"] == "X" and sent[0]["home_score"] == 1


def test_bets_mode_stays_silent_without_a_bet(monkeypatch):
    sent = _ping_harness(monkeypatch, "bets", has_bets=False)
    lm._send_live_event_notifications("m", {"home_team": "A", "away_team": "B"}, [], [_goal()])
    assert sent == []


def test_the_same_goal_on_the_next_tick_is_not_pinged_again(monkeypatch):
    sent = _ping_harness(monkeypatch, "all", has_bets=False)
    md = {"home_team": "A", "away_team": "B"}
    lm._send_live_event_notifications("m", md, [_goal()], [_goal()])
    assert sent == []
    lm._send_live_event_notifications("m", md, [_goal()], [_goal(), _goal(70, "Y")])
    assert len(sent) == 1 and sent[0]["scorer"] == "Y" and sent[0]["home_score"] == 2


def test_goal_ping_mode_reads_state_and_defaults_to_all(monkeypatch):
    import scripts.pipeline.pipeline_state as ps
    monkeypatch.setattr(ps, "load_state", lambda: {"live_goal_pings": "bets"})
    assert lm._goal_ping_mode() == "bets"
    monkeypatch.setattr(ps, "load_state", lambda: {"live_goal_pings": "garbage"})
    assert lm._goal_ping_mode() == "all"
    monkeypatch.setattr(ps, "load_state", lambda: {})
    assert lm._goal_ping_mode() == "all"


def test_fast_espn_roster_fills_player_stats_when_sofascore_is_absent(monkeypatch):
    monkeypatch.setattr(lm, "_send_live_event_notifications", lambda *a, **k: None)
    entry = {"status": "first_half"}
    data = _espn()
    data["player_stats"] = {"home": [{"name": "A", "shots": 2}], "away": []}
    data["fetched"]["player_stats"] = True
    lm._apply_live_data("m", entry, data, fast=True)
    assert entry["live_player_stats"]["home"][0]["shots"] == 2 and entry["live_player_source"] == "espn"


def test_fresh_sofascore_player_stats_are_not_replaced_by_the_fast_roster(monkeypatch):
    monkeypatch.setattr(lm, "_send_live_event_notifications", lambda *a, **k: None)
    entry = {"status": "first_half"}
    sofa = _sofa()
    sofa["fetched_at"] = datetime.now(timezone.utc).isoformat()
    lm._apply_live_data("m", entry, sofa)
    data = _espn()
    data["player_stats"] = {"home": [{"name": "A", "shots": 2}], "away": []}
    data["fetched"]["player_stats"] = True
    lm._apply_live_data("m", entry, data, fast=True)
    assert entry["live_player_stats"] == {"home": [{"name": "X"}]} and entry["live_player_source"] == "sofascore"
