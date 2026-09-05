"""The /live poll interval survives a Flask restart (it is pipeline state, not process state)."""

import web.app as appmod


def _fake_state(store):
    def load_state():
        return dict(store)

    def save_state(state):
        store.clear(); store.update(state)
        return "fake"

    return load_state, save_state


def test_config_endpoint_persists_and_reload_reads_it_back(monkeypatch):
    store = {}
    load_state, save_state = _fake_state(store)
    import scripts.pipeline.pipeline_state as ps
    monkeypatch.setattr(ps, "load_state", load_state)
    monkeypatch.setattr(ps, "save_state", save_state)

    r = appmod.app.test_client().post("/api/live/config", json={"interval": 60})
    assert r.get_json()["ok"] is True and r.get_json()["interval"] == 60
    assert store["live_poll_interval"] == 60
    # a fresh process reads the choice back
    assert appmod._load_live_poll_interval() == 60


def test_no_saved_choice_falls_back_to_the_given_default(monkeypatch):
    import scripts.pipeline.pipeline_state as ps
    monkeypatch.setattr(ps, "load_state", lambda: {})
    assert appmod._load_live_poll_interval() == appmod._AUTO_POLL_DEFAULT_S
    assert appmod._load_live_poll_interval(default=120) == 120


def test_saved_choice_is_clamped_and_garbage_ignored(monkeypatch):
    import scripts.pipeline.pipeline_state as ps
    monkeypatch.setattr(ps, "load_state", lambda: {"live_poll_interval": 5})
    assert appmod._load_live_poll_interval() == 30
    monkeypatch.setattr(ps, "load_state", lambda: {"live_poll_interval": "banana"})
    assert appmod._load_live_poll_interval() == appmod._AUTO_POLL_DEFAULT_S


def test_live_window_opens_30_min_before_kickoff_and_closes_3h_after():
    """2026-09-05: the live loop armed only at boot (calendar match day) or on a
    /api/live visit — tonight's goal pings existed because a tab was open. One
    gate now decides, and the arming thread reads it every minute."""
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 9, 6, 16, 0, tzinfo=timezone.utc)

    def ko(minutes_from_now):
        return [{"match": "Bologna vs Sassuolo", "kickoff_utc": now + timedelta(minutes=minutes_from_now)}]

    assert appmod._live_window_open(now, ko(4), 0.0) is True         # T-4: arm
    assert appmod._live_window_open(now, ko(6), 0.0) is False        # T-6: not yet — the scores feed is empty before the whistle
    assert appmod._live_window_open(now, ko(-149), 0.0) is True      # long stoppage: still armed
    assert appmod._live_window_open(now, ko(-151), 0.0) is False     # long over
    assert appmod._live_window_open(now, [], 0.0) is False
    assert appmod._live_window_open(now, [{"match": "broken"}], 0.0) is False   # bad row -> closed, never a crash
    # the loop stopped for lack of live matches 100' after kickoff: that match is
    # over, the window's tail must not re-arm for it...
    k = ko(-120)
    stopped_after_100 = (now - timedelta(minutes=20)).timestamp()
    assert appmod._live_window_open(now, k, stopped_after_100) is False
    # ...but a stop BEFORE the match (empty polls at T-4..T-1) never blocks the re-arm at kickoff
    stopped_pre_kickoff = (k[0]["kickoff_utc"] - timedelta(minutes=1)).timestamp()
    assert appmod._live_window_open(now, k, stopped_pre_kickoff) is True
    # ...and a later kickoff is judged on its own clock
    later = ko(-120) + ko(2)
    assert appmod._live_window_open(now, later, stopped_after_100) is True
