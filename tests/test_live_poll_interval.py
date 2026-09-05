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
