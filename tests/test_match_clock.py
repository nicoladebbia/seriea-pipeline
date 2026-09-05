"""/api/match-clock hands the dashboard the live monitor's OWN key.

The dashboard row is keyed by the pipeline name ("Roma vs Atalanta"); the /live
card id is derived from the Odds API name ("AS Roma vs Atalanta BC"). A row that
is on the pitch links to /live#match-<slug(live_key)> — without live_key the
link would be built from the wrong name and land nowhere.
"""

from datetime import datetime, timedelta, timezone

import web.app as appmod


def _fake_loader(live_status):
    ko = (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = {
        "predictions.json": {"predictions": [{"match": "Roma vs Atalanta", "home_team": "Roma", "away_team": "Atalanta", "league": "serie_a"}]},
        "odds_full.json": {"matches": {"Roma vs Atalanta": {"commence_time": ko}}},
    }
    live = {"matches": {"AS Roma vs Atalanta BC": {
        "home_team": "AS Roma", "away_team": "Atalanta BC", "status": live_status,
        "snapshots": [{"score": [1, 0], "min": 26, "avg_odds": {}}],
        "final_score": [1, 0] if live_status == "completed" else None,
    }}}

    def fake(path, default=None):
        name = getattr(path, "name", str(path))
        if name in files:
            return files[name]
        if name.endswith(".json") and name[:4].isdigit():  # data/live/<date>.json
            return live
        return default if default is not None else {}

    return fake


def _clock(monkeypatch, status):
    monkeypatch.setattr(appmod, "_load_json", _fake_loader(status))
    r = appmod.app.test_client().get("/api/match-clock")
    assert r.status_code == 200
    return r.get_json()["matches"]["Roma vs Atalanta"]


def test_live_row_carries_the_live_monitor_key(monkeypatch):
    info = _clock(monkeypatch, "first_half")
    assert info["status"] == "live"
    assert info["clock"] == "26'"
    assert info["live_key"] == "AS Roma vs Atalanta BC"


def test_completed_row_carries_the_live_monitor_key(monkeypatch):
    info = _clock(monkeypatch, "completed")
    assert info["status"] == "completed"
    assert info["live_key"] == "AS Roma vs Atalanta BC"
