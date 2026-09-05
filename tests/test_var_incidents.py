"""The incidents parser keeps VAR decisions (they were dropped until 2026-09-05)."""
from scraper.sofascore_events import _parse_incidents

SPECIMEN = {"incidents": [
    {"incidentType": "period", "text": "HT"},
    {"incidentType": "goal", "incidentClass": "regular", "time": 51, "isHome": True, "player": {"name": "Thomas Kristensen", "id": 1}},
    {"incidentType": "varDecision", "incidentClass": "goalAwarded", "time": 23, "isHome": True, "confirmed": False,
     "player": {"name": "Nicolò Zaniolo", "id": 2}},
    {"incidentType": "inGamePenalty", "incidentClass": "missed", "time": 70, "isHome": False, "player": {"name": "X", "id": 3}},
]}


def test_var_and_missed_penalty_rows_survive_the_parser():
    rows = _parse_incidents(SPECIMEN, 13980095)
    types = [r["incident_type"] for r in rows]
    assert types == ["goal", "varDecision", "inGamePenalty"]          # period marker still dropped
    var = rows[1]
    assert var["incident_class"] == "goalAwarded" and var["confirmed"] is False and var["minute"] == 23
    assert var["player_name"] == "Nicolò Zaniolo" and var["is_home"] is True
    assert "confirmed" not in rows[0]


def _var(mid, cls, minute, player, confirmed=False):
    return {"incidentType": "varDecision", "incidentClass": cls, "time": minute, "isHome": True,
            "confirmed": confirmed, "player": {"name": player, "id": 1}}


def test_backfill_resumes_past_checked_matches_and_marks_var_free_matches(tmp_path, monkeypatch):
    """Resume path: a match with a VAR row or a var_checked marker is not re-fetched;
    a match with no VAR incident gets the marker so it is never re-fetched either."""
    import pandas as pd

    from scraper import sofascore_events as se
    path = tmp_path / "match_incidents.parquet"
    pd.DataFrame([{"match_id": 1, "incident_type": "varDecision", "incident_class": "goalAwarded", "minute": 5,
                   "added_time": 0, "player_name": "A", "player_id": "1", "is_home": True, "confirmed": False},
                  {"match_id": 2, "incident_type": "var_checked", "incident_class": "", "minute": 0, "added_time": 0,
                   "player_name": "", "player_id": "", "is_home": None}]).to_parquet(path, index=False)
    monkeypatch.setattr(se, "_INCIDENTS_PATH", path)
    monkeypatch.setattr(se, "_jitter_delay", lambda *a, **k: None)
    fetched = []
    def fake_get(url, session=None):
        mid = int(url.removesuffix("/incidents").rsplit("/", 1)[-1])
        fetched.append(mid)
        return {"incidents": [_var(mid, "penaltyNotAwarded", 30, "B")] if mid == 3 else [{"incidentType": "goal", "incidentClass": "regular", "time": 10, "isHome": True, "player": {"name": "C", "id": 2}}]}
    monkeypatch.setattr(se, "_get_json", fake_get)
    n = se.backfill_var_incidents([1, 2, 3, 4], save_every=1)
    assert fetched == [3, 4] and n == 2                       # 1 and 2 skipped: already checked
    d = pd.read_parquet(path)
    assert set(d[d.match_id == 3].incident_type) == {"varDecision"}
    assert set(d[d.match_id == 4].incident_type) == {"var_checked"}   # VAR-free → marker, not re-fetched next time
    assert se.backfill_var_incidents([1, 2, 3, 4]) == 0


def test_two_var_reviews_same_minute_same_player_both_survive_the_dedup(tmp_path, monkeypatch):
    import pandas as pd

    from scraper import sofascore_events as se
    monkeypatch.setattr(se, "_INCIDENTS_PATH", tmp_path / "inc.parquet")
    rows = se._parse_incidents({"incidents": [_var(9, "penaltyAwarded", 60, "X"), _var(9, "cardUpgrade", 60, "X")]}, 9)
    se._save_incidents(rows, set())
    se._save_incidents(rows, set())                                     # idempotent on a second save
    d = pd.read_parquet(tmp_path / "inc.parquet")
    assert sorted(d.incident_class) == ["cardUpgrade", "penaltyAwarded"]
