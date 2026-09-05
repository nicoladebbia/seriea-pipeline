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
