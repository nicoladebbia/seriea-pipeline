"""Fair-odds ledger: one row per fixture, date-aware settlement, real score parsing.

Pins the three defects found 2026-08-31: (1) key mismatch appended a copy on
every dashboard hit (3,404 rows / 80 fixtures); (2) settlement joined by bare
fixture name, settling future fixtures with last season's result; (3) journal
"1-1" strings were indexed like lists, so a draw settled as HOME.
"""

import json

import pytest

import scripts.betting.fair_odds_tracker as fot


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(fot, "LEDGER_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(fot, "SUMMARY_PATH", tmp_path / "summary.json")
    return tmp_path


def _pred(match="Inter vs Napoli", date="2026-08-30", home=0.5, draw=0.2, away=0.3,
          outcome="HOME"):
    h, a = match.split(" vs ")
    return {"match": match, "home_team": h, "away_team": a, "date": date,
            "probabilities": {"home": home, "draw": draw, "away": away},
            "predicted_outcome": outcome, "confidence_level": "medium",
            "odds": {"h2h": {"home": 2.0, "draw": 3.4, "away": 3.8}}}


def test_parse_score_handles_strings_lists_dicts():
    assert fot._parse_score("1-1") == (1, 1)
    assert fot._parse_score("2:0") == (2, 0)
    assert fot._parse_score([0, 3]) == (0, 3)
    assert fot._parse_score({"home": 4, "away": 2}) == (4, 2)
    assert fot._parse_score("n/a") is None


def test_string_draw_settles_as_draw_not_home(iso):
    """The mutation that was live: "1-1" indexed as ('1', '-') -> HOME."""
    fot.record_predictions([_pred(outcome="HOME")])
    fot.settle_predictions({"Inter vs Napoli": [{"date": "2026-08-30", "score": "1-1"}]})
    row = json.load(open(iso / "ledger.json"))[0]
    assert row["settled"] is True
    assert row["actual_outcome"] == "DRAW"
    assert row["actual_score"] == [1, 1]
    assert row["prediction_correct"] is False


def test_record_is_one_row_per_fixture_and_updates_in_place(iso):
    r1 = fot.record_predictions([_pred(home=0.5, draw=0.2, away=0.3)])
    r2 = fot.record_predictions([_pred(home=0.6, draw=0.2, away=0.2)])
    r3 = fot.record_predictions([_pred(home=0.6, draw=0.2, away=0.2)])  # identical
    rows = json.load(open(iso / "ledger.json"))
    assert len(rows) == 1
    assert rows[0]["prob_home"] == 0.6
    assert (r1["recorded"], r2["updated"], r3["updated"]) == (1, 1, 0)


def test_same_name_result_from_another_round_does_not_settle(iso):
    fot.record_predictions([_pred(date="2026-08-30")])
    fot.settle_predictions({"Inter vs Napoli": [{"date": "2026-01-15", "score": [3, 0]}]})
    row = json.load(open(iso / "ledger.json"))[0]
    assert row["settled"] is False


def test_future_fixture_is_never_settled(iso):
    fot.record_predictions([_pred(date="2099-09-05")])
    fot.settle_predictions({"Inter vs Napoli": [{"date": "2099-09-05", "score": [1, 0]}]})
    assert json.load(open(iso / "ledger.json"))[0]["settled"] is False


def test_result_within_tolerance_settles(iso):
    fot.record_predictions([_pred(date="2026-08-30", outcome="AWAY")])
    fot.settle_predictions({"Inter vs Napoli": [{"date": "2026-08-31", "score": (0, 2)}]})
    row = json.load(open(iso / "ledger.json"))[0]
    assert row["settled"] is True and row["actual_outcome"] == "AWAY"
    assert row["prediction_correct"] is True


def test_rebuild_dedups_and_resettles_from_scratch(iso, monkeypatch):
    dup = {"match": "Roma vs Atalanta", "date": "2026-09-05", "prediction_date": "2026-08-20",
           "predicted_outcome": "HOME", "settled": True, "actual_outcome": "HOME",
           "actual_score": "1-1", "prediction_correct": True}          # wrongly settled future
    past = {"match": "Lecce vs Roma", "date": "2026-08-31", "prediction_date": "2026-08-30",
            "predicted_outcome": "AWAY", "settled": False}
    fot.save_json_ledger(fot.LEDGER_PATH, [dup, dict(dup), dict(dup, prediction_date="2026-08-25"), past])
    monkeypatch.setattr(fot, "_load_results", lambda: {
        "Roma vs Atalanta": [{"date": "2026-01-10", "score": "1-1"}],
        "Lecce vs Roma": [{"date": "2026-08-31", "score": "0-2"}],
    })
    out = fot.rebuild_ledger()
    rows = {r["match"]: r for r in json.load(open(iso / "ledger.json"))}
    assert out["before"] == 4 and out["after"] == 2
    assert rows["Roma vs Atalanta"]["settled"] is False      # future + wrong-round result
    assert rows["Roma vs Atalanta"]["prediction_date"] == "2026-08-25"  # latest kept
    assert rows["Lecce vs Roma"]["settled"] is True
    assert rows["Lecce vs Roma"]["actual_outcome"] == "AWAY"
    assert rows["Lecce vs Roma"]["prediction_correct"] is True
