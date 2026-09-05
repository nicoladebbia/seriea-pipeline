"""A prop market whose stat the grading feed cannot carry is VOID, not lost.

ESPN rosters have no tackles / key passes / crosses. A missing key reads as 0,
which is correct for Sofascore (it omits zero counts) and a false loss for
ESPN — so the source of the player rows decides.
"""
from scripts.betting.prop_tracker import _evaluate_prop_outcome

ESPN_ROW = {"name": "Manu Koné", "shots": 4, "shots_on_target": 1, "fouls_committed": 1}


def test_espn_row_voids_a_tackles_prop_instead_of_losing_it():
    out = _evaluate_prop_outcome("Tackles Over 2.5", ESPN_ROW, source="espn")
    assert out["status"] == "void" and out["actual"] is None


def test_espn_row_still_grades_the_stats_it_carries():
    assert _evaluate_prop_outcome("Shots Over 2.5", ESPN_ROW, source="espn")["status"] == "hit"
    assert _evaluate_prop_outcome("SOT Over 1.5", ESPN_ROW, source="espn")["status"] == "lost"
    assert _evaluate_prop_outcome("Fouls Over 0.5", ESPN_ROW, source="espn")["status"] == "hit"


def test_sofascore_row_with_no_tackles_key_is_a_real_zero():
    # Sofascore omits zero counts: a missing key IS zero there, so the loss stands.
    out = _evaluate_prop_outcome("Tackles Over 2.5", {"name": "X", "shots": 1}, source="sofascore")
    assert out["status"] == "lost" and out["actual"] == 0
    assert _evaluate_prop_outcome("Tackles Over 2.5", {"name": "X"})["status"] == "lost"


def test_settle_props_voids_espn_ungradable_and_grades_the_rest(tmp_path, monkeypatch):
    """End to end: the match entry's live_player_source reaches the evaluator."""
    import json
    from scripts.betting import prop_tracker as pt

    live_dir = tmp_path / "live"; live_dir.mkdir()
    up_dir = tmp_path / "upcoming"; up_dir.mkdir()
    monkeypatch.setattr(pt, "LIVE_DIR", live_dir)
    monkeypatch.setattr(pt, "UPCOMING_DIR", up_dir)
    monkeypatch.setattr(pt, "LEDGER_PATH", tmp_path / "prop_ledger.json")
    monkeypatch.setattr(pt, "_update_performance", lambda ledger: None)
    (live_dir / "2026-09-05.json").write_text(json.dumps({"matches": {"AS Roma vs Atalanta BC": {
        "status": "completed", "final_score": [1, 1], "live_player_source": "espn",
        "live_player_stats": {"home": [dict(ESPN_ROW)], "away": []}}}}))
    base = {"match": "AS Roma vs Atalanta BC", "player": "Manu Koné", "best_odds": 2.0}
    (up_dir / "player_prop_value_bets.json").write_text(json.dumps({"bets": [
        {**base, "market": "Tackles Over 2.5"}, {**base, "market": "Shots Over 2.5"}]}))
    pt.settle_props("2026-09-05")
    ledger = json.loads((tmp_path / "prop_ledger.json").read_text())
    by = {e["market"]: e["outcome"] for e in ledger}
    assert by == {"Tackles Over 2.5": "void", "Shots Over 2.5": "hit"}
