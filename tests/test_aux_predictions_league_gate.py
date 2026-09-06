"""Auxiliary prediction files are merged both-league files — gate them row-level.

Found 2026-08-31 via near-miss logging: goal_predictions.json held 23 rows
(10 Serie A + 10 EPL + 3 stale) with no league field, and scan_ou_market — the
ONLY enabled market — priced every one of them against the merged odds. The
per-league gate lived in load_predictions() alone, so an EPL O/U bet would have
been journaled as Serie A the moment its edge landed in band. Two layers now:
run() filters every aux list to the gated match set, and scan_ou_market skips
matches outside `pred_by_match` when it is given.
"""

import inspect

from scripts.betting.betting_unified import (
    UnifiedBettingEngine,
    _gate_aux_predictions,
)

_EPL = "Arsenal vs Chelsea"
_SA = "Inter vs Napoli"


def _odds(match):
    return {match: {"totals": [{
        "line": 1.5, "over": 1.29, "under": 3.50, "bookmakers_count": 5,
        "all_bookmakers": [
            {"bookmaker": "Pinnacle", "over": 1.28, "under": 3.60},
            {"bookmaker": "bet365", "over": 1.30, "under": 3.40},
            {"bookmaker": "Unibet", "over": 1.29, "under": 3.50},
        ],
    }]}}


def _goal_pred(match):
    # De-vigged Pinnacle over-1.5 prob ~0.7377. O/U 1.5 is shrunk 0.6, so
    # over_1_5 = 0.84 -> edge 0.6*(0.84-0.7377) ~= 6.1%, inside the [3.5, 7]
    # band a high-confidence row gets. (O/U 2.5 can't be used here: its band
    # is [7.0, 7.0] because the golden-zone -0.5 runs AFTER the first min
    # check — see the session notes, that is a config finding, not a test bug.)
    return [{"match": match, "date": "2026-09-05", "over_1_5": 0.84}]


def test_precondition_ungated_scanner_prices_an_epl_match():
    """True positive: without the gate the scanner DOES emit the EPL bet."""
    eng = UnifiedBettingEngine()
    bets = eng.scan_ou_market(_goal_pred(_EPL), _odds(_EPL), None)
    assert len(bets) == 1 and bets[0].match == _EPL, eng.near_misses


def test_scanner_skips_matches_outside_the_gated_prediction_set():
    eng = UnifiedBettingEngine()
    gated = {_SA: {"match": _SA}}  # what load_predictions() let through
    assert eng.scan_ou_market(_goal_pred(_EPL), _odds(_EPL), gated) == []
    assert eng.near_misses == []  # never priced, not a near miss
    # and the gated SA match still prices
    sa = eng.scan_ou_market(_goal_pred(_SA), _odds(_SA), gated)
    assert len(sa) == 1 and sa[0].match == _SA


def test_gate_aux_predictions_keeps_only_gated_matches(caplog):
    rows = [{"match": _SA, "x": 1}, {"match": _EPL}, {"no_match_key": 1}, "junk"]
    with caplog.at_level("WARNING"):
        kept = _gate_aux_predictions(rows, {_SA}, "goal")
    assert kept == [{"match": _SA, "x": 1}]
    assert "dropped 3/4" in caplog.text and _EPL in caplog.text


def test_gate_aux_predictions_passes_non_lists_through():
    assert _gate_aux_predictions({"matches": {}}, {_SA}, "btts") == {"matches": {}}
    assert _gate_aux_predictions(None, {_SA}, "btts") is None


def test_run_gates_every_auxiliary_list():
    src = inspect.getsource(UnifiedBettingEngine.run)
    for label in ("goal", "btts", "cards", "corners", "margin"):
        assert f'_gate_aux_predictions({label}_preds, _allowed, "{label}", league=_league)' in src


def test_row_league_stamp_is_checked_even_when_the_match_is_allowed(caplog):
    # The stamp is the row's own word; an EPL-stamped row is dropped even if a
    # same-named match slipped into the allowed set. Unstamped rows fall back
    # to the allowed set alone (files written before the stamp existed).
    rows = [{"match": _SA, "league": "serie_a"}, {"match": _SA, "league": "premier_league"},
            {"match": _EPL, "league": "premier_league"}, {"match": _SA}]
    with caplog.at_level("WARNING"):
        kept = _gate_aux_predictions(rows, {_SA, _EPL}, "goal", league="serie_a")
    assert kept == [{"match": _SA, "league": "serie_a"}, {"match": _SA}]
    assert "dropped 2/4" in caplog.text
    # no league given: allowed set only (the pre-stamp behaviour)
    assert len(_gate_aux_predictions(rows, {_SA, _EPL}, "goal")) == 4


def test_goal_and_margin_writers_stamp_league(tmp_path, monkeypatch):
    import scripts.models.handicap_model as hm
    import scripts.models.over_under_model as ou
    monkeypatch.setattr(ou, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hm, "DATA_DIR", tmp_path)
    (tmp_path / "upcoming").mkdir()
    gp = ou.GoalPrediction(match=_EPL, home_team="Arsenal", away_team="Chelsea", date="2026-09-06",
                           expected_home_goals=1.5, expected_away_goals=1.2, expected_total_goals=2.7,
                           over_0_5=0.9, over_1_5=0.75, over_2_5=0.55, over_3_5=0.3, over_4_5=0.1,
                           factors=[], confidence="MEDIUM", confidence_rank=2,
                           home_attack_strength=1.0, away_attack_strength=1.0,
                           home_defense_strength=1.0, away_defense_strength=1.0)
    ou.save_over_under_predictions([gp], [])
    import json
    rows = json.loads((tmp_path / "upcoming" / "goal_predictions.json").read_text())["predictions"]
    assert rows[0]["league"] == "premier_league"
    mp = hm.MarginPrediction(match=_SA, home_team="Inter", away_team="Napoli", date="2026-09-06",
                             expected_margin=0.4, margin_std_dev=1.4, home_rating=1.0, away_rating=0.8,
                             rating_diff=0.2, handicap_probs={}, factors=[], confidence="MEDIUM",
                             confidence_rank=2)
    hm.save_handicap_predictions([mp], [])
    rows = json.loads((tmp_path / "upcoming" / "margin_predictions.json").read_text())["predictions"]
    assert rows[0]["league"] == "serie_a"
