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
    UnifiedBettingEngine, _gate_aux_predictions,
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
        assert f'_gate_aux_predictions({label}_preds, _allowed, "{label}")' in src
