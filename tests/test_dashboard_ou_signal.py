"""The dashboard row's headline is the O/U signal — the market that is actually bet.

`_ou_signal` turns goal_predictions + totals odds + the unified bet slip into one
row-level block: which line, model vs market Over probability, and the slip's
verdict (selected / near-miss / none / gated / no odds / no model).
"""
import pytest

from web.app import _ou_signal


def _totals(line, pin_over, pin_under, n=5):
    return {
        "line": line,
        "over": pin_over + 0.05,
        "under": pin_under + 0.05,
        "bookmakers_count": n,
        "all_bookmakers": [
            {"bookmaker": "Pinnacle", "over": pin_over, "under": pin_under},
            {"bookmaker": "Bet365", "over": pin_over + 0.1, "under": pin_under},
        ],
    }


GP = {"match": "Lazio vs Milan", "over_1_5": 0.80, "over_2_5": 0.56, "expected_total_goals": 3.1}
TOTALS = [_totals(2.5, 2.10, 1.80), _totals(1.5, 1.30, 3.50)]


def test_selected_bet_wins_the_row():
    sel = {"match": "Lazio vs Milan", "market": "O/U 2.5", "selection": "Over 2.5",
           "edge_pct": 8.91, "best_odds": 2.32, "stake_amount": 8.93, "model_prob": 0.504}
    s = _ou_signal("Lazio vs Milan", "serie_a", GP, TOTALS, sel, None, True)
    assert s["line"] == 2.5
    assert s["bet"]["status"] == "selected"
    assert s["bet"]["selection"] == "Over 2.5"
    assert s["bet"]["edge_pct"] == 8.91
    assert s["bet"]["stake_amount"] == 8.93
    # market prob is the de-vigged Pinnacle price for the same line
    assert s["market_over"] == pytest.approx((1 / 2.10) / (1 / 2.10 + 1 / 1.80), abs=1e-3)
    assert s["model_over"] == 0.56


def test_near_miss_carries_the_gate_it_missed():
    nm = {"match": "Udinese vs Lazio", "market": "O/U 1.5", "selection": "Over 1.5",
          "edge_pct": 3.5, "min_edge": 5.0, "max_edge": 7.0, "gap_pp": 1.5,
          "reason": "below_min_edge", "best_odds": 1.44}
    s = _ou_signal("Udinese vs Lazio", "serie_a", GP, TOTALS, None, nm, True)
    assert s["line"] == 1.5
    assert s["bet"]["status"] == "near_miss"
    assert s["bet"]["min_edge"] == 5.0
    assert s["bet"]["reason"] == "below_min_edge"


def test_no_slip_entry_picks_the_line_with_the_largest_raw_edge():
    s = _ou_signal("Lazio vs Milan", "serie_a", GP, TOTALS, None, None, True)
    assert s["bet"]["status"] == "none"
    # 2.5: 0.56 vs ~0.46 (+10pp); 1.5: 0.80 vs ~0.73 (+7pp) → 2.5 wins
    assert s["line"] == 2.5
    assert s["raw_edge_pct"] == pytest.approx(10.0, abs=1.0)


def test_thin_market_line_is_skipped():
    thin = [_totals(2.5, 2.10, 1.80, n=2), _totals(1.5, 1.30, 3.50)]
    s = _ou_signal("Lazio vs Milan", "serie_a", GP, thin, None, None, True)
    assert s["line"] == 1.5


def test_gated_league_says_so_even_with_a_juicy_edge():
    s = _ou_signal("Arsenal vs Chelsea", "premier_league", GP, TOTALS, None, None, False)
    assert s["bet"]["status"] == "gated"
    assert s["line"] == 2.5  # still shows the model/market comparison


def test_missing_inputs_are_named_not_zeroed():
    assert _ou_signal("A vs B", "serie_a", GP, [], None, None, True)["bet"]["status"] == "no_odds"
    assert _ou_signal("A vs B", "serie_a", {}, TOTALS, None, None, True)["bet"]["status"] == "no_model"
    assert _ou_signal("A vs B", "serie_a", None, None, None, None, True)["bet"]["status"] == "no_model"


def test_slip_entry_on_an_unpriced_line_falls_back_to_avg_odds():
    # Slip says 2.5 but only the consensus price exists (no Pinnacle) → use avg over/under
    t = {"line": 2.5, "over": 2.0, "under": 1.9, "bookmakers_count": 4,
         "all_bookmakers": [{"bookmaker": "Bet365", "over": 2.0, "under": 1.9}]}
    sel = {"match": "X vs Y", "market": "O/U 2.5", "selection": "Over 2.5", "edge_pct": 7.5}
    s = _ou_signal("X vs Y", "serie_a", GP, [t], sel, None, True)
    assert s["market_over"] == pytest.approx((1 / 2.0) / (1 / 2.0 + 1 / 1.9), abs=1e-3)


def test_candidate_outranks_near_miss_but_not_a_journaled_bet():
    cand = {"match": "Lazio vs Milan", "market": "O/U 1.5", "selection": "Over 1.5",
            "edge_pct": 6.22, "best_odds": 1.41, "stake_amount": 15.75}
    nm = {"match": "Lazio vs Milan", "market": "O/U 2.5", "selection": "Over 2.5",
          "edge_pct": 3.0, "min_edge": 7.0, "reason": "below_min_edge"}
    s = _ou_signal("Lazio vs Milan", "serie_a", GP, TOTALS, None, nm, True, candidate=cand)
    assert s["bet"]["status"] == "candidate"
    assert s["line"] == 1.5
    sel = {"match": "Lazio vs Milan", "market": "O/U 2.5", "selection": "Over 2.5", "edge_pct": 8.9}
    s = _ou_signal("Lazio vs Milan", "serie_a", GP, TOTALS, sel, nm, True, candidate=cand)
    assert s["bet"]["status"] == "selected"
    assert s["line"] == 2.5
