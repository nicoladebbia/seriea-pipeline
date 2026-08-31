"""Live bet commentary + settlement against the REAL journal shapes.

The pre-2026-08-31 code compared market literals ('totals', 'btts') and bare
selection tokens ('yes', '1X') that NEVER matched what the journal actually
writes ('O/U 2.5', 'BTTS No', '1X (Home or Draw)', 'DNB Home', 'Away -0.2') —
so commentary fell through to 'in play' and check_bet_settlement returned None
for every real bet. These tests pin the real shapes.
"""

import pytest

from scripts.betting.live_bet_context import (
    _check_winning,
    _generate_commentary as g,
    resolve_bet_line,
)
from scripts.data.live_monitor import check_bet_settlement


# ─── The user's worked example ───────────────────────────────────────────────

def test_over_25_at_1_0_says_two_more_goals():
    """'if we bet over 2.5, and we get 1-0 then it tells me hey, 2 goals remaining'"""
    out = g("O/U 2.5", "Over 2.5", 1, 0, 55)
    assert "needs 2 more goals" in out
    assert "35 min" in out  # 90 - 55


# ─── Totals commentary ───────────────────────────────────────────────────────

def test_over_covered():
    assert g("O/U 2.5", "Over 2.5", 2, 1, 70) == "COVERED"


def test_over_integer_line_push_zone():
    out = g("O/U 2.0", "Over 2.0", 1, 1, 60)
    assert "PUSH zone" in out and "stake refunded" in out


def test_under_integer_line_push_zone():
    out = g("O/U 2.0", "Under 2.0", 1, 1, 70)
    assert "PUSH zone" in out


def test_under_busted():
    assert g("O/U 2.5", "Under 2.5", 2, 1, 50) == "BUSTED"


def test_under_on_track_margin():
    out = g("O/U 3.5", "Under 3.5", 1, 0, 60)
    assert "ON TRACK" in out


def test_over_quarter_75_half_won():
    out = g("O/U 2.75", "Over 2.75", 2, 1, 60)
    assert "HALF WON" in out


def test_line_falls_back_to_market_string():
    # Selection without a number — line must come from the market
    out = g("O/U 2.5", "Over", 1, 0, 55)
    assert "needs 2 more goals" in out


# ─── 1X2 / DC / DNB / BTTS commentary (real selection strings) ───────────────

def test_1x2_home_winning():
    assert g("1X2", "Home", 2, 0, 80) == "WINNING"


def test_dc_real_selection_shape():
    assert "LOSING" in g("DC", "1X (Home or Draw)", 0, 1, 60)
    assert g("DC", "1X (Home or Draw)", 1, 0, 60) == "ON TRACK"


def test_dnb_draw_is_push_not_generic():
    out = g("DNB", "DNB Home", 1, 1, 80)
    assert "PUSH" in out and "refunded" in out
    assert _check_winning("DNB", "DNB Home", 1, 1) is None


def test_btts_real_selection_shape():
    assert "ON TRACK" in g("BTTS", "BTTS No", 1, 0, 30)
    assert g("BTTS", "BTTS No", 1, 1, 30) == "BUSTED"
    assert g("BTTS", "BTTS Yes", 1, 1, 30) == "HIT"


# ─── AH quarter lines (journal rounds the selection: 'AH 0.25' -> 'Away -0.2') ──

def test_ah_line_magnitude_from_market_sign_from_selection():
    assert resolve_bet_line("AH 0.25", "Away -0.2") == -0.25
    assert resolve_bet_line("AH 1.75", "Home +1.8") == 1.75
    assert resolve_bet_line("AH -1.0", "Away +1.0") == 1.0
    assert resolve_bet_line("spreads", "HOME +0") == 0.0


def test_ah_quarter_half_loss_position():
    out = g("AH 0.25", "Away -0.2", 1, 1, 50)
    assert "HALF LOSS" in out


def test_spreads_level_is_push_position():
    out = g("spreads", "HOME +0", 0, 0, 40)
    assert "PUSH" in out


# ─── Player props (structural — for when SoT/passes activate) ────────────────

_PS = {"home": [{"name": "Lautaro Martínez", "shots_on_target": 1}], "away": []}


def test_player_prop_live_stat_accent_insensitive():
    out = g("Shots on Target", "Lautaro Martinez Over 1.5", 0, 0, 30, player_stats=_PS)
    assert "needs 1 more" in out and "Martínez" in out


def test_player_prop_hit():
    ps = {"home": [{"name": "Lautaro Martínez", "shots_on_target": 3}], "away": []}
    out = g("Shots on Target", "Lautaro Martinez Over 1.5", 0, 0, 30, player_stats=ps)
    assert "HIT" in out


def test_player_prop_graceful_without_stats():
    out = g("Shots on Target", "Lautaro Martinez Over 1.5", 0, 0, 30)
    assert "not tracked yet" in out


# ─── check_bet_settlement with real journal shapes ───────────────────────────

@pytest.mark.parametrize("bet,score,completed,expected", [
    ({"market": "O/U 2.5", "selection": "Over 2.5"}, (2, 1), True, "won"),
    ({"market": "O/U 2.5", "selection": "Over 2.5"}, (1, 0), True, "lost"),
    ({"market": "O/U 2.0", "selection": "Under 2.0"}, (1, 1), True, "push"),
    ({"market": "O/U 2.75", "selection": "Over 2.75"}, (2, 1), True, "half_won"),
    ({"market": "O/U 2.75", "selection": "Under 2.75"}, (2, 1), True, "half_lost"),
    ({"market": "1X2", "selection": "Home"}, (2, 0), True, "won"),
    ({"market": "DC", "selection": "1X (Home or Draw)"}, (1, 1), True, "won"),
    ({"market": "DC", "selection": "1X (Home or Draw)"}, (0, 1), True, "lost"),
    ({"market": "DNB", "selection": "DNB Home"}, (1, 1), True, "push"),
    ({"market": "DNB", "selection": "DNB Home"}, (2, 1), True, "won"),
    ({"market": "BTTS", "selection": "BTTS No"}, (1, 0), True, "won"),
    ({"market": "BTTS", "selection": "BTTS No"}, (1, 1), True, "lost"),
    ({"market": "AH 0.25", "selection": "Away -0.2"}, (1, 1), True, "half_lost"),
    ({"market": "AH -1.0", "selection": "Away +1.0"}, (0, 1), True, "won"),
    ({"market": "spreads", "selection": "HOME +0"}, (1, 1), True, "push"),
])
def test_settlement_real_shapes(bet, score, completed, expected):
    assert check_bet_settlement(bet, score[0], score[1], 90, completed) == expected


def test_settlement_over_live_states():
    bet = {"market": "O/U 2.5", "selection": "Over 2.5"}
    # 3-0 live -> virtually won
    assert check_bet_settlement(bet, 3, 0, 60, False) == "virtually_won"
    # 0-0 at 86' needing 3 -> virtually lost
    assert check_bet_settlement(bet, 0, 0, 86, False) == "virtually_lost"
    # 1-0 at 86' needing 2 -> still open
    assert check_bet_settlement(bet, 1, 0, 86, False) is None


# ─── Goal-ping gating: bet-games only ────────────────────────────────────────

def _goal_event(minute=30):
    return {"type": "goal", "minute": minute, "player": "Someone",
            "is_home": True, "goal_type": "regular"}


def _run_notifications(monkeypatch, has_bets):
    import scripts.data.live_monitor as lm
    import scripts.pipeline.notify as notify_mod

    calls = []
    monkeypatch.setattr(notify_mod, "notify_goal",
                        lambda **kw: calls.append(("goal", kw)))
    monkeypatch.setattr(notify_mod, "notify",
                        lambda *a, **kw: calls.append(("notify", a, kw)))
    ctx = {"has_bets": has_bets, "total_stake": 5.0 if has_bets else 0.0,
           "bets": ([{"market": "O/U 2.5", "selection": "Over 2.5", "odds": 1.9,
                      "stake": 5.0, "commentary": "needs 2 more goals",
                      "is_winning": None, "parlay_legs": []}] if has_bets else [])}
    monkeypatch.setattr(lm, "_get_bet_context", lambda *a, **kw: ctx)

    match_data = {"home_team": "Inter", "away_team": "Milan", "snapshots": []}
    lm._send_live_event_notifications("Inter vs Milan", match_data,
                                      old_events=[], new_events=[_goal_event()])
    return calls


def test_goal_ping_suppressed_without_bets(monkeypatch):
    assert _run_notifications(monkeypatch, has_bets=False) == []


def test_goal_ping_fires_with_bets(monkeypatch):
    calls = _run_notifications(monkeypatch, has_bets=True)
    assert any(c[0] == "goal" for c in calls)
