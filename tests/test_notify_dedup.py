#!/usr/bin/env python3
"""Notification volume discipline — pins the 2026-08-27 cleanup.

Measured from data/notification_history.jsonl (Aug 9–27): 200 sends in 18
days, 92% machine-status noise. The mechanisms fixed, each pinned here:

1. Health issue identity was NOT number-stable: a changing count like
   "missing 18/32" minted a fresh issue key every 30-min monitor cycle and
   re-alerted each time — 77 health sends in 18 days.
2. notify_scheduler_run sent a one-line "✅ done" card for every routine
   success (~2/day forever).
3. The morning picks card surfaced selections+odds 12h before the T-30
   commit — the journal-measured −5% ROI path. Deleted outright; the T-30
   Order Ticket (sent from run_pre_kickoff at the commit moment) is now the
   one message money is placed from.
4. The loss-streak alert could never fire: it read streak keys
   get_journal_stats() never emitted. Now emitted, and the alert carries a
   computed CLV verdict instead of canned coaching.

Nothing here sends: notify() is stubbed at module level per test.

Run with: python3 -m pytest tests/test_notify_dedup.py
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.pipeline.notify as N  # noqa: E402


@pytest.fixture
def sent(monkeypatch, tmp_path):
    """Stub every outbound channel; collect would-be sends."""
    calls = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: (calls.append((a, k)) or {}))
    monkeypatch.setattr(N, "_HEALTH_STATE_PATH", tmp_path / "health_state.json")
    monkeypatch.setattr(N, "_SCHEDULER_STATE_PATH", tmp_path / "scheduler_state.json")
    return calls


def _health(issues):
    overall = "CRITICAL" if any(lv == "CRITICAL" for lv, _ in issues) else "HEALTHY"
    return {"overall_status": overall, "issues": issues, "checks": {}}


def test_first_health_run_is_silent(sent):
    N.notify_health_state_change(_health([("CRITICAL", "odds_full.json: missing 18/32 prediction matches")]))
    assert not sent


def test_count_churn_does_not_realert(sent):
    """The 77-send bug: same issue, moving number, new alert every cycle."""
    N.notify_health_state_change(_health([("CRITICAL", "odds_full.json: missing 18/32 prediction matches")]))
    N.notify_health_state_change(_health([("CRITICAL", "odds_full.json: missing 17/31 prediction matches")]))
    N.notify_health_state_change(_health([("CRITICAL", "odds_full.json: missing 3/9 prediction matches")]))
    assert not sent, f"count churn re-alerted: {sent}"


def test_genuinely_new_issue_alerts_once_and_names_itself(sent):
    N.notify_health_state_change(_health([("CRITICAL", "odds_full.json: missing 18/32 prediction matches")]))
    N.notify_health_state_change(_health([
        ("CRITICAL", "odds_full.json: missing 17/31 prediction matches"),
        ("WARNING", "Odds API key expired"),
    ]))
    assert len(sent) == 1
    macos_text = sent[0][0][0]
    assert "Odds API key expired" in macos_text, macos_text


def test_fast_resolve_is_transient_silent(sent):
    """Appear → disappear inside the transient window = flap, no alert."""
    N.notify_health_state_change(_health([("WARNING", "something flapped")]))
    N.notify_health_state_change(_health([("WARNING", "something flapped"),
                                          ("WARNING", "second thing")]))
    sent.clear()  # drop the legit "second thing" alert
    N.notify_health_state_change(_health([("WARNING", "something flapped")]))
    assert not sent, f"fast resolve announced: {sent}"


def test_scheduler_routine_success_is_silent_but_state_persists(sent, tmp_path):
    r = N.notify_scheduler_run("morning", "success", duration_sec=120)
    assert r == {} and not sent
    # The digest's Systems block still sees the run
    state = (tmp_path / "scheduler_state.json").read_text()
    assert '"morning"' in state and '"success"' in state


def test_scheduler_warn_and_fail_still_send(sent):
    N.notify_scheduler_run("weekly-data-refresh", "warn", duration_sec=200)
    N.notify_scheduler_run("evening", "fail", duration_sec=50, error="boom")
    assert len(sent) == 2


# ---------------------------------------------------------------------------
# T-30 Order Ticket — the one message money is placed from
# ---------------------------------------------------------------------------

def _ticket_bet(**over):
    b = {
        "match": "Milan vs Venezia", "market": "O/U 2.5", "selection": "Over 2.5",
        "stake": 25.0, "odds": 1.85, "bookmaker": "Bet365", "edge_pct": 6.2,
        "confidence": "HIGH", "model_prob": 0.733,
        "_kickoff": "2026-08-28T18:45:00Z", "_xi_confirmed": True,
        "_floor_odds": 1.46, "_resend": False,
    }
    b.update(over)
    return b


def test_order_ticket_carries_floor_book_and_urgency(sent):
    N.notify_order_ticket([_ticket_bet()])
    assert len(sent) == 1
    a, k = sent[0]
    html = k.get("tg_html", "")
    assert "min price" in html and "1.46" in html, html
    assert "Bet365" in html and "€25" in html, html
    assert k.get("priority") == N.PRIORITY_URGENT
    assert k.get("category") == "live"


def test_order_ticket_resend_flag(sent):
    N.notify_order_ticket([_ticket_bet(_resend=True)])
    html = sent[0][1].get("tg_html", "")
    assert "Replaces an earlier ticket" in html, html


def test_empty_order_ticket_is_silent(sent):
    assert N.notify_order_ticket([]) == {} and not sent


def test_no_action_notice_names_matches(sent):
    N.notify_no_action(["Milan vs Venezia", "Roma vs Lecce"])
    a, k = sent[0]
    assert "no edge cleared the bar" in a[0].lower() or "no bets" in a[0].lower(), a[0]
    assert "Milan vs Venezia" in a[0]


# ---------------------------------------------------------------------------
# Min acceptable price — inversion of the engine's own edge formula
# ---------------------------------------------------------------------------

def test_min_acceptable_odds_inverts_the_configured_bar():
    from scripts.betting.betting_unified import BettingConfig
    from scripts.pipeline.run_full_pipeline import _min_acceptable_odds

    rules = BettingConfig.for_league("serie_a").market_rules
    ou = rules["O/U_Over"]
    bar = float((ou.get("line_min_edge") or {}).get(2.5, ou["min_edge_pct"]))

    bet = {"market": "O/U 2.5", "selection": "Over 2.5",
           "model_prob": 0.733, "league": "serie_a"}
    floor = _min_acceptable_odds(bet)
    assert floor is not None
    # At the floor price, edge == the configured bar (to rounding)
    assert abs((0.733 * floor - 1) * 100 - bar) < 1.0, (floor, bar)


def test_min_acceptable_odds_none_without_model_prob():
    from scripts.pipeline.run_full_pipeline import _min_acceptable_odds
    assert _min_acceptable_odds({"market": "O/U 2.5", "model_prob": 0}) is None


# ---------------------------------------------------------------------------
# Loss streak — facts + CLV verdict, no canned coaching
# ---------------------------------------------------------------------------

def _losses(clv):
    return [{"match": f"M{i}", "selection": "Over 2.5", "odds": 1.8, "clv_pct": c}
            for i, c in enumerate(clv)]


def test_loss_streak_clv_intact_says_variance(sent):
    N.notify_loss_streak(5, total_loss=87.5, recent_bets=_losses([2.1, 0.8, 1.2]))
    html = sent[0][1].get("tg_html", "")
    assert "CLV intact" in html and "Variance" in html, html
    assert "Stay disciplined" not in html  # the canned coaching is gone


def test_loss_streak_clv_negative_says_stop(sent):
    N.notify_loss_streak(5, total_loss=87.5, recent_bets=_losses([-2.1, -0.8, -1.2]))
    html = sent[0][1].get("tg_html", "")
    assert "CLV negative" in html and "Stop and review" in html, html


def test_loss_streak_no_clv_no_verdict(sent):
    bets = [{"match": "M1", "selection": "Over 2.5", "odds": 1.8}]
    N.notify_loss_streak(5, total_loss=20.0, recent_bets=bets)
    html = sent[0][1].get("tg_html", "")
    assert "no verdict" in html, html


# ---------------------------------------------------------------------------
# get_journal_stats streak keys — the alert's trigger could never fire before
# ---------------------------------------------------------------------------

def test_journal_stats_emits_streak_keys(monkeypatch):
    import scripts.betting.bet_journal as J

    def row(i, status, stake=10.0):
        return {"bet_id": f"b{i}", "status": status, "stake": stake,
                "profit": stake if status == "won" else -stake,
                "settled_at": f"2026-08-2{i}T12:00:00", "market": "O/U 2.5"}

    # W W L P L  → pushes neither break nor extend: trailing decisive = L, L
    journal = {"bets": {f"b{i}": row(i, s) for i, s in
               enumerate(["won", "won", "lost", "push", "lost"])}}
    monkeypatch.setattr(J, "_load_journal", lambda: journal)
    s = J.get_journal_stats()
    assert s["current_streak"] == -2, s["current_streak"]
    assert s["streak_loss"] == 20.0
    assert len(s["recent_losses"]) == 2
    # most recent last
    assert s["recent_losses"][-1]["bet_id"] == "b4"


def test_journal_stats_win_streak_positive(monkeypatch):
    import scripts.betting.bet_journal as J
    journal = {"bets": {f"b{i}": {"bet_id": f"b{i}", "status": s, "stake": 10.0,
                                  "profit": 0, "settled_at": f"2026-08-2{i}T12:00:00"}
                        for i, s in enumerate(["lost", "won", "won", "won"])}}
    monkeypatch.setattr(J, "_load_journal", lambda: journal)
    s = J.get_journal_stats()
    assert s["current_streak"] == 3
    assert s["streak_loss"] == 0.0
    assert s["recent_losses"] == []
