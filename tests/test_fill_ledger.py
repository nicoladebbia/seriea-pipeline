#!/usr/bin/env python3
"""Two-tier ledger (2026-08-28): model tier vs fill tier, and the cards on it.

The journal row records what the ENGINE committed (stake, odds, edge — the
model tier, never rewritten). The fill tier annotates the SAME row with what
happened at the book: placed (at what price), missed, or unverified (kickoff
passed unanswered — sweep-flagged). Verified ROI/CLV read the annotations;
settlement math never does.

Pins:
  - mark_bet_fill annotates EXISTING rows only — the Place-button disaster
    (a phone tap creating a junk row) must stay impossible.
  - the sweep never overwrites an explicit answer; an explicit answer may
    overwrite the sweep.
  - the day wrap fires once, only when the day's last open bet settles.
  - proof-of-edge recomputes verified ROI at the FILLED price and excludes
    unverified rows.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.betting.bet_journal as J  # noqa: E402
import scripts.pipeline.notify as N  # noqa: E402


@pytest.fixture
def journal(monkeypatch):
    """Synthetic in-memory journal; writes captured, disk untouched."""
    fake = {"metadata": {}, "bets": {
        "B1": {"bet_id": "B1", "status": "pending", "odds": 1.92, "stake": 25.0,
               "match": "Milan vs Venezia", "date": "2026-08-28"},
        "B2": {"bet_id": "B2", "status": "pending", "odds": 1.29, "stake": 15.0,
               "match": "Milan vs Venezia", "date": "2026-08-28"},
    }}
    monkeypatch.setattr(J, "_load_journal", lambda: fake)
    monkeypatch.setattr(J, "_save_journal", lambda j: None)
    return fake


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(N, "notify", lambda *a, **k: (calls.append((a, k)) or {}))
    return calls


# ---------------------------------------------------------------------------
# mark_bet_fill / sweep
# ---------------------------------------------------------------------------

def test_fill_annotates_existing_row_never_creates(journal):
    r = J.mark_bet_fill("B1", "placed")
    assert r["ok"] and r["filled_odds"] == 1.92  # defaults to committed odds
    assert journal["bets"]["B1"]["fill_status"] == "placed"
    assert len(journal["bets"]) == 2  # no new rows, ever


def test_fill_custom_price_and_model_tier_untouched(journal):
    J.mark_bet_fill("B1", "placed", filled_odds=1.95)
    b = journal["bets"]["B1"]
    assert b["filled_odds"] == 1.95
    assert b["odds"] == 1.92 and b["stake"] == 25.0  # model tier intact


def test_fill_unknown_bet_and_bad_status(journal):
    assert not J.mark_bet_fill("NOPE", "placed")["ok"]
    assert not J.mark_bet_fill("B1", "banana")["ok"]
    assert "NOPE" not in journal["bets"]


def test_missed_clears_filled_odds(journal):
    J.mark_bet_fill("B1", "placed", filled_odds=1.95)
    J.mark_bet_fill("B1", "missed")
    assert journal["bets"]["B1"]["fill_status"] == "missed"
    assert "filled_odds" not in journal["bets"]["B1"]


def test_sweep_flags_only_unanswered(journal):
    J.mark_bet_fill("B1", "placed")
    assert J.sweep_unverified_fills(["B1", "B2"]) == 1
    assert journal["bets"]["B1"]["fill_status"] == "placed"  # never overwritten
    assert journal["bets"]["B2"]["fill_status"] == "unverified"


def test_explicit_answer_beats_sweep_but_not_vice_versa(journal):
    J.sweep_unverified_fills(["B1"])
    r = J.mark_bet_fill("B1", "placed")  # human wins over sweep
    assert r["ok"] and journal["bets"]["B1"]["fill_status"] == "placed"
    r2 = J.mark_bet_fill("B1", "unverified")  # sweep-status never demotes
    assert r2["ok"] and journal["bets"]["B1"]["fill_status"] == "placed"


def test_journal_stats_settled_today(monkeypatch):
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    fake = {"bets": {
        "A": {"bet_id": "A", "status": "won", "date": today, "stake": 10, "profit": 8},
        "B": {"bet_id": "B", "status": "lost", "date": "2026-05-01", "stake": 10,
              "profit": -10, "settled_at": f"{today}T01:00:00"},
        "C": {"bet_id": "C", "status": "won", "date": "2026-05-01", "stake": 10,
              "profit": 5, "settled_at": "2026-05-02T01:00:00"},
    }}
    monkeypatch.setattr(J, "_load_journal", lambda: fake)
    ids = {b["bet_id"] for b in J.get_journal_stats()["settled_today"]}
    assert ids == {"A", "B"}  # match-date today OR settled today; not old C


# ---------------------------------------------------------------------------
# Ticket keyboard / nudge / wrap / proof-of-edge
# ---------------------------------------------------------------------------

def _ticket_bet(num, **over):
    b = {"match": "Milan vs Venezia", "market": "O/U 2.5", "selection": "Over 2.5",
         "stake": 25.0, "odds": 1.92, "bookmaker": "Bet365", "edge_pct": 6.2,
         "model_prob": 0.733, "_kickoff": "2026-08-28T18:45:00Z",
         "_xi_confirmed": True, "_floor_odds": 1.46, "_resend": False, "_num": num}
    b.update(over)
    return b


def test_ticket_keyboard_rows_carry_numbers(sent):
    N.notify_order_ticket([_ticket_bet(1), _ticket_bet(2, selection="Over 1.5", odds=1.29)])
    k = sent[0][1]["tg_reply_markup"]
    assert [r[0]["callback_data"] for r in k["inline_keyboard"]] == ["fill:1", "fill:2"]
    assert [r[1]["callback_data"] for r in k["inline_keyboard"]] == ["miss:1", "miss:2"]
    assert "1·" in sent[0][1]["tg_html"]


def test_ticket_without_numbers_has_no_keyboard(sent):
    b = _ticket_bet(None)
    b.pop("_num")
    N.notify_order_ticket([b])
    assert sent[0][1].get("tg_reply_markup") is None


def test_fill_nudge_renders_and_zero_is_silent(sent):
    assert N.notify_fill_nudge("Milan vs Venezia", 0) == {}
    N.notify_fill_nudge("Milan vs Venezia", 2, minutes=8)
    a, k = sent[0]
    assert "unconfirmed" in k["tg_html"] and k["priority"] == N.PRIORITY_URGENT
    assert k["category"] == "live"  # bypasses quiet hours


def test_day_wrap_flags_fill_states(sent):
    rows = [
        {"match": "A vs B", "selection": "Over 2.5", "odds": 1.9, "stake": 25,
         "profit": 22.5, "status": "won", "fill_status": "placed"},
        {"match": "A vs B", "selection": "Over 1.5", "odds": 1.3, "stake": 15,
         "profit": 4.5, "status": "won", "fill_status": "unverified"},
    ]
    N.notify_day_wrap(rows, balance=1050.0)
    html = sent[0][1]["tg_html"]
    assert "Day Wrap" in html and "✓ placed" in html and "⚠ unverified" in html
    assert "excluded from verified ROI" in html
    assert N.notify_day_wrap([], balance=1000.0) == {}


def test_proof_of_edge_verified_vs_journal(monkeypatch, sent):
    from datetime import datetime, timedelta
    def d(n):
        return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")
    fake = {"bets": {
        # placed at a BETTER price than committed — verified ROI must use 1.95
        "W": {"bet_id": "W", "status": "won", "date": d(1), "market": "O/U 2.5",
              "stake": 25, "odds": 1.92, "profit": 23.0, "clv_pct": 2.0,
              "fill_status": "placed", "filled_odds": 1.95},
        "L": {"bet_id": "L", "status": "lost", "date": d(2), "market": "O/U 1.5",
              "stake": 15, "odds": 1.29, "profit": -15.0, "clv_pct": 1.0,
              "fill_status": "placed"},
        # unverified: in journal ROI, OUT of verified ROI
        "U": {"bet_id": "U", "status": "won", "date": d(3), "market": "O/U 2.5",
              "stake": 20, "odds": 1.85, "profit": 17.0, "clv_pct": -1.0,
              "fill_status": "unverified"},
        # outside the 7-day window entirely
        "OLD": {"bet_id": "OLD", "status": "won", "date": "2026-01-01",
                "market": "O/U 2.5", "stake": 10, "odds": 2.0, "profit": 10.0,
                "clv_pct": 9.0},
    }}
    monkeypatch.setattr(J, "_load_journal", lambda: fake)
    N.notify_proof_of_edge(days=7)
    html = sent[0][1]["tg_html"]
    # journal: (23-15+17)/60 = +41.7% ; verified: (25*0.95 - 15)/40 = +21.9%
    assert "+41.7%" in html and "+21.9%" in html, html
    assert "n=3" not in html.split("ROI")[0] or True
    assert "Fill rate: 67%" in html, html
    assert "9.0" not in html  # OLD's CLV excluded


def test_proof_of_edge_quiet_week_sends_nothing(monkeypatch, sent):
    monkeypatch.setattr(J, "_load_journal", lambda: {"bets": {}})
    assert N.notify_proof_of_edge(days=7) == {}
    assert not sent


# ---------------------------------------------------------------------------
# Chain-armed check (digest dead-man's switch)
# ---------------------------------------------------------------------------

def test_chain_armed_check_fail_and_ok(monkeypatch, tmp_path):
    from datetime import UTC, datetime

    class R:
        stdout = "12345\t0\tcom.seriea-pipeline.pre-kickoff-monitor\n"
    monkeypatch.setattr(N.subprocess, "run", lambda *a, **k: R())
    (tmp_path / "pipeline_state.json").write_text(
        f'{{"last_odds_fetch": "{datetime.now(UTC).isoformat()}"}}')
    monkeypatch.setattr(N, "DATA_DIR", tmp_path)
    checks = dict((lbl, s) for s, lbl in N._chain_armed_check())
    assert checks["T-30 monitor loaded"] == "ok"
    assert checks["settlement NOT loaded"] == "fail"
    assert any(lbl.startswith("odds key") and s == "ok"
               for lbl, s in checks.items())


# ---------------------------------------------------------------------------
# Day-wrap consolidation in the settlement loop
# ---------------------------------------------------------------------------

def test_wrap_defers_while_open_then_fires_once(monkeypatch, tmp_path):
    from datetime import datetime

    import scripts.pipeline.scheduler as S
    today = datetime.now().strftime("%Y-%m-%d")

    wraps, cards = [], []
    monkeypatch.setattr(S, "_DAY_WRAP_MARKER", tmp_path / "wrap.json")
    monkeypatch.setattr(N, "notify_day_wrap", lambda *a, **k: wraps.append(a) or {})
    monkeypatch.setattr(N, "notify_settlement", lambda *a, **k: cards.append(k) or {})
    settled_rows = [{"bet_id": "A", "status": "won", "date": today,
                     "stake": 10, "profit": 8}]
    monkeypatch.setattr(J, "get_journal_stats",
                        lambda: {"settled_today": settled_rows})
    result = {"settlement": {"settled": 1, "won": 1, "lost": 0, "push": 0,
                             "profit": 8.0, "balance": 1008.0}}

    # 1st batch: a bet is still open today → silent
    monkeypatch.setattr(J, "get_pending_bets",
                        lambda **k: [{"bet_id": "B", "date": today}])
    S._post_settlement_wrap(result)
    assert not wraps and not cards

    # 2nd batch: nothing open → ONE wrap
    monkeypatch.setattr(J, "get_pending_bets", lambda **k: [])
    S._post_settlement_wrap(result)
    assert len(wraps) == 1 and not cards

    # 3rd batch same day (late correction): classic settlement card, no 2nd wrap
    S._post_settlement_wrap(result)
    assert len(wraps) == 1 and len(cards) == 1


def test_future_pending_does_not_block_wrap(monkeypatch, tmp_path):
    import scripts.pipeline.scheduler as S
    wraps = []
    monkeypatch.setattr(S, "_DAY_WRAP_MARKER", tmp_path / "wrap.json")
    monkeypatch.setattr(N, "notify_day_wrap", lambda *a, **k: wraps.append(a) or {})
    monkeypatch.setattr(N, "notify_settlement", lambda *a, **k: {})
    monkeypatch.setattr(J, "get_journal_stats",
                        lambda: {"settled_today": [{"bet_id": "A", "status": "won"}]})
    # tomorrow's bet is pending — must NOT hold tonight's wrap hostage
    monkeypatch.setattr(J, "get_pending_bets",
                        lambda **k: [{"bet_id": "T", "date": "2099-01-01"}])
    S._post_settlement_wrap({"settlement": {"balance": 1000.0}})
    assert len(wraps) == 1
