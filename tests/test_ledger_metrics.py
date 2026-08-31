"""ledger.get_metrics() — the ONE computation every surface renders from.

Pins the definitions decided 2026-08-28 (.plans/ledger-metrics-plan.md):
ROI = profit/stake (bankroll growth is a different, separately named number);
betting day = match `date` (fixture calendar, Europe/Rome); win rate is
decisive-only; two streaks under two names; peak/lowest immutable along the
settled_at-ordered equity curve; CLV from per-bet values; fill tier recomputed
at filled_odds; initial_bankroll comes from journal metadata, and its absence
is a visible alert rather than a silent 1000.

Every test passes the journal dict directly and freezes `now` — no real files
touched except where a tmp_path is explicitly monkeypatched in.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

import scripts.betting.ledger as L

ROME = ZoneInfo("Europe/Rome")


def _bet(i, status, stake=10.0, odds=2.0, date="2026-08-20", settled_at=None,
         market="O/U 2.5", league="serie_a", clv=None, **extra):
    profit = {"won": round(stake * (odds - 1), 2), "lost": -stake,
              "push": 0.0, "voided": 0.0, "pending": None}[status]
    row = {"bet_id": f"b{i}", "match": f"M{i}", "status": status, "stake": stake,
           "odds": odds, "profit": profit, "date": date, "market": market,
           "league": league, "selection": "Over 2.5",
           "settled_at": settled_at or (f"{date}T20:00:00" if status != "pending" else None),
           "match_kickoff_at": f"{date}T18:45:00"}
    if clv is not None:
        row["clv_pct"] = clv
    row.update(extra)
    return row


def _journal(rows, initial=1000.0):
    md = {"version": 1}
    if initial is not None:
        md["initial_bankroll"] = initial
    return {"metadata": md, "bets": {r["bet_id"]: r for r in rows}}


NOW = datetime(2026, 8, 28, 22, 0, tzinfo=UTC)  # 2026-08-29 00:00 Rome


@pytest.fixture
def no_external_alerts(monkeypatch, tmp_path):
    """Point every external alert source at an empty tmp dir."""
    monkeypatch.setattr(L, "HEALTH_STATUS_PATH", tmp_path / "health.json")
    monkeypatch.setattr(L, "CANDIDATES_PATH", tmp_path / "cand.json")
    monkeypatch.setattr(L, "T30_MARKER_PATH", tmp_path / "t30.json")
    monkeypatch.setattr(L, "_risk_gate_alerts", lambda metrics: [])


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

def test_roi_is_on_stake_and_growth_is_named_separately(no_external_alerts):
    # +10 on 20 staked; bankroll 1000 -> 1010
    j = _journal([_bet(1, "won", stake=10, odds=2.0), _bet(2, "lost", stake=10)])
    m = L.get_metrics(journal=j, now=NOW)
    assert m["roi"]["all_time_pct"] == 0.0  # +10 -10 = 0 profit / 20 staked
    j = _journal([_bet(1, "won", stake=10, odds=3.0), _bet(2, "lost", stake=10)])
    m = L.get_metrics(journal=j, now=NOW)
    assert m["roi"]["all_time_pct"] == 50.0           # +20 -10 = 10 / 20
    assert m["bankroll"]["bankroll_growth_pct"] == 1.0  # 10 / 1000
    assert m["roi"]["all_time_n"] == 2
    assert "roi" not in {k.lower() for k in m["bankroll"]}  # growth is never called ROI


def test_equity_curve_peak_lowest_and_both_drawdowns(no_external_alerts):
    # 1000 -> 1100 -> 1000 -> 900 -> 1050 : peak 1100, lowest 900,
    # max drawdown (1100-900)/1100 = 18.18, current (1100-1050)/1100 = 4.55
    rows = [_bet(1, "won", stake=100, odds=2.0, settled_at="2026-08-20T20:00:00"),
            _bet(2, "lost", stake=100, settled_at="2026-08-21T20:00:00"),
            _bet(3, "lost", stake=100, settled_at="2026-08-22T20:00:00"),
            _bet(4, "won", stake=150, odds=2.0, settled_at="2026-08-23T20:00:00")]
    m = L.get_metrics(journal=_journal(rows), now=NOW)
    b = m["bankroll"]
    assert (b["current"], b["peak"], b["lowest"]) == (1050.0, 1100.0, 900.0)
    assert b["drawdown_pct"] == 4.55
    assert b["max_drawdown_pct"] == 18.18


def test_win_rate_is_decisive_only_but_pushes_count_in_turnover(no_external_alerts):
    rows = [_bet(1, "won", odds=2.0), _bet(2, "lost"), _bet(3, "push"), _bet(4, "voided")]
    m = L.get_metrics(journal=_journal(rows), now=NOW)
    r = m["record"]
    assert (r["won"], r["lost"], r["push"], r["voided"], r["settled_n"]) == (1, 1, 1, 1, 4)
    assert r["win_rate_decisive"] == 50.0
    assert m["roi"]["all_time_n"] == 4 and m["roi"]["all_time_pct"] == 0.0  # 0 / 40 staked


def test_two_streaks_two_names(no_external_alerts):
    # chronological: W W L P L  -> decisive streak -2 ; non-win run 3 (L P L)
    seq = ["won", "won", "lost", "push", "lost"]
    rows = [_bet(i, s, settled_at=f"2026-08-2{i}T20:00:00") for i, s in enumerate(seq)]
    m = L.get_metrics(journal=_journal(rows), now=NOW)
    assert m["streak"]["streak_decisive"] == -2
    assert m["streak"]["non_win_run"] == 3
    assert m["streak"]["streak_loss_eur"] == 20.0


def test_rolling_window_reports_n_when_short(no_external_alerts):
    rows = [_bet(i, "won", odds=2.0, settled_at=f"2026-08-{10+i:02d}T20:00:00") for i in range(7)]
    m = L.get_metrics(journal=_journal(rows), now=NOW, rolling_window=50)
    assert m["roi"]["rolling_n"] == 7 and m["roi"]["rolling_window"] == 50
    assert m["roi"]["rolling_pct"] == 100.0
    # window binds: last 3 of W W W L L -> L L W? make it explicit
    rows = [_bet(i, s, odds=2.0, settled_at=f"2026-08-{10+i:02d}T20:00:00")
            for i, s in enumerate(["won", "won", "won", "lost", "lost"])]
    m = L.get_metrics(journal=_journal(rows), now=NOW, rolling_window=3)
    assert m["roi"]["rolling_n"] == 3
    assert m["roi"]["rolling_pct"] == round((10 - 10 - 10) / 30 * 100, 2)


# ---------------------------------------------------------------------------
# Betting day = match date (Europe/Rome fixture calendar)
# ---------------------------------------------------------------------------

def test_late_kickoff_settled_after_midnight_belongs_to_match_date(no_external_alerts):
    # Match Sat 2026-08-22 (22:45 Rome), settled 2026-08-23T00:50 -> belongs to 08-22
    rows = [_bet(1, "won", odds=2.0, date="2026-08-22", settled_at="2026-08-23T00:50:00")]
    now = datetime(2026, 8, 23, 8, 0, tzinfo=ROME)
    m = L.get_metrics(journal=_journal(rows), now=now)
    p = m["periods"]
    assert p["today"]["date"] == "2026-08-23" and p["today"]["n"] == 0
    assert p["last_betting_day"]["date"] == "2026-08-22"
    assert p["last_betting_day"]["pnl"] == 10.0
    days = [d["date"] for d in p["last_7_days"]["per_day"]]
    assert days == [f"2026-08-{d:02d}" for d in range(17, 24)]  # 7 calendar days ending today
    assert p["last_7_days"]["pnl"] == 10.0 and p["last_7_days"]["n"] == 1
    assert m["meta"]["day_tz"] == "Europe/Rome"


def test_today_is_rome_not_utc(no_external_alerts):
    # 22:30 UTC on 08-28 is 00:30 on 08-29 in Rome
    m = L.get_metrics(journal=_journal([]), now=datetime(2026, 8, 28, 22, 30, tzinfo=UTC))
    assert m["periods"]["today"]["date"] == "2026-08-29"


# ---------------------------------------------------------------------------
# CLV and fill tier
# ---------------------------------------------------------------------------

def test_clv_from_per_bet_values_only(no_external_alerts):
    rows = [_bet(1, "won", odds=2.0, clv=4.0), _bet(2, "lost", clv=-2.0), _bet(3, "lost")]
    m = L.get_metrics(journal=_journal(rows), now=NOW)
    assert m["clv"]["n"] == 2 and m["clv"]["avg_pct"] == 1.0 and m["clv"]["positive_rate"] == 50.0


def test_fill_tier_recomputes_at_filled_odds(no_external_alerts):
    rows = [_bet(1, "won", stake=10, odds=2.0, fill_status="placed", filled_odds=2.20),
            _bet(2, "lost", stake=10, fill_status="placed", filled_odds=1.90),
            _bet(3, "won", stake=10, odds=2.0, fill_status="missed"),
            _bet(4, "won", stake=10, odds=2.0)]  # no fill annotation
    m = L.get_metrics(journal=_journal(rows), now=NOW)
    f = m["fill"]
    # placed: +12 -10 = +2 on 20 staked
    assert f["verified_n"] == 2 and f["verified_roi_pct"] == 10.0
    assert f["fill_rate"] == round(2 / 3 * 100, 2)  # of annotated rows
    assert f["unverified_n"] == 0
    assert m["roi"]["all_time_pct"] == 50.0  # model tier untouched: +10-10+10+10 = 20 / 40


# ---------------------------------------------------------------------------
# State transition: settle one more bet -> everything moves together
# ---------------------------------------------------------------------------

def test_settling_one_more_bet_moves_every_surface_consistently(no_external_alerts):
    rows = [_bet(i, "won", odds=2.0, date=f"2026-08-{20+i}", settled_at=f"2026-08-{20+i}T20:00:00")
            for i in range(3)]
    before = L.get_metrics(journal=_journal(rows), now=NOW)
    rows.append(_bet(9, "lost", stake=30, date="2026-08-27", settled_at="2026-08-27T21:00:00"))
    after = L.get_metrics(journal=_journal(rows), now=NOW)
    assert after["bankroll"]["current"] == before["bankroll"]["current"] - 30
    assert after["bankroll"]["peak"] == before["bankroll"]["peak"]  # peak immutable
    assert after["roi"]["all_time_n"] == before["roi"]["all_time_n"] + 1
    assert after["streak"]["streak_decisive"] == -1 and before["streak"]["streak_decisive"] == 3
    assert after["periods"]["last_betting_day"]["date"] == "2026-08-27"
    assert after["periods"]["last_betting_day"]["pnl"] == -30.0
    assert after["record"]["lost"] == before["record"]["lost"] + 1
    assert after["bankroll"]["drawdown_pct"] > before["bankroll"]["drawdown_pct"]


# ---------------------------------------------------------------------------
# initial_bankroll: metadata, not a literal
# ---------------------------------------------------------------------------

def test_missing_initial_bankroll_is_a_visible_alert(no_external_alerts):
    m = L.get_metrics(journal=_journal([_bet(1, "won", odds=2.0)], initial=None), now=NOW)
    assert any("initial_bankroll" in a["message"] for a in m["alerts"])
    assert m["bankroll"]["initial"] == 1000.0  # documented default, but never silent


def test_ensure_initial_bankroll_metadata_writes_once(monkeypatch, tmp_path):
    jp = tmp_path / "bet_journal.json"
    jp.write_text(json.dumps(_journal([_bet(1, "won", odds=2.0)], initial=None)))
    monkeypatch.setattr(L, "JOURNAL_PATH", jp)
    import scripts.betting.bet_journal as J
    monkeypatch.setattr(J, "_JOURNAL_LOCK_PATH", tmp_path / ".lock")
    assert L.ensure_initial_bankroll_metadata(default=1000.0) is True
    assert json.loads(jp.read_text())["metadata"]["initial_bankroll"] == 1000.0
    assert L.ensure_initial_bankroll_metadata(default=999.0) is False  # already set, untouched
    assert json.loads(jp.read_text())["metadata"]["initial_bankroll"] == 1000.0


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def test_settlement_lag_alert_for_pending_past_full_time(no_external_alerts):
    rows = [_bet(1, "pending", date="2026-08-28", match_kickoff_at="2026-08-28T16:00:00+00:00"),
            _bet(2, "pending", date="2026-08-29", match_kickoff_at="2026-08-29T16:00:00+00:00")]
    m = L.get_metrics(journal=_journal(rows), now=NOW)  # 22:00Z: bet1 is FT+4h, bet2 tomorrow
    lag = [a for a in m["alerts"] if a["source"] == "settlement_lag"]
    assert len(lag) == 1 and "M1" in lag[0]["message"]
    assert m["bankroll"]["pending_n"] == 2 and m["bankroll"]["pending_stakes"] == 20.0
    assert m["bankroll"]["available"] == 980.0


def test_t30_funnel_alert_when_candidates_but_no_tickets(monkeypatch, tmp_path, no_external_alerts):
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"generated_at": "2026-08-28T08:00:00", "bets": [
        {"match": "Milan vs Venezia", "date": "2026-08-28", "commence_time": "2026-08-28T18:45:00Z"}]}))
    (tmp_path / "t30.json").write_text(json.dumps({"date": "2026-08-28", "matches": [], "bets": {}}))
    monkeypatch.setattr(L, "CANDIDATES_PATH", cand)
    monkeypatch.setattr(L, "T30_MARKER_PATH", tmp_path / "t30.json")
    now = datetime(2026, 8, 28, 19, 0, tzinfo=UTC)  # KO-30 passed 45 min ago
    m = L.get_metrics(journal=_journal([]), now=now)
    fun = [a for a in m["alerts"] if a["source"] == "t30_funnel"]
    assert len(fun) == 1 and "1 candidate" in fun[0]["message"] and "0 ticket" in fun[0]["message"]
    assert m["funnel"] == {"date": "2026-08-28", "candidates_n": 1, "tickets_n": 0, "filled_n": 0}
    # before the T-30 moment: no alert
    early = L.get_metrics(journal=_journal([]), now=datetime(2026, 8, 28, 17, 0, tzinfo=UTC))
    assert not [a for a in early["alerts"] if a["source"] == "t30_funnel"]


def test_health_issues_flow_into_alerts(monkeypatch, tmp_path, no_external_alerts):
    hp = tmp_path / "health.json"
    hp.write_text(json.dumps({"overall_status": "WARNING",
                              "issues": [["WARNING", "[health_check] Disk space low: 18% free"]]}))
    monkeypatch.setattr(L, "HEALTH_STATUS_PATH", hp)
    m = L.get_metrics(journal=_journal([]), now=NOW)
    h = [a for a in m["alerts"] if a["source"] == "health"]
    assert h == [{"level": "WARNING", "source": "health", "message": "[health_check] Disk space low: 18% free"}]


# ---------------------------------------------------------------------------
# Invariants: state.json drift is now caught
# ---------------------------------------------------------------------------

def test_verify_invariants_catches_state_json_drift(monkeypatch, tmp_path):
    rows = [_bet(1, "won", stake=100, odds=2.0)]
    j = _journal(rows)
    (tmp_path / "bet_journal.json").write_text(json.dumps(j))
    monkeypatch.setattr(L, "JOURNAL_PATH", tmp_path / "bet_journal.json")
    monkeypatch.setattr(L, "BANKROLL_PATH", tmp_path / "bankroll.json")
    monkeypatch.setattr(L, "HISTORY_PATH", tmp_path / "history.json")
    monkeypatch.setattr(L, "STATE_JSON_PATH", tmp_path / "state.json")
    import scripts.betting.bet_journal as J
    monkeypatch.setattr(J, "_JOURNAL_LOCK_PATH", tmp_path / ".lock")
    monkeypatch.setattr(L, "_LEDGER_LOCK_PATH", tmp_path / ".ledger.lock")
    L.rebuild_caches()
    assert L.verify_invariants()["ok"] is True
    (tmp_path / "state.json").write_text(json.dumps({"current_bankroll": 1015.35, "peak_bankroll": 1278.06}))
    v = L.verify_invariants()
    assert v["ok"] is False
    assert any("state.json" in x and "1278.06" in x for x in v["violations"]), v["violations"]
