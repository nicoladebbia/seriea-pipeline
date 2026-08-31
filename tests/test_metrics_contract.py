"""Cross-surface contract: every money surface renders ledger.get_metrics().

Phase 2 of the one-computation doctrine (2026-08-31): web endpoints, the
advisor, the Telegram /bankroll tool, notify's bankroll context, and the
Kelly-sizing loader must all agree with the payload — byte-for-byte on the
numbers, no local recomputes. Runs against the repo's live journal; skips
cleanly on a checkout without data.
"""

from pathlib import Path

import pytest

_JOURNAL = Path(__file__).resolve().parent.parent / "data" / "betting" / "bet_journal.json"

pytestmark = pytest.mark.skipif(
    not _JOURNAL.exists(), reason="no live bet_journal.json in this checkout"
)


@pytest.fixture(scope="module")
def payload():
    from scripts.betting import ledger

    return ledger.get_metrics(include_alerts=False)


def test_web_endpoints_render_the_payload(payload):
    from web.app import app

    c = app.test_client()
    b = c.get("/api/betting").get_json()["bankroll"]
    a = c.get("/api/analytics").get_json()["bankroll"]
    s = c.get("/api/system").get_json()["bankroll_health"]
    mb = payload["bankroll"]

    for surface, current, peak in (
        ("betting", b["current_balance"], b["peak_balance"]),
        ("analytics", a["current_balance"], a["peak_balance"]),
        ("system", s["current_bankroll"], s["peak_bankroll"]),
    ):
        assert current == mb["current"], surface
        assert peak == mb["peak"], surface

    assert b["win_rate"] == payload["record"]["win_rate_decisive"]
    assert b["roi"] == payload["roi"]["all_time_pct"]
    assert b["drawdown"] == round(mb["drawdown_pct"] / 100, 4)
    assert s["roi"] == payload["roi"]["all_time_pct"]


def test_performance_endpoint_carries_the_payload(payload):
    from web.app import app

    r = app.test_client().get("/api/performance").get_json()
    m = r.get("metrics") or {}
    assert m.get("record", {}).get("settled_n") == payload["record"]["settled_n"]
    assert m.get("roi", {}).get("all_time_pct") == payload["roi"]["all_time_pct"]


def test_advisor_and_bot_tool_agree(payload):
    import json

    from web import advisor

    bk = advisor._get_bankroll()
    assert bk["current_bankroll"] == payload["bankroll"]["current"]
    assert bk["bankroll_growth_pct"] == payload["bankroll"]["bankroll_growth_pct"]
    assert bk["roi_on_stake_pct"] == payload["roi"]["all_time_pct"]

    tool = json.loads(advisor._tool_get_bankroll_status({}))
    assert tool["current_bankroll"] == payload["bankroll"]["current"]
    assert tool["roi_pct"] == payload["roi"]["all_time_pct"]  # on stake, not growth
    assert tool["win_rate_decisive"] == payload["record"]["win_rate_decisive"]


def test_notify_context_agrees(payload):
    from scripts.pipeline import notify

    ctx = notify._get_bankroll_context()
    assert ctx["current"] == payload["bankroll"]["current"]
    assert ctx["growth_pct"] == payload["bankroll"]["bankroll_growth_pct"]
    assert ctx["roi_on_stake_pct"] == payload["roi"]["all_time_pct"]
    assert ctx["drawdown_pct"] == payload["bankroll"]["drawdown_pct"]


def test_kelly_base_is_available_balance(payload):
    from scripts.betting import bankroll_loader

    info = bankroll_loader.compute_current_bankroll()
    assert info["current_balance"] == payload["bankroll"]["current"]
    assert info["available_balance"] == payload["bankroll"]["available"]
    assert bankroll_loader.get_effective_bankroll() == payload["bankroll"]["available"]


def test_health_check_agrees(payload):
    from scripts.pipeline.health_check import check_betting_health

    j = check_betting_health()["details"]["journal"]
    assert j["roi_pct"] == payload["roi"]["all_time_pct"]
    assert j["settled"] == payload["record"]["settled_n"]
    assert j["clv_avg_pct"] == payload["clv"]["avg_pct"]


def test_roi_is_on_stake_never_growth(payload):
    """ROI and growth are different numbers with different names."""
    roi = payload["roi"]["all_time_pct"]
    growth = payload["bankroll"]["bankroll_growth_pct"]
    # With pending-free 186 settled bets, stake turnover >> initial bankroll,
    # so the two must differ unless profit is exactly zero.
    if payload["roi"]["all_time_profit"] != 0:
        assert roi != growth
