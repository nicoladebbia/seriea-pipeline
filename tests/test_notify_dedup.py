#!/usr/bin/env python3
"""Notification volume discipline — pins the 2026-08-27 cleanup.

Measured from data/notification_history.jsonl (Aug 9–27): 200 sends in 18
days, 92% machine-status noise. The three mechanisms fixed, each pinned here:

1. Health issue identity was NOT number-stable: a changing count like
   "missing 18/32" minted a fresh issue key every 30-min monitor cycle and
   re-alerted each time — 77 health sends in 18 days.
2. notify_scheduler_run sent a one-line "✅ done" card for every routine
   success (~2/day forever).
3. notify_pipeline_run_with_picks sent "No value bets today." on dry days.

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


def test_dry_pipeline_card_is_silent(sent, monkeypatch, tmp_path):
    """No picks, slip fresh-empty → nothing to say → no send."""
    monkeypatch.setattr(N, "DATA_DIR", tmp_path)  # no betting_slip.json at all
    r = N.notify_pipeline_run_with_picks("morning", "success", duration_sec=300)
    assert r == {} and not sent
